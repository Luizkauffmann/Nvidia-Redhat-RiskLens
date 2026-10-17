"""Deterministic CECL engine (ASC 326 lifetime expected credit loss).

Transparent by design: vectorised numpy over loans x scenarios x years, no
hidden state. See config/methodology.yaml for the formulas. Every number the
agents report traces back to `run_cecl`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from risklens import config

SCENARIOS = ("baseline", "adverse", "severe_adverse")


@dataclass
class RunInputs:
    """Everything one allowance calculation needs."""
    as_of_date: str
    loans: pd.DataFrame
    macro_dev: np.ndarray            # (scenario, variable, quarter): deviation from long-run anchor
    weights: dict[str, float]        # scenario -> weight
    overlays: list[dict] = field(default_factory=list)


@dataclass
class RunResult:
    as_of_date: str
    loan_id: np.ndarray
    portfolio: np.ndarray
    ecl: np.ndarray                  # scenario-weighted lifetime ECL per loan
    ecl_by_scenario: np.ndarray      # (scenario, loan)
    pd_12m: np.ndarray               # weighted first-year PD (with macro)
    pd_12m_base: np.ndarray          # first-year PD from the scorecard alone (no macro)
    pd_lifetime: np.ndarray          # weighted cumulative lifetime default probability
    lgd: np.ndarray                  # weighted first-year LGD
    ead: np.ndarray                  # current exposure at default
    overlay_total: float
    overlay_by_portfolio: dict[str, float]

    @property
    def modeled_ecl(self) -> float:
        return float(self.ecl.sum())

    @property
    def allowance(self) -> float:
        return self.modeled_ecl + self.overlay_total

    def loan_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "loan_id": self.loan_id, "portfolio": self.portfolio, "ecl": self.ecl,
            "pd_12m": self.pd_12m, "pd_12m_base": self.pd_12m_base, "pd_lifetime": self.pd_lifetime,
            "lgd": self.lgd, "ead": self.ead,
        })

    def by_portfolio(self) -> dict[str, float]:
        s = pd.Series(self.ecl).groupby(self.portfolio).sum()
        out = {p: float(s.get(p, 0.0)) + self.overlay_by_portfolio.get(p, 0.0)
               for p in config.methodology()["portfolios"]}
        return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def annual_deviation(macro_dev: np.ndarray, n_years: int) -> np.ndarray:
    """Quarterly deviations (S, V, Q) -> annual means (S, V, Y); zero after the forecast horizon."""
    s, v, q = macro_dev.shape
    yrs = int(np.ceil(q / 4))
    padded = np.zeros((s, v, yrs * 4))
    padded[:, :, :q] = macro_dev
    annual = padded.reshape(s, v, yrs, 4).mean(axis=3)
    out = np.zeros((s, v, n_years))
    out[:, :, : min(yrs, n_years)] = annual[:, :, :n_years]
    return out


def base_logit(loans: pd.DataFrame) -> np.ndarray:
    """PD scorecard logit per loan, using the parameters of the loan's pd_model_id."""
    meth, mdl = config.methodology(), config.models()["pd_models"]
    n = len(loans)
    out = np.full(n, np.nan)
    model_ids = loans["pd_model_id"].to_numpy()
    fico = loans["fico"].fillna(700).to_numpy(float)
    dti = loans["dti"].fillna(0.35).to_numpy(float)
    util = loans["utilization"].fillna(0.3).to_numpy(float)
    notch = loans["rating_notch"].fillna(10).to_numpy(float)
    dscr = loans["dscr"].fillna(1.35).to_numpy(float)
    lev = loans["leverage"].fillna(3.0).to_numpy(float)
    ltv = loans["ltv"].to_numpy(float)
    for mid in np.unique(model_ids):
        if mid not in mdl:
            raise KeyError(f"Unknown PD model {mid}")
        p = mdl[mid]
        m = model_ids == mid
        if p["type"] == "retail":
            x = (p["intercept"] + p["fico"] * (fico[m] - 700) / 100 + p["dti"] * (dti[m] - 0.35)
                 + p["ltv"] * (np.nan_to_num(ltv[m], nan=0.8) - 0.8) + p["util"] * (util[m] - 0.3))
        else:
            x = (p["intercept"] + p["rating"] * (notch[m] - 10) + p["dscr"] * (dscr[m] - 1.35)
                 + p["ltv"] * (np.nan_to_num(ltv[m], nan=0.65) - 0.65) + p["leverage"] * (lev[m] - 3.0))
        out[m] = x
    calc = meth["calculation"]
    dpd = loans["days_past_due"].fillna(0).to_numpy(float)
    out += np.where((dpd >= 30) & (dpd < 60), calc["dpd_logit_addon"]["30"], 0.0)
    out += np.where((dpd >= 60) & (dpd < 90), calc["dpd_logit_addon"]["60"], 0.0)
    out += np.where(loans["watchlist_flag"].fillna(False).to_numpy(bool), calc["watchlist_logit_addon"], 0.0)
    return out


def base_lgd(loans: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base LGD per loan plus the floor/cap of its LGD model."""
    mdl = config.models()["lgd_models"]
    n = len(loans)
    lgd, lo, hi = np.zeros(n), np.zeros(n), np.ones(n)
    ids = loans["lgd_model_id"].to_numpy()
    ltv = loans["ltv"].fillna(0).to_numpy(float)
    for mid in np.unique(ids):
        p = mdl[mid]
        m = ids == mid
        lgd[m] = p["base"] + p["ltv_slope"] * np.maximum(ltv[m] - p["ltv_floor"], 0)
        lo[m], hi[m] = p["floor"], p["cap"]
    return np.clip(lgd, lo, hi), lo, hi


def current_ead(loans: pd.DataFrame) -> np.ndarray:
    ead_models = config.models()["ead_models"]
    ccf = np.zeros(len(loans))
    ids = loans["ead_model_id"].to_numpy()
    prod = loans["product_type"].to_numpy()
    for mid in np.unique(ids):
        c = ead_models[mid]["ccf"]
        m = ids == mid
        if isinstance(c, dict):
            for p, v in c.items():
                ccf[m & (prod == p)] = v
        else:
            ccf[m] = c
    return loans["current_balance"].to_numpy(float) + ccf * loans["undrawn_amount"].fillna(0).to_numpy(float)


def run_cecl(inp: RunInputs) -> RunResult:
    meth = config.methodology()
    prods = meth["products"]
    loans = inp.loans.reset_index(drop=True)
    n = len(loans)
    max_life = meth["calculation"]["max_life_years"]

    product = loans["product_type"].to_numpy()
    profile = np.array([prods[p]["profile"] for p in product])
    prepay = np.array([prods[p]["prepay"] for p in product])
    life = loans["remaining_term_months"].to_numpy(float) / 12.0
    rev_life = np.array([prods[p].get("life_years", 0) for p in product], float)
    life = np.where(profile == "revolving", rev_life, life)
    life = np.clip(life, 1 / 12, max_life)
    n_years = int(np.ceil(life.max())) if n else 1
    yrs = np.arange(n_years)
    frac = np.clip(life[:, None] - yrs[None, :], 0, 1)                       # (N, Y)

    ead0 = current_ead(loans)
    t_mid = yrs[None, :] + 0.5 * frac
    amort = np.clip(1 - t_mid / life[:, None], 0, 1)
    shape = np.where((profile == "amortising")[:, None], amort, 1.0)
    ead = ead0[:, None] * shape * (1 - prepay[:, None]) ** yrs[None, :]       # (N, Y)

    # Macro shift per portfolio, scenario, year.
    variables = meth["macro"]["variables"]
    dev = annual_deviation(inp.macro_dev, n_years)                            # (S, V, Y)
    portfolio = loans["portfolio"].to_numpy()
    portfolios = meth["portfolios"]
    pidx = np.array([portfolios.index(p) for p in portfolio])
    beta = np.array([[meth["macro"]["pd_beta"][p][v] for v in variables] for p in portfolios])  # (P, V)
    shift_p = np.einsum("pv,svy->psy", beta, dev)                           # (P, S, Y)
    shift = shift_p[pidx]                                                    # (N, S, Y)

    # LGD collateral sensitivity.
    lgd0, lgd_lo, lgd_hi = base_lgd(loans)
    kappa = np.zeros(len(portfolios))
    var_idx = np.zeros(len(portfolios), int)
    for p, spec in meth["macro"].get("lgd_collateral", {}).items():
        kappa[portfolios.index(p)] = spec["kappa"]
        var_idx[portfolios.index(p)] = variables.index(spec["variable"])
    coll_dev = dev[:, var_idx, :].transpose(1, 0, 2)                         # (P, S, Y)
    lgd_add = (kappa[:, None, None] * -coll_dev)[pidx]                       # (N, S, Y)
    lgd = np.clip(lgd0[:, None, None] + lgd_add, lgd_lo[:, None, None], lgd_hi[:, None, None])

    logit = base_logit(loans)
    pd_y = _sigmoid(logit[:, None, None] + shift)                             # (N, S, Y)
    p_frac = 1 - (1 - pd_y) ** frac[:, None, :]
    surv_prev = np.cumprod(np.concatenate([np.ones((n, len(SCENARIOS), 1)), 1 - p_frac[:, :, :-1]], axis=2), axis=2)
    marg = surv_prev * p_frac
    ecl_s = (marg * lgd * ead[:, None, :]).sum(axis=2)                       # (N, S)

    defaulted = (loans["days_past_due"].fillna(0).to_numpy() >= meth["calculation"]["default_dpd"]) | \
        loans["default_flag"].fillna(False).to_numpy(bool)
    ecl_s[defaulted] = (lgd[defaulted, :, 0] * ead0[defaulted, None])

    w = np.array([inp.weights[s] for s in SCENARIOS])
    ecl = ecl_s @ w
    pd12 = pd_y[:, :, 0] @ w
    pd12[defaulted] = 1.0
    pd_life = marg.sum(axis=2) @ w
    pd_life[defaulted] = 1.0

    ov_by = {}
    for o in inp.overlays:
        ov_by[o["portfolio"]] = ov_by.get(o["portfolio"], 0.0) + float(o["amount"])

    return RunResult(
        as_of_date=inp.as_of_date, loan_id=loans["loan_id"].to_numpy(), portfolio=portfolio,
        ecl=ecl, ecl_by_scenario=ecl_s.T, pd_12m=pd12, pd_12m_base=_sigmoid(logit),
        pd_lifetime=pd_life, lgd=lgd[:, :, 0] @ w, ead=ead0,
        overlay_total=float(sum(ov_by.values())), overlay_by_portfolio=ov_by,
    )
