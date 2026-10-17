"""Deterministic, transparent ECL engine supporting CECL and IFRS 9.

Design goals: correctness and explainability over sophistication. Every
number the agents will ever quote comes from here (via the tool layer).

    ECL_i = SUM_y MarginalPD_i,y * LGD_i,y * EAD_i,y * DF_i,y     (annual steps)

* PD_i,y   = logistic(base_logit_i + macro_shift_m,s,y)  (scorecard + scenario)
* Marginal = S_i,y-1 * (1 - (1 - PD_i,y) ^ frac_i,y)       (frac handles stub years)
* LGD_i,y  = clip(base LGD_i + macro LGD shift_m,s,y)
* EAD_i,y  = (balance + CCF * undrawn) * amortisation profile
* DF_i,y   = (1 + EIR)^-(y + frac/2) for IFRS 9, 1 for CECL (configurable)
* CECL: lifetime ECL for all loans.  IFRS 9: Stage 1 12-month, Stage 2/3 lifetime.
* Scenario-weighted across baseline / adverse / severe_adverse.
* Allowance = SUM ECL + management overlays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.cecl.scenarios import scenario_annual_deviations
from src.utils import config

FRAMEWORKS = ("CECL", "IFRS9")


@dataclass
class EngineInputs:
    as_of_date: str
    loans: pd.DataFrame
    macro: pd.DataFrame                  # one vintage: scenario, forecast_quarter, variables
    weights: dict[str, float]
    overlays: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["portfolio", "amount"]))
    scenario_set_id: str = ""


@dataclass
class EngineResult:
    as_of_date: str
    framework: str
    loans: pd.DataFrame                  # loan-level results
    scenario_ecl: dict[str, float]       # unweighted ECL by scenario
    total_ecl: float                     # weighted, before overlays
    overlay_total: float
    allowance: float                     # total_ecl + overlays
    weights: dict[str, float]

    def summary(self) -> dict[str, Any]:
        l = self.loans
        ead = float(l["ead"].sum())
        return {
            "as_of_date": self.as_of_date,
            "framework": self.framework,
            "loan_count": int(len(l)),
            "total_balance": float(l["current_balance"].sum()),
            "total_ead": ead,
            "modeled_ecl": self.total_ecl,
            "overlay_total": self.overlay_total,
            "allowance": self.allowance,
            "allowance_rate_pct": 100.0 * self.allowance / float(l["current_balance"].sum()),
            "ead_weighted_pd_12m": float((l["pd_12m"] * l["ead"]).sum() / ead) if ead else 0.0,
            "ead_weighted_lgd": float((l["lgd"] * l["ead"]).sum() / ead) if ead else 0.0,
            "scenario_ecl": self.scenario_ecl,
            "scenario_weights": self.weights,
        }


def _lookup(keys: pd.Series, table: dict, what: str) -> pd.Series:
    """Vectorised dict lookup that fails loudly on unknown keys (like a dict index)."""
    unknown = set(pd.unique(keys)) - set(table)
    if unknown:
        raise KeyError(f"Unknown {what}: {sorted(map(str, unknown))}")
    return keys.map(table)


def _check_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {s: float(v) for s, v in weights.items()}
    if abs(sum(w.values()) - 1.0) > 1e-9:
        raise ValueError(f"Scenario weights must sum to 1, got {sum(w.values())}")
    return w


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _feature_contributions(loans: pd.DataFrame, model_col: str, model_params: dict) -> pd.DataFrame:
    """Linear contributions coef * (x - center) per feature, per loan."""
    cols: dict[str, np.ndarray] = {}
    n = len(loans)
    model_ids = loans[model_col].to_numpy()
    for model_id in pd.unique(model_ids):
        if model_id not in model_params:
            raise KeyError(f"Model '{model_id}' not found in config/models.yaml")
        mask = model_ids == model_id
        for feat, spec in model_params[model_id].get("features", {}).items():
            x = pd.to_numeric(loans.loc[mask, feat], errors="coerce").to_numpy(dtype=float) if feat in loans else np.full(mask.sum(), np.nan)
            contrib = spec["coef"] * (np.nan_to_num(x, nan=spec["center"]) - spec["center"])
            cols.setdefault(feat, np.zeros(n))[mask] = contrib
    return pd.DataFrame(cols, index=loans.index)


def pd_logit_components(loans: pd.DataFrame, models_cfg: dict | None = None,
                        method_cfg: dict | None = None) -> pd.DataFrame:
    """Decompose each loan's base (through-the-cycle) PD logit into components.

    Columns: intercept, one per scorecard feature, delinquency. Their row sum
    is the base logit. Used by the engine and, later, by the Credit Agent to
    explain *why* a PD moved.
    """
    models_cfg = models_cfg or config.models()
    method_cfg = method_cfg or config.methodology()
    pd_models = models_cfg["pd_models"]
    comps = _feature_contributions(loans, "pd_model_id", pd_models)
    comps.insert(0, "intercept", _lookup(loans["pd_model_id"], {m: v["intercept"] for m, v in pd_models.items()}, "PD model").to_numpy(dtype=float))
    dpd = loans["days_past_due"].fillna(0).to_numpy(dtype=float)
    dq = method_cfg["delinquency_logit"]
    comps["delinquency"] = np.where((dpd >= 30) & (dpd < 60), dq["30"], np.where((dpd >= 60) & (dpd < 90), dq["60"], 0.0))
    return comps


def base_pd(loans: pd.DataFrame, models_cfg: dict | None = None, method_cfg: dict | None = None) -> np.ndarray:
    """Annual base PD with no macro adjustment (defaults are not forced to 1)."""
    return _logistic(pd_logit_components(loans, models_cfg, method_cfg).sum(axis=1).to_numpy())


def _is_defaulted(loans: pd.DataFrame, default_dpd: int) -> np.ndarray:
    return (loans["default_flag"].fillna(0).to_numpy() == 1) | (loans["days_past_due"].fillna(0).to_numpy() >= default_dpd)


def _horizon_years(loans: pd.DataFrame, method_cfg: dict) -> np.ndarray:
    prods = method_cfg["products"]
    max_life = method_cfg["calculation"]["max_life_years"]
    rem = loans["remaining_term_months"].to_numpy(dtype=float) / 12.0
    amort = _lookup(loans["product_type"], {p: v["amortization"] for p, v in prods.items()}, "product_type").to_numpy()
    behav = _lookup(loans["product_type"], {p: v.get("behavioral_life_years", np.nan) for p, v in prods.items()},
                    "product_type").to_numpy(dtype=float)
    h = np.where(amort == "REVOLVING", np.fmin(behav, np.where(np.isnan(rem), np.inf, rem)), rem)
    return np.clip(np.nan_to_num(h, nan=1.0), 1.0 / 12.0, max_life)


def _ccf(loans: pd.DataFrame, method_cfg: dict, framework: str) -> np.ndarray:
    ead_cfg = method_cfg["ead_models"]

    def lookup(model_id: str, product: str) -> float:
        spec = ead_cfg[model_id][framework]
        return float(spec.get(product, 0.0)) if isinstance(spec, dict) else float(spec)

    keys = loans["ead_model_id"].astype(str) + "|" + loans["product_type"].astype(str)
    table = {k: lookup(*k.split("|", 1)) for k in pd.unique(keys)}
    return keys.map(table).to_numpy(dtype=float)


def _model_matrix(model_ids: np.ndarray, params: dict, key: str, variables: list[str]):
    """Return (unique ids, index per loan, beta matrix [n_models, n_vars])."""
    uniq, idx = np.unique(model_ids, return_inverse=True)
    beta = np.array([[params[m].get(key, {}).get(v, 0.0) for v in variables] for m in uniq], dtype=float)
    return uniq, idx, beta


class ECLEngine:
    def __init__(self, method_cfg: dict | None = None, models_cfg: dict | None = None):
        self.method = method_cfg or config.methodology()
        self.models = models_cfg or config.models()

    # ------------------------------------------------------------------ #
    def run(self, inputs: EngineInputs, framework: str = "CECL") -> EngineResult:
        if framework not in FRAMEWORKS:
            raise ValueError(f"framework must be one of {FRAMEWORKS}")
        loans = inputs.loans.reset_index(drop=True)
        n = len(loans)
        weights = _check_weights(inputs.weights)
        st = self._prepare(loans, framework)
        base_logit, defaulted = st["base_logit"], st["defaulted"]
        h = st["h"]

        devs = self._scenario_devs(inputs.macro, st["n_years"], weights)

        # --- per-scenario lifetime projection ------------------------------ #
        per_s = {s: self._project(st, devs[s]) for s in weights}

        wsum = lambda key: sum(weights[s] * per_s[s][key] for s in weights)  # noqa: E731
        pd_12m_w = wsum("pd_12m")
        pd_annual_w = wsum("pd_annual_y1")

        # --- IFRS 9 staging (computed for both frameworks; used by IFRS 9) -- #
        # Simplification: the SICR PD test compares through-the-cycle base PD
        # with origination base PD (both ex-macro), so staging reflects
        # borrower credit quality; macro flows through PD levels.
        stage, use_life = st["stage"], st["use_life"]

        ecl_by_s = {s: np.where(use_life, per_s[s]["ecl_life"], per_s[s]["ecl_12m"]) for s in weights}
        ecl = sum(weights[s] * ecl_by_s[s] for s in weights)

        out = pd.DataFrame({
            "as_of_date": inputs.as_of_date,
            "framework": framework,
            "loan_id": loans["loan_id"].to_numpy(),
            "counterparty_id": loans["counterparty_id"].to_numpy(),
            "portfolio": loans["portfolio"].to_numpy(),
            "product_type": loans["product_type"].to_numpy(),
            "current_balance": loans["current_balance"].to_numpy(float),
            "undrawn_amount": loans["undrawn_amount"].fillna(0).to_numpy(float),
            "ead": st["ead0"],
            "pd_base_annual": _logistic(base_logit),
            "pd_12m": pd_12m_w,
            "pd_annual_y1": pd_annual_w,
            "lifetime_pd": wsum("lifetime_pd"),
            "lgd": wsum("lgd_y1"),
            "horizon_years": h,
            "ifrs9_stage": stage,
            "measurement": np.where(use_life, "LIFETIME", "12M"),
            "defaulted": defaulted,
            "ecl": ecl,
            **{f"ecl_{s}": ecl_by_s[s] for s in weights},
            "pd_model_id": loans["pd_model_id"].to_numpy(),
            "lgd_model_id": loans["lgd_model_id"].to_numpy(),
            "ead_model_id": loans["ead_model_id"].to_numpy(),
            "scenario_set_id": inputs.scenario_set_id,
        })

        overlay_total = float(inputs.overlays["amount"].sum()) if len(inputs.overlays) else 0.0
        total = float(ecl.sum())
        return EngineResult(
            as_of_date=inputs.as_of_date, framework=framework, loans=out,
            scenario_ecl={s: float(ecl_by_s[s].sum()) for s in weights},
            total_ecl=total, overlay_total=overlay_total, allowance=total + overlay_total,
            weights=weights,
        )

    # ------------------------------------------------------------------ #
    # Static (macro-independent) per-loan quantities. Split out so macro-only
    # analyses (variable Shapley, scenario runs, sensitivities) can re-project
    # many macro paths without recomputing scorecards and exposure profiles.
    def _prepare(self, loans: pd.DataFrame, framework: str) -> dict[str, Any]:
        m = self.method
        n = len(loans)
        variables = m["macro"]["variables"]
        prods = m["products"]
        base_logit = pd_logit_components(loans, self.models, m).sum(axis=1).to_numpy()
        defaulted = _is_defaulted(loans, m["calculation"]["default_dpd"])
        h = _horizon_years(loans, m)
        n_years = int(np.ceil(h.max())) if n else 1
        yrs = np.arange(n_years, dtype=float)
        frac = np.clip(h[:, None] - yrs[None, :], 0.0, 1.0)                    # (n, Y)

        amort = _lookup(loans["product_type"], {p: v["amortization"] for p, v in prods.items()}, "product_type").to_numpy()
        ead0 = loans["current_balance"].fillna(0).to_numpy(float) + _ccf(loans, m, framework) * loans["undrawn_amount"].fillna(0).to_numpy(float)
        mid = yrs[None, :] + frac / 2.0
        amort_factor = np.where((amort == "AMORTIZING")[:, None], np.clip(1.0 - mid / h[:, None], 0.0, 1.0), 1.0)
        ead_y = ead0[:, None] * amort_factor * (frac > 0)

        if m["calculation"]["discount"][framework]:
            eir = loans["interest_rate"].fillna(0).to_numpy(float)
            df_y = np.power(1.0 + eir[:, None], -mid)
        else:
            df_y = np.ones_like(frac)

        lgd_params = self.models["lgd_models"]
        lgd0 = _feature_contributions(loans, "lgd_model_id", lgd_params).sum(axis=1).to_numpy() if n else np.zeros(0)
        lgd0 = lgd0 + _lookup(loans["lgd_model_id"], {m: v["base"] for m, v in lgd_params.items()}, "LGD model").to_numpy(float)
        lgd_floor = _lookup(loans["lgd_model_id"], {m: v["floor"] for m, v in lgd_params.items()}, "LGD model").to_numpy(float)
        lgd_cap = _lookup(loans["lgd_model_id"], {m: v["cap"] for m, v in lgd_params.items()}, "LGD model").to_numpy(float)

        _, pd_idx, pd_beta = _model_matrix(loans["pd_model_id"].to_numpy(), self.models["pd_models"], "macro_sensitivity", variables)
        _, lgd_idx, lgd_beta = _model_matrix(loans["lgd_model_id"].to_numpy(), lgd_params, "macro_sensitivity", variables)

        stage = self._ifrs9_stage(loans, _logistic(base_logit), defaulted)
        use_life = stage >= 2 if framework == "IFRS9" else np.ones(n, dtype=bool)
        return {"n": n, "base_logit": base_logit, "defaulted": defaulted, "h": h, "n_years": n_years,
                "frac": frac, "ead0": ead0, "ead_y": ead_y, "df_y": df_y, "lgd0": lgd0,
                "lgd_floor": lgd_floor, "lgd_cap": lgd_cap, "pd_idx": pd_idx, "pd_beta": pd_beta,
                "lgd_idx": lgd_idx, "lgd_beta": lgd_beta, "stage": stage, "use_life": use_life}

    def _scenario_devs(self, macro: pd.DataFrame, n_years: int, weights: dict[str, float]) -> dict[str, np.ndarray]:
        devs = scenario_annual_deviations(macro, self.method["macro"], n_years)
        missing = set(weights) - set(devs)
        if missing:
            raise ValueError(f"Macro paths missing for scenarios {missing}")
        return devs

    @staticmethod
    def _project(st: dict[str, Any], dev: np.ndarray) -> dict[str, np.ndarray]:
        """Lifetime projection for one scenario path (dev: (Y, V) deviations)."""
        n = st["n"]
        pd_shift = (st["pd_beta"] @ dev.T)[st["pd_idx"]]                       # (n, Y)
        lgd_shift = (st["lgd_beta"] @ dev.T)[st["lgd_idx"]]
        pd_y = _logistic(st["base_logit"][:, None] + pd_shift)
        pd_y[st["defaulted"], 0] = 1.0
        surv_frac = np.power(1.0 - pd_y, st["frac"])
        surv = np.cumprod(surv_frac, axis=1)
        surv_prev = np.hstack([np.ones((n, 1)), surv[:, :-1]])
        marginal = surv_prev * (1.0 - surv_frac)
        lgd_y = np.clip(st["lgd0"][:, None] + lgd_shift, st["lgd_floor"][:, None], st["lgd_cap"][:, None])
        contrib = marginal * lgd_y * st["ead_y"] * st["df_y"]
        return {
            "ecl_life": contrib.sum(axis=1),
            "ecl_12m": contrib[:, 0],
            "pd_12m": marginal[:, 0],
            "pd_annual_y1": pd_y[:, 0],
            "lifetime_pd": 1.0 - surv[:, -1],
            "lgd_y1": lgd_y[:, 0],
        }

    def macro_sweep(self, inputs: EngineInputs, macros: list[pd.DataFrame], framework: str = "CECL",
                    weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
        """Evaluate the same loans under several macro vintages (or hybrids).

        Returns, per macro frame, weighted loan-level ECL (aligned with
        ``inputs.loans`` row order), unweighted scenario totals and the
        allowance (modeled ECL + overlays). Numerically identical to calling
        ``run`` once per frame, but the scorecards and exposure profiles are
        computed only once.
        """
        if framework not in FRAMEWORKS:
            raise ValueError(f"framework must be one of {FRAMEWORKS}")
        loans = inputs.loans.reset_index(drop=True)
        w = _check_weights(weights or inputs.weights)
        st = self._prepare(loans, framework)
        overlay_total = float(inputs.overlays["amount"].sum()) if len(inputs.overlays) else 0.0
        out = []
        for macro in macros:
            devs = self._scenario_devs(macro, st["n_years"], w)
            by_s = {}
            for s in w:
                p = self._project(st, devs[s])
                by_s[s] = np.where(st["use_life"], p["ecl_life"], p["ecl_12m"])
            ecl = sum(w[s] * by_s[s] for s in w)
            total = float(ecl.sum())
            out.append({"loan_ecl": ecl, "scenario_ecl": {s: float(v.sum()) for s, v in by_s.items()},
                        "modeled_ecl": total, "allowance": total + overlay_total})
        return out

    # ------------------------------------------------------------------ #
    def _ifrs9_stage(self, loans: pd.DataFrame, pd_annual: np.ndarray, defaulted: np.ndarray) -> np.ndarray:
        cfg = self.method["ifrs9"]
        orig_pd = loans["origination_pd"].fillna(np.nan).to_numpy(float)
        ratio = np.where(orig_pd > 0, pd_annual / np.where(orig_pd > 0, orig_pd, 1.0), 1.0)
        notch = loans["rating_notch"].to_numpy(float) if "rating_notch" in loans else np.full(len(loans), np.nan)
        onotch = loans["origination_rating_notch"].to_numpy(float) if "origination_rating_notch" in loans else np.full(len(loans), np.nan)
        downgrade = np.nan_to_num(notch - onotch, nan=0.0)
        dpd = loans["days_past_due"].fillna(0).to_numpy(float)
        watch = loans["watchlist_flag"].fillna(0).to_numpy(float) == 1
        s2 = (dpd >= cfg["sicr_dpd"]) | (ratio >= cfg["sicr_pd_ratio"]) | (downgrade >= cfg["sicr_rating_downgrade"])
        if cfg.get("watchlist_is_sicr", True):
            s2 |= watch
        return np.where(defaulted, 3, np.where(s2, 2, 1))
