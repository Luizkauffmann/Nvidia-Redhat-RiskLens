"""Seeded synthetic data generator for the July -> August 2026 CECL story.

Produces a loan tape per reporting date, counterparties, macro scenarios,
scenario weights, management overlays, model metadata, a run manifest per
date, and a hidden ground_truth.json describing what was injected.

Everything is driven by config/story.yaml and the seed in config/settings.yaml,
so `python -m scripts.generate_data` reproduces the published files exactly.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from risklens import config
from risklens.cecl.engine import SCENARIOS, RunInputs, run_cecl

STATES = ["NC", "SC", "VA", "GA", "TN", "FL", "TX", "NY", "CA", "IL", "OH", "PA"]
STATE_P = np.array([0.22, 0.08, 0.09, 0.09, 0.06, 0.1, 0.1, 0.06, 0.07, 0.05, 0.04, 0.04])
INDUSTRIES = ["Manufacturing", "Wholesale Trade", "Retail Trade", "Healthcare", "Construction",
              "Transportation", "Professional Services", "Hospitality", "Agriculture", "Technology"]
CRE_TYPES = ["Office", "Retail", "Multifamily", "Industrial", "Hotel"]
FIRST = ["James", "Maria", "Robert", "Linda", "Michael", "Ana", "David", "Grace", "Daniel", "Sofia",
         "Kevin", "Priya", "Luis", "Emily", "Chen", "Aisha", "Marcus", "Olivia", "Tomas", "Hannah"]
LAST = ["Smith", "Johnson", "Garcia", "Brown", "Lee", "Martinez", "Davis", "Nguyen", "Patel", "Wilson",
        "Lopez", "Clark", "Walker", "Young", "Silva", "Hall", "King", "Wright", "Scott", "Green"]
CO_A = ["Summit", "Blue Ridge", "Cardinal", "Harbor", "Pinecrest", "Oak Hollow", "Triangle", "Riverbend",
        "Ironwood", "Crescent", "Granite", "Magnolia", "Northgate", "Lakeshore", "Redstone", "Atlas"]
CO_B = ["Holdings", "Partners", "Industries", "Logistics", "Properties", "Realty", "Manufacturing",
        "Health", "Foods", "Systems", "Development", "Capital", "Supply", "Group"]
DPD_VALUES = np.array([0, 15, 45, 75, 120])


def _bucket(dpd: np.ndarray) -> np.ndarray:
    return np.select([dpd >= 90, dpd >= 60, dpd >= 30, dpd >= 1], ["90+", "60-89", "30-59", "1-29"], "Current")


def _rating(notch: np.ndarray) -> np.ndarray:
    scale = np.array(config.methodology()["rating_scale"], dtype=object)
    out = np.full(len(notch), None, dtype=object)
    ok = ~np.isnan(notch)
    out[ok] = scale[np.clip(notch[ok].astype(int), 1, 20) - 1]
    return out


def macro_table() -> pd.DataFrame:
    """Macro scenario dataset: absolute levels per vintage, scenario and forecast quarter."""
    meth, st = config.methodology(), config.story()
    m = meth["macro"]
    shape = np.array(m["path_shape"])
    rows = []
    for as_of, scen in st["macro_peak_deviation"].items():
        for s in SCENARIOS:
            for q in range(m["forecast_quarters"]):
                row = {"as_of_date": as_of, "scenario": s, "scenario_weight": st["scenario_weights"][as_of][s],
                       "forecast_quarter": q + 1}
                for v in m["variables"]:
                    row[v] = round(m["long_run_anchor"][v] + scen[s][v] * shape[q], 4)
                rows.append(row)
    return pd.DataFrame(rows)


def _company_name(rng) -> str:
    return f"{rng.choice(CO_A)} {rng.choice(CO_B)} {rng.choice(['LLC', 'Inc.', 'LP', 'Co.'])}"


class _Builder:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n_cp = 0
        self.n_loan = 0
        self.counterparties: list[dict] = []

    def cp_id(self) -> str:
        self.n_cp += 1
        return f"CP_{self.n_cp:06d}"

    def loan_ids(self, k: int) -> list[str]:
        ids = [f"LN_{i:07d}" for i in range(self.n_loan + 1, self.n_loan + k + 1)]
        self.n_loan += k
        return ids

    # ---- counterparties ---------------------------------------------------
    def retail_cp(self, fico: float) -> str:
        r = self.rng
        cid = self.cp_id()
        self.counterparties.append({
            "counterparty_id": cid, "counterparty_type": "RETAIL",
            "customer_name": f"{r.choice(FIRST)} {r.choice(LAST)}",
            "industry": None, "state": r.choice(STATES, p=STATE_P),
            "annual_revenue": None, "total_assets": None, "employees": None,
            "fico": int(fico), "income": float(round(r.lognormal(np.log(72000), 0.45), -2)),
            "sp_rating": None, "moodys_rating": None, "economic_group_id": None,
        })
        return cid

    def commercial_cp(self, industry: str, notch: float, size: float, group: str | None = None,
                      name: str | None = None) -> str:
        r = self.rng
        cid = self.cp_id()
        revenue = float(round(size * r.uniform(3, 8), -3))
        rated = revenue > 150e6
        agency = _rating(np.array([notch + r.integers(-1, 2)], float))[0] if rated else None
        moodys_map = {"AAA": "Aaa", "AA+": "Aa1", "AA": "Aa2", "AA-": "Aa3", "A+": "A1", "A": "A2", "A-": "A3",
                      "BBB+": "Baa1", "BBB": "Baa2", "BBB-": "Baa3", "BB+": "Ba1", "BB": "Ba2", "BB-": "Ba3",
                      "B+": "B1", "B": "B2", "B-": "B3"}
        self.counterparties.append({
            "counterparty_id": cid, "counterparty_type": "COMMERCIAL",
            "customer_name": name or _company_name(r), "industry": industry,
            "state": r.choice(STATES, p=STATE_P), "annual_revenue": revenue,
            "total_assets": float(round(revenue * r.uniform(0.8, 2.5), -3)),
            "employees": int(max(5, revenue / r.uniform(150e3, 400e3))),
            "fico": None, "income": None, "sp_rating": agency,
            "moodys_rating": moodys_map.get(agency) if agency else None,
            "economic_group_id": group,
        })
        return cid


def _retail_loans(b: _Builder, product: str, k: int, as_of: date, new: bool) -> pd.DataFrame:
    r = b.rng
    spec = {
        "Mortgage":      dict(bal=(220e3, 0.5), term=(60, 340), fico=(745, 45), ltv=(0.45, 0.95), rate=(4.0, 7.0)),
        "Auto Loan":     dict(bal=(22e3, 0.5), term=(6, 66), fico=(705, 60), ltv=(0.60, 1.15), rate=(5.0, 11.0)),
        "Personal Loan": dict(bal=(10e3, 0.6), term=(6, 54), fico=(690, 60), ltv=None, rate=(9.0, 18.0)),
        "Credit Card":   dict(bal=(8e3, 0.6), term=None, fico=(710, 65), ltv=None, rate=(18.0, 28.0)),
    }[product]
    fico = np.clip(r.normal(*spec["fico"], k) + (15 if new else 0), 560, 850).round()
    dti = np.clip(r.beta(4, 8, k), 0.05, 0.65).round(3)
    if product == "Credit Card":
        limit = np.clip(r.lognormal(np.log(spec["bal"][0]), spec["bal"][1], k), 500, 50000).round(-2)
        util = np.clip(r.beta(2, 5, k), 0.0, 1.0).round(3)
        bal = (limit * util).round(2)
        undrawn = (limit - bal).round(2)
        term = np.full(k, 36)
    else:
        bal = r.lognormal(np.log(spec["bal"][0]), spec["bal"][1], k).round(2)
        undrawn = np.zeros(k)
        util = np.full(k, np.nan)
        lo, hi = spec["term"]
        term = (np.full(k, hi) if new else r.integers(lo, hi + 1, k))
    ltv = r.uniform(*spec["ltv"], k).round(3) if spec["ltv"] else np.full(k, np.nan)
    if new:
        dpd = np.zeros(k)
    else:
        dpd = r.choice(DPD_VALUES, k, p=config.story()["background"]["retail_dpd_mix"]).astype(float)
    months_on_book = np.zeros(k, int) if new else r.integers(3, 60, k)
    orig = [pd.Timestamp(as_of) - pd.DateOffset(months=int(m)) for m in months_on_book]
    mat = [pd.Timestamp(as_of) + pd.DateOffset(months=int(t)) for t in term]
    cps = [b.retail_cp(f) for f in fico]
    return pd.DataFrame({
        "loan_id": b.loan_ids(k), "counterparty_id": cps, "product_type": product,
        "industry": None, "property_type": None,
        "state": [c["state"] for c in b.counterparties[-k:]],
        "origination_date": orig, "maturity_date": mat, "remaining_term_months": term.astype(int),
        "current_balance": bal, "undrawn_amount": undrawn, "utilization": util,
        "interest_rate": r.uniform(*spec["rate"], k).round(2), "property_value": (bal / ltv).round(2) if spec["ltv"] else np.nan,
        "fico": fico, "dti": dti, "ltv": ltv, "dscr": np.nan, "leverage": np.nan,
        "rating_notch": np.nan, "watchlist_flag": False, "days_past_due": dpd,
    })


def _commercial_loans(b: _Builder, product: str, k: int, as_of: date, new: bool,
                      cp_pool: list[str] | None = None, balances: np.ndarray | None = None) -> pd.DataFrame:
    r = b.rng
    spec = {
        "Commercial Term Loan":      dict(bal=(420e3, 0.8), term=(6, 84), rate=(6.0, 9.5)),
        "Revolving Credit Facility": dict(bal=(0.9e6, 0.7), term=None, rate=(6.5, 9.0)),
        "Equipment Lease":           dict(bal=(200e3, 0.7), term=(6, 60), rate=(6.0, 9.0)),
        "Commercial Real Estate":    dict(bal=(1.8e6, 0.75), term=(6, 96), rate=(5.5, 8.5)),
    }[product]
    is_cre = product == "Commercial Real Estate"
    notch = np.clip(r.normal(10.0 if new else 10.6, 1.6, k).round(), 4, 17)
    dscr = np.clip(r.normal(1.55 if new else 1.5, 0.25, k), 0.9, 2.6).round(3)
    if balances is None:
        bal = np.clip(r.lognormal(np.log(spec["bal"][0]), spec["bal"][1], k), 50e3, 25e6).round(2)
    else:
        bal = balances
    if product == "Revolving Credit Facility":
        commit = bal
        util = np.clip(r.beta(3, 4, k), 0.05, 0.95).round(3)
        bal = (commit * util).round(2)
        undrawn = (commit - bal).round(2)
        term = np.full(k, 36)
    else:
        undrawn, util = np.zeros(k), np.full(k, np.nan)
        lo, hi = spec["term"]
        term = np.full(k, hi) if new else r.integers(lo, hi + 1, k)
    ltv = r.uniform(0.45, 0.72, k).round(3) if is_cre else np.full(k, np.nan)
    lev = np.full(k, np.nan) if is_cre else np.clip(r.normal(3.0, 0.9, k), 0.5, 7).round(2)
    dpd = np.zeros(k) if new else r.choice(DPD_VALUES, k, p=config.story()["background"]["commercial_dpd_mix"]).astype(float)
    if cp_pool is None:
        industries = ["Real Estate"] * k if is_cre else list(r.choice(INDUSTRIES, k))
        cps = [b.commercial_cp(ind, n, s) for ind, n, s in zip(industries, notch, bal + undrawn)]
    else:
        cps = cp_pool
    cp_lookup = {c["counterparty_id"]: c for c in b.counterparties}
    months_on_book = np.zeros(k, int) if new else r.integers(3, 72, k)
    return pd.DataFrame({
        "loan_id": b.loan_ids(k), "counterparty_id": cps, "product_type": product,
        "industry": [cp_lookup[c]["industry"] for c in cps],
        "property_type": list(r.choice(CRE_TYPES, k, p=[0.25, 0.2, 0.3, 0.17, 0.08])) if is_cre else None,
        "state": [cp_lookup[c]["state"] for c in cps],
        "origination_date": [pd.Timestamp(as_of) - pd.DateOffset(months=int(m)) for m in months_on_book],
        "maturity_date": [pd.Timestamp(as_of) + pd.DateOffset(months=int(t)) for t in term],
        "remaining_term_months": term.astype(int), "current_balance": bal, "undrawn_amount": undrawn,
        "utilization": util, "interest_rate": r.uniform(*spec["rate"], k).round(2),
        "property_value": (bal / ltv).round(2) if is_cre else np.nan,
        "fico": np.nan, "dti": np.nan, "ltv": ltv, "dscr": dscr, "leverage": lev,
        "rating_notch": notch, "watchlist_flag": False, "days_past_due": dpd,
    })


def _finalise(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    meth = config.methodology()
    prods = meth["products"]
    df = df.copy()
    df.insert(0, "as_of_date", as_of)
    df["portfolio"] = df["product_type"].map(lambda p: prods[p]["portfolio"])
    df["segment"] = df["product_type"].map(lambda p: prods[p]["segment"])
    df["internal_rating"] = _rating(df["rating_notch"].to_numpy(float))
    df["delinquency_bucket"] = _bucket(df["days_past_due"].to_numpy(float))
    df["default_flag"] = df["days_past_due"] >= meth["calculation"]["default_dpd"]
    for col, key in (("pd_model_id", "pd_model"), ("lgd_model_id", "lgd_model"), ("ead_model_id", "ead_model")):
        df[col] = df["product_type"].map(lambda p, k=key: prods[p][k])
    df["scenario_set_id"] = f"MACRO_{as_of[:7].replace('-', '')}"
    df["watchlist_flag"] = df["watchlist_flag"].astype(bool)
    return df


def build_july(b: _Builder) -> tuple[pd.DataFrame, list[str]]:
    st = config.story()
    as_of = date(2026, 7, 31)
    r = b.rng
    parts = []
    for prod in ("Mortgage", "Auto Loan", "Personal Loan", "Credit Card"):
        parts.append(_retail_loans(b, prod, st["product_mix"][prod], as_of, new=False))

    # C&I: companies with one or more facilities, grouped into economic groups.
    n_ci = sum(st["product_mix"][p] for p in ("Commercial Term Loan", "Revolving Credit Facility", "Equipment Lease"))
    n_companies = int(n_ci * 0.7)
    companies = []
    for i in range(n_companies):
        notch = float(np.clip(round(r.normal(10.6, 1.6)), 4, 17))
        grp = f"EG_{(i // 4) + 1:05d}" if r.random() < 0.25 else None
        companies.append(b.commercial_cp(str(r.choice(INDUSTRIES)), notch, r.lognormal(np.log(3e6), 0.9), grp))
    for prod in ("Commercial Term Loan", "Revolving Credit Facility", "Equipment Lease"):
        k = st["product_mix"][prod]
        pool = list(r.choice(companies, k))
        parts.append(_commercial_loans(b, prod, k, as_of, new=False, cp_pool=pool))

    # CRE: deterioration cohort + rest of the book.
    coh = st["cre_cohort"]
    cohort_cps, cohort_parts = [], []
    exposures = r.uniform(*coh["exposure_range"], coh["n_borrowers"])
    exposures[0] = coh["largest_exposure"]
    for i, expo in enumerate(exposures):
        notch = float(r.integers(coh["july"]["rating_notch"][0], coh["july"]["rating_notch"][1] + 1))
        cid = b.commercial_cp("Real Estate", notch, expo, name=f"{r.choice(CO_A)} {r.choice(['Office', 'Plaza', 'Commons', 'Tower', 'Center'])} Properties LLC")
        cohort_cps.append(cid)
        n_l = 2 if expo > 9e6 else 1
        bal = np.round(expo * (np.array([0.6, 0.4]) if n_l == 2 else np.array([1.0])), 2)
        part = _commercial_loans(b, "Commercial Real Estate", n_l, as_of, new=False, cp_pool=[cid] * n_l, balances=bal)
        part["rating_notch"] = notch
        part["dscr"] = r.uniform(*coh["july"]["dscr"], n_l).round(3)
        part["ltv"] = r.uniform(*coh["july"]["ltv"], n_l).round(3)
        part["property_value"] = (part["current_balance"] / part["ltv"]).round(2)
        part["property_type"] = r.choice(coh["property_types"], n_l)
        part["days_past_due"] = 0.0
        part["remaining_term_months"] = r.integers(24, 72, n_l)
        part["maturity_date"] = [pd.Timestamp(as_of) + pd.DateOffset(months=int(t)) for t in part["remaining_term_months"]]
        cohort_parts.append(part)
    cohort = pd.concat(cohort_parts, ignore_index=True)
    rest = st["product_mix"]["Commercial Real Estate"] - len(cohort)
    parts.append(cohort)
    parts.append(_commercial_loans(b, "Commercial Real Estate", rest, as_of, new=False))
    july = pd.concat(parts, ignore_index=True)
    return _finalise(july, "2026-07-31"), cohort_cps


def build_august(b: _Builder, july: pd.DataFrame, cohort_cps: list[str]) -> tuple[pd.DataFrame, dict]:
    st, meth = config.story(), config.methodology()
    bg = st["background"]
    r = b.rng
    as_of = date(2026, 8, 31)
    aug = july.drop(columns=["as_of_date", "portfolio", "segment", "internal_rating", "delinquency_bucket",
                             "default_flag", "pd_model_id", "lgd_model_id", "ead_model_id", "scenario_set_id"]).copy()
    n = len(aug)
    protected = aug["counterparty_id"].isin(cohort_cps).to_numpy()
    commercial = aug["product_type"].map(lambda p: meth["products"][p]["segment"] == "COMMERCIAL").to_numpy()
    revolving = aug["product_type"].map(lambda p: meth["products"][p]["profile"] == "revolving").to_numpy()

    # 1. Exits: voluntary payoffs, maturities (commercial mostly renewed), charge-offs.
    payoff = r.random(n) < aug["product_type"].map(bg["monthly_payoff_rate"]).to_numpy()
    maturing = (aug["remaining_term_months"].to_numpy() <= 1) & ~revolving
    renew = maturing & commercial & (r.random(n) < bg["commercial_renewal_rate"])
    chargeoff = (aug["days_past_due"].to_numpy() >= 90) & (r.random(n) < bg["chargeoff_rate"])
    exit_ = (payoff | (maturing & ~renew) | chargeoff) & ~protected
    aug = aug.loc[~exit_].reset_index(drop=True)
    renew = renew[~exit_]
    revolving = revolving[~exit_]
    commercial = commercial[~exit_]
    m = len(aug)

    # 2. Amortisation, revolver usage, term roll-forward.
    term = aug["remaining_term_months"].to_numpy(float)
    profile = aug["product_type"].map(lambda p: meth["products"][p]["profile"]).to_numpy()
    amort = profile == "amortising"
    bal = aug["current_balance"].to_numpy(float).copy()
    bal[amort] = bal[amort] * (1 - 1 / np.maximum(term[amort], 1))
    limit = bal + aug["undrawn_amount"].to_numpy(float)
    util = aug["utilization"].to_numpy(float).copy()
    util[revolving] = np.clip(util[revolving] + r.normal(0, 0.03, revolving.sum()), 0, 1)
    bal[revolving] = limit[revolving] * util[revolving]
    aug["current_balance"] = bal.round(2)
    aug["utilization"] = np.where(revolving, util.round(3), np.nan)
    aug["undrawn_amount"] = np.where(revolving, (limit - bal).round(2), 0.0)
    new_term = np.where(revolving, term, np.maximum(term - 1, 1))
    new_term = np.where(renew, r.choice([12, 24, 36], m), new_term)
    aug["remaining_term_months"] = new_term.astype(int)
    aug["maturity_date"] = [pd.Timestamp(as_of) + pd.DateOffset(months=int(t)) for t in new_term]
    has_value = aug["property_value"].notna().to_numpy()
    aug.loc[has_value, "ltv"] = (aug.loc[has_value, "current_balance"] / aug.loc[has_value, "property_value"]).round(3)

    # 3. Background credit migration.
    retail = ~commercial
    fico = aug["fico"].to_numpy(float).copy()
    fico[retail] = np.clip(fico[retail] + r.normal(0, bg["retail_fico_drift_sd"], retail.sum()), 560, 850).round()
    aug["fico"] = fico
    dpd = aug["days_past_due"].to_numpy(float).copy()
    u = r.random(m)
    nxt = dpd.copy()
    cure, roll = bg["retail_cure_rate"], bg["retail_roll_rate"]
    nxt[(dpd == 0) & (u < roll)] = 15
    for frm, to in ((15, 45), (45, 75), (75, 120)):
        sel = dpd == frm
        nxt[sel & (u < cure)] = 0
        nxt[sel & (u >= cure) & (u < cure + 0.30)] = to
    dpd = np.where(aug["counterparty_id"].isin(cohort_cps), dpd, nxt)
    aug["days_past_due"] = dpd
    notch = aug["rating_notch"].to_numpy(float).copy()
    u2 = r.random(m)
    mig = bg["commercial_migration_rate"]
    eligible = commercial & ~aug["counterparty_id"].isin(cohort_cps).to_numpy()
    notch[eligible & (u2 < mig)] += 1
    notch[eligible & (u2 > 1 - mig)] -= 1
    aug["rating_notch"] = np.where(commercial, np.clip(notch, 1, 20), np.nan)
    dscr = aug["dscr"].to_numpy(float).copy()
    aug["dscr"] = np.where(eligible, (dscr + r.normal(0, 0.02, m)).round(3), dscr)

    # 4. Injected CRE deterioration cohort.
    shock = st["cre_cohort"]["august_shock"]
    cp_shocks = {}
    for i, cid in enumerate(cohort_cps):
        lo, hi = shock["rating_downgrade_notches"]
        cp_shocks[cid] = dict(
            notches=3 if i == 0 else int(r.integers(lo, hi + 1)),
            dscr_drop=0.17 if i == 0 else float(r.uniform(*shock["dscr_drop"])),
            value_decline=0.08 if i == 0 else float(r.uniform(*shock["property_value_decline"])),
            watchlist=bool(i == 0 or r.random() < shock["watchlist_share"]),
        )
    for cid, s in cp_shocks.items():
        sel = aug["counterparty_id"] == cid
        aug.loc[sel, "rating_notch"] = aug.loc[sel, "rating_notch"] + s["notches"]
        aug.loc[sel, "dscr"] = (aug.loc[sel, "dscr"] - s["dscr_drop"]).round(3)
        aug.loc[sel, "property_value"] = (aug.loc[sel, "property_value"] * (1 - s["value_decline"])).round(2)
        aug.loc[sel, "ltv"] = (aug.loc[sel, "current_balance"] / aug.loc[sel, "property_value"]).round(3)
        aug.loc[sel, "watchlist_flag"] = s["watchlist"]

    # 5. New originations (August), including the new CRE exposure.
    new_parts = []
    for prod, k in st["originations"].items():
        if prod == "Commercial Real Estate":
            split = r.dirichlet(np.full(k, 4.0)) * st["new_cre_exposure"]
            part = _commercial_loans(b, prod, k, as_of, new=True, balances=split.round(2))
            part["property_type"] = r.choice(["Multifamily", "Industrial"], k)
        elif meth["products"][prod]["segment"] == "RETAIL":
            part = _retail_loans(b, prod, k, as_of, new=True)
        else:
            part = _commercial_loans(b, prod, k, as_of, new=True)
        part["origination_date"] = pd.Timestamp(as_of)
        new_parts.append(part)
    new = pd.concat(new_parts, ignore_index=True)
    aug = pd.concat([aug, new], ignore_index=True)
    info = {"exited_loans": int(exit_.sum()), "new_loans": len(new), "cohort_shocks": cp_shocks,
            "new_cre_loan_ids": new.loc[new["product_type"] == "Commercial Real Estate", "loan_id"].tolist()}
    return _finalise(aug, "2026-08-31"), info


def macro_dev_array(macro: pd.DataFrame, as_of: str) -> np.ndarray:
    """(scenario, variable, quarter) deviations from the long-run anchor for one vintage."""
    m = config.methodology()["macro"]
    sub = macro[macro["as_of_date"] == as_of].sort_values(["scenario", "forecast_quarter"])
    out = np.zeros((len(SCENARIOS), len(m["variables"]), m["forecast_quarters"]))
    for si, s in enumerate(SCENARIOS):
        rows = sub[sub["scenario"] == s]
        for vi, v in enumerate(m["variables"]):
            out[si, vi, :] = rows[v].to_numpy() - m["long_run_anchor"][v]
    return out


def generate(out_dir=None) -> dict:
    """Generate and write all datasets. Returns a summary dict."""
    st, sett = config.story(), config.settings()
    out = out_dir or config.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    b = _Builder(sett["seed"])
    july, cohort_cps = build_july(b)
    assert len(july) == sett["n_loans"], len(july)
    aug, info = build_august(b, july, cohort_cps)
    macro = macro_table()
    cps = pd.DataFrame(b.counterparties)
    overlays = pd.DataFrame([{"as_of_date": d, **o} for d in config.reporting_dates() for o in st["management_overlays"]])
    model_meta = pd.DataFrame(config.models()["registry"])

    summary, manifests = {}, []
    for as_of, loans in (("2026-07-31", july), ("2026-08-31", aug)):
        res = run_cecl(RunInputs(as_of, loans, macro_dev_array(macro, as_of), st["scenario_weights"][as_of],
                                 st["management_overlays"]))
        lf = res.loan_frame().set_index("loan_id")
        for col in ("pd_12m", "pd_lifetime", "lgd", "ead", "ecl"):
            loans[col] = lf[col].reindex(loans["loan_id"]).to_numpy()
        loans["stage"] = None   # IFRS 9 staging not applicable under CECL
        loans.to_parquet(out / f"loans_{as_of}.parquet", index=False)
        summary[as_of] = {"loans": len(loans), "balance": float(loans["current_balance"].sum()),
                          "modeled_ecl": res.modeled_ecl, "overlay": res.overlay_total, "allowance": res.allowance}
        manifests.append({
            "run_id": f"CECL_{as_of.replace('-', '')}_OFFICIAL", "as_of_date": as_of, "framework": "CECL",
            "status": "APPROVED", "scenario_set_id": f"MACRO_{as_of[:7].replace('-', '')}",
            "scenario_weights": st["scenario_weights"][as_of],
            "models": {p: {"pd": v["pd_model"], "lgd": v["lgd_model"], "ead": v["ead_model"]}
                       for p, v in config.methodology()["products"].items()},
            "overlays": st["management_overlays"],
            "modeled_ecl": round(res.modeled_ecl, 2), "overlay_total": res.overlay_total,
            "allowance": round(res.allowance, 2), "loan_count": len(loans),
        })

    cps.to_parquet(out / "counterparties.parquet", index=False)
    macro.to_csv(out / "macro_scenarios.csv", index=False)
    overlays.to_csv(out / "management_overlays.csv", index=False)
    model_meta.to_csv(out / "model_metadata.csv", index=False)
    with open(out / "run_manifest.json", "w") as f:
        json.dump(manifests, f, indent=2)

    # Small CSV samples so the data can be browsed on GitHub.
    samples = out / "samples"
    samples.mkdir(exist_ok=True)
    for as_of, loans in (("2026-07-31", july), ("2026-08-31", aug)):
        loans.groupby("product_type", group_keys=False).head(60).to_csv(samples / f"loans_{as_of}_sample.csv", index=False)
    cps.head(300).to_csv(samples / "counterparties_sample.csv", index=False)

    # Ground truth: what was injected. Evaluation only.
    j = july.set_index("loan_id")
    a = aug.set_index("loan_id")
    largest = cohort_cps[0]
    gt = {
        "description": "Injected July -> August 2026 story. Evaluation only: tools and agents must never read this file.",
        "previous_date": "2026-07-31", "current_date": "2026-08-31",
        "primary_driver": "macroeconomic",
        "primary_segment": "CRE",
        "primary_macro_variable": "unemployment_rate",
        "primary_macro_variable_in_cre": "cre_price_growth",
        "model_changed": False,
        "scenario_weights_changed": True,
        "scenario_weights": st["scenario_weights"],
        "overlays_changed": False,
        "cre_cohort_counterparties": cohort_cps,
        "cre_cohort_size": len(cohort_cps),
        "largest_borrower": largest,
        "largest_borrower_shock": info["cohort_shocks"][largest],
        "new_cre_exposure": st["new_cre_exposure"],
        "new_cre_loan_ids": info["new_cre_loan_ids"],
        "exited_loans": info["exited_loans"],
        "new_loans": info["new_loans"],
        "macro_changes": {v: {s: [st["macro_peak_deviation"]["2026-07-31"][s][v], st["macro_peak_deviation"]["2026-08-31"][s][v]]
                              for s in SCENARIOS} for v in config.methodology()["macro"]["variables"]},
        "allowance": {d: round(s["allowance"], 2) for d, s in summary.items()},
        "cohort_pd_12m": {
            "july": float(np.average(j.loc[j["counterparty_id"].isin(cohort_cps), "pd_12m"], weights=j.loc[j["counterparty_id"].isin(cohort_cps), "ead"])),
            "august": float(np.average(a.loc[a["counterparty_id"].isin(cohort_cps), "pd_12m"], weights=a.loc[a["counterparty_id"].isin(cohort_cps), "ead"])),
        },
    }
    with open(out / config.settings()["ground_truth_file"], "w") as f:
        json.dump(gt, f, indent=2)

    # Checksums so anyone can verify the published data matches the generator.
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
    with open(out / "SHA256SUMS.json", "w") as f:
        json.dump(sums, f, indent=2)
    return {"summary": summary, "ground_truth": gt}
