"""Seeded synthetic data generator for the July -> August 2026 demo story.

Ported in spirit from the legacy Dataiku `generate_demo_data.py` (product
taxonomy, borrower attributes, macro scenario structure) but rebuilt around a
*known, injected* story so agents can be evaluated against ground truth:

* macro forecasts worsen (unemployment most), severe weight rises
* 20 CRE office/retail borrowers deteriorate (rating, DSCR, LTV, watchlist)
* ~$40M of new CRE plus normal retail/commercial originations
* normal paydowns, maturities, payoffs and charge-offs
* NO model version change between the two months

All data is synthetic. Names are generated from neutral word lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.cecl.engine import base_pd
from src.utils import config

SP_RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC"]
MOODYS = ["Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3", "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1"]
STATES = ["NC", "SC", "VA", "GA", "FL", "TX", "NY", "PA", "OH", "IL", "CA", "AZ"]
CI_INDUSTRIES = ["Manufacturing", "Wholesale Trade", "Retail Trade", "Professional Services", "Healthcare",
                 "Construction", "Technology", "Transportation & Logistics", "Hospitality", "Agriculture"]
CRE_PROPERTY_TYPES = ["Office", "Retail", "Multifamily", "Industrial", "Hotel"]
CRE_PROPERTY_PROBS = [0.25, 0.20, 0.27, 0.20, 0.08]
NAME_A = ["Northgate", "Riverbend", "Oakridge", "Summit", "Harbor", "Pinecrest", "Bluewater", "Ironwood", "Granite",
          "Maple", "Crescent", "Westfield", "Silverline", "Brookside", "Cedar", "Lakeshore", "Highpoint", "Redstone"]
NAME_B = ["Holdings", "Partners", "Group", "Industries", "Properties", "Ventures", "Capital", "Logistics", "Systems",
          "Realty", "Development", "Manufacturing", "Services", "Enterprises"]
NAME_C = ["LLC", "Inc.", "LP", "Co."]

PRODUCT_SPECS: dict[str, dict[str, Any]] = {
    # term_months: choices; age: max seasoning in months; balance: lognormal (median, sigma); rate range
    "Mortgage": {"terms": [360, 360, 360, 180], "max_age": 120, "median": 260_000, "sigma": 0.45, "rate": (0.030, 0.075)},
    "Auto Loan": {"terms": [48, 60, 72], "max_age": 48, "median": 26_000, "sigma": 0.40, "rate": (0.050, 0.120)},
    "Personal Loan": {"terms": [24, 36, 48, 60], "max_age": 40, "median": 11_000, "sigma": 0.55, "rate": (0.090, 0.180)},
    "Credit Card": {"terms": [None], "max_age": 120, "median": 8_000, "sigma": 0.60, "rate": (0.180, 0.280)},
    "Commercial Term Loan": {"terms": [36, 60, 84], "max_age": 48, "median": 550_000, "sigma": 0.90, "rate": (0.060, 0.100)},
    "Revolving Credit Facility": {"terms": [24, 36, 60], "max_age": 24, "median": 1_400_000, "sigma": 0.80, "rate": (0.065, 0.105)},
    "Equipment Lease": {"terms": [36, 48, 60, 84], "max_age": 48, "median": 160_000, "sigma": 0.70, "rate": (0.055, 0.095)},
    "Commercial Real Estate": {"terms": [60, 84, 120], "max_age": 84, "median": 1_500_000, "sigma": 0.85, "rate": (0.045, 0.085)},
}


def month_index(ts: pd.Timestamp) -> int:
    return ts.year * 12 + ts.month - 1


def mid_month_dates(idx: np.ndarray) -> pd.Series:
    idx = np.asarray(idx, dtype=int)
    return pd.to_datetime(pd.DataFrame({"year": idx // 12, "month": idx % 12 + 1, "day": 15}))


def annuity_balance(principal, annual_rate, n_months, k_paid):
    r = np.asarray(annual_rate, float) / 12.0
    n = np.asarray(n_months, float)
    k = np.asarray(k_paid, float)
    growth_n, growth_k = np.power(1 + r, n), np.power(1 + r, k)
    return np.asarray(principal, float) * (growth_n - growth_k) / (growth_n - 1)


def delinquency_bucket(dpd: np.ndarray) -> np.ndarray:
    return np.select([dpd >= 90, dpd >= 60, dpd >= 30, dpd >= 1], ["90+", "60-89", "30-59", "1-29"], "CURRENT")


@dataclass
class GeneratedData:
    loans: pd.DataFrame
    counterparties: pd.DataFrame
    macro: pd.DataFrame
    scenario_weights: pd.DataFrame
    overlays: pd.DataFrame
    model_metadata: pd.DataFrame
    model_assignments: pd.DataFrame
    model_parameters: pd.DataFrame
    ground_truth: dict[str, Any] = field(default_factory=dict)


class DemoDataGenerator:
    def __init__(self, seed: int | None = None):
        self.settings = config.settings()
        self.method = config.methodology()
        self.models = config.models()
        self.story = config.demo_story()
        self.seed = seed if seed is not None else self.settings["seed"]
        self.rng = np.random.default_rng(self.seed)
        d0, d1 = self.settings["reporting_dates"]
        self.d0, self.d1 = pd.Timestamp(d0), pd.Timestamp(d1)
        self._loan_seq = 0
        self._cp_seq = 0

    # ------------------------------------------------------------------ ids
    def _loan_ids(self, n: int) -> list[str]:
        ids = [f"LN_{i:07d}" for i in range(self._loan_seq + 1, self._loan_seq + n + 1)]
        self._loan_seq += n
        return ids

    def _cp_ids(self, n: int) -> list[str]:
        ids = [f"CP_{i:06d}" for i in range(self._cp_seq + 1, self._cp_seq + n + 1)]
        self._cp_seq += n
        return ids

    def _company_names(self, n: int) -> list[str]:
        r = self.rng
        return [f"{NAME_A[a]} {NAME_B[b]} {NAME_C[c]}" for a, b, c in
                zip(r.integers(0, len(NAME_A), n), r.integers(0, len(NAME_B), n), r.integers(0, len(NAME_C), n))]

    # ------------------------------------------------------- counterparties
    def _make_retail_cps(self, n: int, as_of: pd.Timestamp) -> pd.DataFrame:
        r = self.rng
        ids = self._cp_ids(n)
        return pd.DataFrame({
            "as_of_date": as_of, "counterparty_id": ids, "counterparty_type": "INDIVIDUAL",
            "customer_name": [f"Retail Customer {i[3:]}" for i in ids],
            "industry": pd.NA, "state": r.choice(STATES, n),
            "annual_revenue": np.nan, "total_assets": np.nan, "employees": np.nan,
            "fico": np.clip(np.round(r.normal(722, 58, n)), 540, 850),
            "income": np.round(np.exp(r.normal(np.log(72_000), 0.45, n)), -2),
            "dti": np.clip(r.beta(4, 8, n), 0.05, 0.65),
            "rating_notch": np.nan, "internal_rating": pd.NA, "sp_rating": pd.NA, "moodys_rating": pd.NA,
            "dscr": np.nan, "leverage": np.nan, "watchlist_flag": 0, "economic_group_id": pd.NA,
        })

    def _make_company_cps(self, n: int, as_of: pd.Timestamp, kind: str) -> pd.DataFrame:
        r = self.rng
        ids = self._cp_ids(n)
        notch = np.clip(np.round(r.normal(10.2 if kind == "CI" else 10.0, 1.7, n)), 4, 16).astype(int)
        revenue = np.round(np.exp(r.normal(np.log(18e6 if kind == "CI" else 6e6), 1.2, n)), -3)
        rated = revenue > 250e6
        groups = np.where(r.random(n) < 0.08, [f"EG_{g:04d}" for g in r.integers(1, 120, n)], None)
        return pd.DataFrame({
            "as_of_date": as_of, "counterparty_id": ids, "counterparty_type": "COMPANY",
            "customer_name": self._company_names(n),
            "industry": r.choice(CI_INDUSTRIES, n) if kind == "CI" else "Real Estate",
            "state": r.choice(STATES, n),
            "annual_revenue": revenue, "total_assets": np.round(revenue * r.uniform(0.8, 2.5, n), -3),
            "employees": np.maximum(5, np.round(revenue / r.uniform(150e3, 400e3, n))),
            "fico": np.nan, "income": np.nan, "dti": np.nan,
            "rating_notch": notch, "internal_rating": [SP_RATINGS[k - 1] for k in notch],
            "sp_rating": np.where(rated, [SP_RATINGS[k - 1] for k in notch], None),
            "moodys_rating": np.where(rated, [MOODYS[k - 1] for k in notch], None),
            "dscr": np.clip(r.lognormal(np.log(1.6), 0.18, n), 0.9, 3.5) if kind == "CI" else np.nan,
            "leverage": np.clip(r.gamma(4.0, 0.75, n), 0.5, 8.0) if kind == "CI" else np.nan,
            "watchlist_flag": (r.random(n) < 0.03).astype(int),
            "economic_group_id": groups,
        })

    # ---------------------------------------------------------------- loans
    def _make_loans(self, product: str, cp_ids: np.ndarray, as_of: pd.Timestamp, new: bool = False) -> pd.DataFrame:
        r = self.rng
        n = len(cp_ids)
        spec = PRODUCT_SPECS[product]
        pcfg = self.method["products"][product]
        asof_idx = month_index(as_of)

        if product == "Credit Card":
            term = np.full(n, np.nan)
            age = np.zeros(n, int) if new else r.integers(1, spec["max_age"], n)
        else:
            term = r.choice(spec["terms"], n).astype(float)
            age = np.zeros(n, int) if new else np.minimum(r.integers(1, spec["max_age"] + 1, n), term.astype(int) - 1)
        orig_idx = asof_idx - age
        remaining = term - age

        original = np.round(np.exp(r.normal(np.log(spec["median"]), spec["sigma"], n)), 2)
        rate = np.round(r.uniform(*spec["rate"], n), 5)
        limit = np.full(n, np.nan)
        if pcfg["amortization"] == "AMORTIZING":
            balance = annuity_balance(original, rate, term, age)
        elif pcfg["amortization"] == "BULLET":
            balance = original * (1 - r.uniform(0, 0.08, n) * (age > 12))
        else:  # REVOLVING: original = commitment / limit
            limit = original
            util = np.clip(r.beta(2.0, 3.5, n) if product == "Credit Card" else r.beta(3, 3, n), 0.0, 1.0)
            balance = limit * util
        balance = np.round(balance, 2)
        undrawn = np.where(np.isnan(limit), 0.0, np.round(limit - balance, 2))
        utilization = np.where(np.isnan(limit), np.nan, balance / np.where(np.isnan(limit), 1, limit))

        collateral = np.full(n, np.nan)
        ltv = np.full(n, np.nan)
        dscr = np.full(n, np.nan)
        prop_type = np.full(n, None, dtype=object)
        if product == "Mortgage":
            value0 = original / r.uniform(0.60, 0.95, n)
            collateral = np.round(value0 * np.power(1.035, age / 12), 2)
        elif product == "Auto Loan":
            value0 = original / r.uniform(0.80, 1.05, n)
            collateral = np.round(value0 * np.power(0.85, age / 12), 2)
        elif product == "Equipment Lease":
            collateral = np.round(original * np.power(0.80, age / 12), 2)
        elif product == "Commercial Real Estate":
            prop_type = r.choice(CRE_PROPERTY_TYPES, n, p=CRE_PROPERTY_PROBS)
            ltv0 = np.clip(r.normal(0.60, 0.08, n), 0.35, 0.85)
            collateral = np.round(balance / ltv0, 2)
            dscr = np.round(np.clip(r.normal(1.50, 0.20, n), 1.0, 2.5), 3)
        if product in ("Mortgage", "Auto Loan", "Commercial Real Estate"):
            ltv = np.round(balance / collateral, 4)

        if new:
            dpd = np.zeros(n)
        else:
            mix = self.story["background"]["initial_dpd_mix"][pcfg["segment"]]
            dpd = r.choice([0, 15, 45, 75, 120], n, p=mix).astype(float)

        return pd.DataFrame({
            "as_of_date": as_of,
            "loan_id": self._loan_ids(n),
            "counterparty_id": cp_ids,
            "portfolio": pcfg["portfolio"],
            "product_type": product,
            "segment": pcfg["segment"],
            "property_type": prop_type,
            "origination_date": mid_month_dates(orig_idx).to_numpy(),
            "maturity_date": mid_month_dates(orig_idx + np.nan_to_num(term, nan=0).astype(int)).to_numpy() if product != "Credit Card" else pd.NaT,
            "original_term_months": term,
            "remaining_term_months": remaining,
            "original_balance": original,
            "current_balance": balance,
            "credit_limit": limit,
            "undrawn_amount": undrawn,
            "utilization": utilization,
            "interest_rate": rate,
            "collateral_value": collateral,
            "ltv": ltv,
            "dscr": dscr,
            "watchlist_flag": 0,
            "days_past_due": dpd,
            "pd_model_id": pcfg["pd_model"],
            "lgd_model_id": pcfg["lgd_model"],
            "ead_model_id": pcfg["ead_model"],
            "_age": age,
        })

    # --------------------------------------------------------------- utils
    def _attach_borrower_fields(self, loans: pd.DataFrame, cps: pd.DataFrame) -> pd.DataFrame:
        """Copy borrower-level credit attributes onto loans (refreshable each month)."""
        cp = cps.set_index("counterparty_id")
        l = loans.copy()
        for col in ["fico", "dti", "rating_notch", "internal_rating", "leverage", "industry", "state"]:
            l[col] = l["counterparty_id"].map(cp[col])
        ci = l["portfolio"].eq("C&I")
        l.loc[ci, "dscr"] = l.loc[ci, "counterparty_id"].map(cp["dscr"])
        com = l["segment"].eq("COMMERCIAL")
        cp_watch = l["counterparty_id"].map(cp["watchlist_flag"]).fillna(0).astype(int)
        l["watchlist_flag"] = np.where(com, np.maximum(l["watchlist_flag"].fillna(0).astype(int), cp_watch), 0)
        cre = l["portfolio"].eq("CRE")
        l.loc[cre, "industry"] = "CRE - " + l.loc[cre, "property_type"].astype(str)
        return l

    def _finalize(self, loans: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        l = loans.copy()
        l["as_of_date"] = as_of
        l["delinquency_bucket"] = delinquency_bucket(l["days_past_due"].to_numpy())
        l["default_flag"] = (l["days_past_due"] >= self.method["calculation"]["default_dpd"]).astype(int)
        l["scenario_set_id"] = f"SCN_{as_of:%Y%m}"
        # maturity is always consistent with remaining term at this as-of date
        termed = l["remaining_term_months"].notna()
        idx = month_index(as_of) + l.loc[termed, "remaining_term_months"].astype(int).to_numpy()
        l.loc[termed, "maturity_date"] = mid_month_dates(idx).to_numpy()
        return l

    def _set_origination_risk(self, loans: pd.DataFrame) -> pd.DataFrame:
        """Origination FICO / rating and the base PD they imply (for IFRS 9 SICR)."""
        r = self.rng
        l = loans.copy()
        n = len(l)
        new = l["_age"].to_numpy() == 0
        l["origination_fico"] = np.where(new, l["fico"], np.clip(np.round(l["fico"] + r.normal(4, 14, n)), 540, 850))
        drift = r.choice([-1, 0, 0, 0, 0, 1, 1, 2], n)
        l["origination_rating_notch"] = np.where(new, l["rating_notch"], np.clip(l["rating_notch"] - drift, 1, 17))
        o = l.copy()
        o["fico"] = l["origination_fico"]
        o["rating_notch"] = l["origination_rating_notch"]
        o["days_past_due"] = 0
        o["watchlist_flag"] = 0
        l["origination_pd"] = base_pd(o, self.models, self.method)
        return l

    # ------------------------------------------------------------ July book
    def build_july(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        r = self.rng
        mix = self.story["product_mix"]
        retail_products = [p for p in mix if self.method["products"][p]["segment"] == "RETAIL"]
        n_retail = sum(mix[p] for p in retail_products)
        ci_products = [p for p in mix if self.method["products"][p]["portfolio"] == "C&I"]
        n_ci = sum(mix[p] for p in ci_products)
        n_cre = mix["Commercial Real Estate"]

        retail_cps = self._make_retail_cps(int(n_retail * 0.86), self.d0)
        ci_cps = self._make_company_cps(int(n_ci * 0.55), self.d0, "CI")
        cre_cps = self._make_company_cps(int(n_cre * 0.70), self.d0, "CRE")

        frames = []
        for p in mix:
            seg = self.method["products"][p]
            pool = retail_cps if seg["segment"] == "RETAIL" else (cre_cps if seg["portfolio"] == "CRE" else ci_cps)
            frames.append(self._make_loans(p, r.choice(pool["counterparty_id"].to_numpy(), mix[p]), self.d0))
        loans = pd.concat(frames, ignore_index=True)
        cps = pd.concat([retail_cps, ci_cps, cre_cps], ignore_index=True)
        cps = cps[cps["counterparty_id"].isin(loans["counterparty_id"])].reset_index(drop=True)

        loans, cohort = self._seed_cre_cohort(loans, cps)
        cps = self._apply_cohort_july(cps, cohort)
        loans = self._attach_borrower_fields(loans, cps)
        loans = self._finalize(loans, self.d0)
        loans = self._set_origination_risk(loans)
        self.cohort = cohort
        return loans, cps

    def _seed_cre_cohort(self, loans: pd.DataFrame, cps: pd.DataFrame):
        """Pick CRE office/retail borrowers and make them meaningful, healthy-looking in July."""
        r = self.rng
        cfg = self.story["cre_cohort"]
        cre = loans[loans["portfolio"].eq("CRE")]
        eligible = cre[cre["property_type"].isin(cfg["property_types"]) & cre["days_past_due"].eq(0)]
        cands = eligible["counterparty_id"].unique()
        cohort_ids = list(r.choice(cands, cfg["n_borrowers"], replace=False))
        largest = cohort_ids[0]
        lo, hi = cfg["borrower_exposure_range"]
        targets = {cp: (cfg["largest_borrower_exposure"] if cp == largest else r.uniform(lo, hi)) for cp in cohort_ids}

        l = loans.copy()
        jl = cfg["july"]
        for cp, target in targets.items():
            m = l["counterparty_id"].eq(cp) & l["portfolio"].eq("CRE")
            idx = l.index[m]
            k = len(idx)
            share = r.dirichlet(np.ones(k) * 3) if k > 1 else np.array([1.0])
            bal = np.round(target * share, 2)
            ltv = r.uniform(*jl["ltv"], k)
            l.loc[idx, "current_balance"] = bal
            l.loc[idx, "original_balance"] = bal
            l.loc[idx, "collateral_value"] = np.round(bal / ltv, 2)
            l.loc[idx, "ltv"] = np.round(ltv, 4)
            l.loc[idx, "dscr"] = np.round(r.uniform(*jl["dscr"], k), 3)
            l.loc[idx, "property_type"] = "Office" if cp == largest else r.choice(cfg["property_types"])
            l.loc[idx, "days_past_due"] = 0.0
            # keep cohort loans on the books through August
            l.loc[idx, "remaining_term_months"] = np.maximum(l.loc[idx, "remaining_term_months"], 30)
            l.loc[idx, "original_term_months"] = np.maximum(l.loc[idx, "original_term_months"],
                                                            l.loc[idx, "remaining_term_months"] + l.loc[idx, "_age"])
        notch = {cp: int(r.integers(jl["rating_notch"][0], jl["rating_notch"][1] + 1)) for cp in cohort_ids}
        return l, {"ids": cohort_ids, "largest": largest, "july_notch": notch}

    def _apply_cohort_july(self, cps: pd.DataFrame, cohort: dict) -> pd.DataFrame:
        c = cps.copy()
        for cp, notch in cohort["july_notch"].items():
            m = c["counterparty_id"].eq(cp)
            c.loc[m, ["rating_notch", "watchlist_flag"]] = [notch, 0]
            c.loc[m, "internal_rating"] = SP_RATINGS[notch - 1]
        return c

    # ---------------------------------------------------------- August book
    def evolve_to_august(self, jl: pd.DataFrame, jc: pd.DataFrame):
        r = self.rng
        bg = self.story["background"]
        cohort = set(self.cohort["ids"])
        events: dict[str, Any] = {}

        l = jl.copy()
        cps = jc.copy()
        cps["as_of_date"] = self.d1

        # 1. exits: maturities, voluntary payoffs, charge-offs of defaulted loans
        rem_next = l["remaining_term_months"] - 1
        matured = rem_next.le(0).fillna(False).to_numpy()
        # most maturing commercial facilities are renewed rather than repaid
        renew = matured & l["segment"].eq("COMMERCIAL").to_numpy() & (r.random(len(l)) < bg["commercial_renewal_rate"])
        ext = r.choice([12, 24, 36], len(l))
        l.loc[renew, "remaining_term_months"] = l.loc[renew, "remaining_term_months"] + ext[renew]
        l.loc[renew, "original_term_months"] = l.loc[renew, "original_term_months"] + ext[renew]
        events["renewals"] = int(renew.sum())
        matured = matured & ~renew
        payoff_p = l["product_type"].map(bg["monthly_payoff_rate"]).to_numpy(float)
        payoff = (r.random(len(l)) < payoff_p) & ~l["counterparty_id"].isin(cohort).to_numpy() & (l["default_flag"].to_numpy() == 0)
        chargeoff = (l["default_flag"].to_numpy() == 1) & (r.random(len(l)) < bg["chargeoff_rate"])
        exit_mask = matured | payoff | chargeoff
        events["exits"] = {"matured": int(matured.sum()), "paid_off": int((payoff & ~matured).sum()),
                           "charged_off": int((chargeoff & ~matured & ~payoff).sum())}
        events["exit_balance"] = float(l.loc[exit_mask, "current_balance"].sum())
        l = l.loc[~exit_mask].copy()
        l["remaining_term_months"] = l["remaining_term_months"] - 1
        l["_age"] = l["_age"] + 1

        # 2. balances: amortisation, utilisation moves, collateral values
        n = len(l)
        amort = l["product_type"].map(lambda p: self.method["products"][p]["amortization"]).to_numpy()
        a = amort == "AMORTIZING"
        bal = l["current_balance"].to_numpy(float).copy()
        rate = l["interest_rate"].to_numpy(float) / 12
        nrem = l["remaining_term_months"].to_numpy(float) + 1
        pmt = np.where(a, bal * rate / (1 - np.power(1 + rate, -np.maximum(nrem, 1))), 0.0)
        bal = np.where(a, np.maximum(bal * (1 + rate) - pmt, 0.0), bal)
        rev = amort == "REVOLVING"
        card = l["product_type"].eq("Credit Card").to_numpy()
        util = l["utilization"].to_numpy(float)
        util = np.where(rev, np.clip(util + r.normal(np.where(card, 0.0, 0.004), 0.035, n), 0, 1), util)
        limit = l["credit_limit"].to_numpy(float)
        bal = np.where(rev, limit * util, bal)
        l["current_balance"] = np.round(bal, 2)
        l["utilization"] = util
        l["undrawn_amount"] = np.where(rev, np.round(limit - bal, 2), 0.0)
        cv = l["collateral_value"].to_numpy(float)
        cv = np.where(l["product_type"].eq("Mortgage"), cv * 1.002, cv)
        cv = np.where(l["product_type"].eq("Auto Loan"), cv * 0.9865, cv)
        cv = np.where(l["product_type"].eq("Equipment Lease"), cv * 0.982, cv)
        l["collateral_value"] = np.round(cv, 2)

        # 3. background credit migration
        ret_cp = cps["counterparty_type"].eq("INDIVIDUAL")
        cps.loc[ret_cp, "fico"] = np.clip(np.round(cps.loc[ret_cp, "fico"] + r.normal(bg["retail_fico_drift"]["mean"], bg["retail_fico_drift"]["sd"], ret_cp.sum())), 540, 850)
        com_cp = cps["counterparty_type"].eq("COMPANY") & ~cps["counterparty_id"].isin(cohort)
        u = r.random(len(cps))
        move = np.where(u < bg["commercial_downgrade_rate"], 1, np.where(u < bg["commercial_downgrade_rate"] + bg["commercial_upgrade_rate"], -1, 0))
        cps.loc[com_cp, "rating_notch"] = np.clip(cps.loc[com_cp, "rating_notch"] + move[com_cp.to_numpy()], 1, 17)
        cps.loc[com_cp & cps["dscr"].notna(), "dscr"] = np.clip(cps.loc[com_cp & cps["dscr"].notna(), "dscr"] + r.normal(0, 0.03, (com_cp & cps["dscr"].notna()).sum()), 0.8, 3.5)

        dpd = l["days_past_due"].to_numpy(float).copy()
        retail = l["segment"].eq("RETAIL").to_numpy()
        u = r.random(n)
        new_dpd = dpd.copy()
        cur = dpd == 0
        new_dpd[cur & retail & (u < bg["retail_roll_rate"])] = 15
        new_dpd[cur & retail & (u >= bg["retail_roll_rate"]) & (u < 2 * bg["retail_roll_rate"])] = 45
        delinquent = (dpd > 0) & (dpd < 90)
        cure = delinquent & (u < bg["retail_cure_rate"])
        roll = delinquent & ~cure
        new_dpd[cure] = 0
        new_dpd[roll] = dpd[roll] + 30
        l["days_past_due"] = new_dpd

        # 4. CRE cohort deterioration (the injected story)
        l, cps, events["cohort"] = self._shock_cohort(l, cps)

        # 5. LTVs refresh from balances / collateral
        has_ltv = l["product_type"].isin(["Mortgage", "Auto Loan", "Commercial Real Estate"])
        l.loc[has_ltv, "ltv"] = np.round(l.loc[has_ltv, "current_balance"] / l.loc[has_ltv, "collateral_value"], 4)

        # 6. new originations
        new_loans, new_cps, events["originations"] = self._originate(cps)
        cps = pd.concat([cps, new_cps], ignore_index=True)
        cps["internal_rating"] = [SP_RATINGS[int(k) - 1] if pd.notna(k) else pd.NA for k in cps["rating_notch"]]
        rated = cps["sp_rating"].notna()
        cps.loc[rated, "sp_rating"] = [SP_RATINGS[int(k) - 1] for k in cps.loc[rated, "rating_notch"]]
        cps.loc[rated, "moodys_rating"] = [MOODYS[int(k) - 1] for k in cps.loc[rated, "rating_notch"]]

        l = self._attach_borrower_fields(l, cps)
        new_loans = self._attach_borrower_fields(new_loans, cps)
        new_loans = self._set_origination_risk(new_loans)
        l = pd.concat([l, new_loans], ignore_index=True)
        l = self._finalize(l, self.d1)
        cps = cps[cps["counterparty_id"].isin(l["counterparty_id"])].reset_index(drop=True)
        return l, cps, events

    def _shock_cohort(self, l: pd.DataFrame, cps: pd.DataFrame):
        r = self.rng
        cfg = self.story["cre_cohort"]["august_shock"]
        ids = self.cohort["ids"]
        largest = self.cohort["largest"]
        watch = set(r.choice(ids, int(round(cfg["watchlist_share"] * len(ids))), replace=False)) | {largest}
        dpd30 = set(r.choice([i for i in ids if i != largest], cfg["dpd_30_count"], replace=False))
        detail = []
        for cp in ids:
            lo, hi = cfg["rating_downgrade_notches"]
            dn = hi + 1 if cp == largest else int(r.integers(lo, hi + 1))
            m = cps["counterparty_id"].eq(cp)
            old = int(cps.loc[m, "rating_notch"].iloc[0])
            cps.loc[m, "rating_notch"] = min(17, old + dn)
            cps.loc[m, "watchlist_flag"] = int(cp in watch)
            lm = l["counterparty_id"].eq(cp) & l["portfolio"].eq("CRE")
            k = int(lm.sum())
            drop = (cfg["dscr_drop"][1] + 0.05) if cp == largest else r.uniform(*cfg["dscr_drop"])
            decline = (cfg["property_value_decline"][1] + 0.02) if cp == largest else r.uniform(*cfg["property_value_decline"])
            l.loc[lm, "dscr"] = np.round(l.loc[lm, "dscr"] - drop, 3)
            l.loc[lm, "collateral_value"] = np.round(l.loc[lm, "collateral_value"] * (1 - decline), 2)
            l.loc[lm, "watchlist_flag"] = int(cp in watch)
            l.loc[lm, "days_past_due"] = 45.0 if cp in dpd30 else 0.0
            detail.append({"counterparty_id": cp, "rating_notch_from": old, "rating_notch_to": min(17, old + dn),
                           "dscr_drop": round(float(drop), 3), "property_value_decline": round(float(decline), 3),
                           "watchlist": cp in watch, "dpd_30_59": cp in dpd30, "loans": k})
        return l, cps, detail

    def _originate(self, cps: pd.DataFrame):
        r = self.rng
        orig = self.story["originations"]
        frames, new_cps = [], []
        existing_retail = cps.loc[cps["counterparty_type"].eq("INDIVIDUAL"), "counterparty_id"].to_numpy()
        for p, n in orig.items():
            seg = self.method["products"][p]
            if seg["segment"] == "RETAIL":
                n_new = int(n * 0.7)
                nc = self._make_retail_cps(n_new, self.d1)
                new_cps.append(nc)
                ids = np.concatenate([nc["counterparty_id"].to_numpy(), r.choice(existing_retail, n - n_new)])
            else:
                kind = "CRE" if seg["portfolio"] == "CRE" else "CI"
                nc = self._make_company_cps(n, self.d1, kind)
                if kind == "CRE":
                    nc["rating_notch"] = r.integers(8, 11, n)
                    nc["watchlist_flag"] = 0
                new_cps.append(nc)
                ids = nc["counterparty_id"].to_numpy()
            nl = self._make_loans(p, ids, self.d1, new=True)
            if p == "Commercial Real Estate":
                target = self.story["new_cre_exposure"]
                share = r.dirichlet(np.ones(n) * 4)
                bal = np.round(target * share, 2)
                bal[-1] = round(target - bal[:-1].sum(), 2)
                ltv = r.uniform(0.55, 0.65, n)
                nl["current_balance"] = bal
                nl["original_balance"] = bal
                nl["collateral_value"] = np.round(bal / ltv, 2)
                nl["ltv"] = np.round(ltv, 4)
                nl["dscr"] = np.round(r.uniform(1.5, 1.8, n), 3)
                nl["property_type"] = r.choice(["Multifamily", "Industrial"], n)
            frames.append(nl)
        new = pd.concat(frames, ignore_index=True)
        summary = {
            "count": int(len(new)),
            "balance": float(new["current_balance"].sum()),
            "by_product": new.groupby("product_type")["current_balance"].agg(["count", "sum"]).round(2).to_dict("index"),
        }
        return new, pd.concat(new_cps, ignore_index=True), summary

    # ----------------------------------------------- scenarios & reference
    def build_macro(self) -> pd.DataFrame:
        mcfg = self.method["macro"]
        shape = np.array(self.story["macro_path_shape"], float)
        weights = self.story["scenario_weights"]
        rows = []
        for d in (self.d0, self.d1):
            key = f"{d:%Y-%m-%d}"
            q0 = pd.Period(d, freq="Q")
            for s in mcfg["scenarios"]:
                peak = self.story["macro_peak_deviation"][key][s]
                for q in range(len(shape)):
                    row = {"as_of_date": d, "scenario_set_id": f"SCN_{d:%Y%m}", "scenario": s,
                           "scenario_weight": weights[key][s], "forecast_quarter": q + 1,
                           "quarter_label": str(q0 + q + 1)}
                    for v in mcfg["variables"]:
                        row[v] = round(mcfg["long_run_anchor"][v] + peak[v] * shape[q], 4)
                    rows.append(row)
        return pd.DataFrame(rows)

    def build_weights(self) -> pd.DataFrame:
        rows = []
        for d in (self.d0, self.d1):
            for s, w in self.story["scenario_weights"][f"{d:%Y-%m-%d}"].items():
                rows.append({"as_of_date": d, "scenario_set_id": f"SCN_{d:%Y%m}", "scenario": s, "weight": w,
                             "approved_by": "ALLL Committee (synthetic)", "approval_date": d - pd.Timedelta(days=3)})
        return pd.DataFrame(rows)

    def build_overlays(self) -> pd.DataFrame:
        rows = []
        for d in (self.d0, self.d1):
            for o in self.story["management_overlays"]:
                rows.append({"as_of_date": d, **o})
        return pd.DataFrame(rows)

    def build_model_reference(self):
        prods = self.method["products"]
        meta = [
            # model_id, type, portfolio, version, deployed, calibrated, status, description
            ("MORT_PD_v3", "PD", "Mortgage", 3, "2025-03-01", "2025-01-31", "ACTIVE", "Mortgage PD scorecard (FICO, LTV, DTI) with macro overlay"),
            ("MORT_LGD_v2", "LGD", "Mortgage", 2, "2024-06-01", "2024-04-30", "ACTIVE", "Mortgage LGD, LTV-driven with HPI sensitivity"),
            ("AUTO_PD_v2", "PD", "Auto", 2, "2024-09-01", "2024-07-31", "ACTIVE", "Auto PD scorecard"),
            ("AUTO_LGD_v2", "LGD", "Auto", 2, "2024-09-01", "2024-07-31", "ACTIVE", "Auto LGD"),
            ("CONS_PD_v2", "PD", "Consumer", 2, "2025-01-15", "2024-12-31", "ACTIVE", "Unsecured personal loan PD"),
            ("CONS_LGD_v1", "LGD", "Consumer", 1, "2023-05-01", "2023-03-31", "ACTIVE", "Unsecured personal loan LGD"),
            ("CC_PD_v5", "PD", "Credit Card", 5, "2025-06-01", "2025-04-30", "ACTIVE", "Credit card PD (FICO, utilization, DTI)"),
            ("CC_PD_v4", "PD", "Credit Card", 4, "2023-06-01", "2023-04-30", "RETIRED", "Superseded by CC_PD_v5 in 2025-06"),
            ("CC_LGD_v3", "LGD", "Credit Card", 3, "2024-02-01", "2023-12-31", "ACTIVE", "Credit card LGD"),
            ("CI_PD_v3", "PD", "C&I", 3, "2025-02-01", "2024-12-31", "ACTIVE", "C&I PD (rating, DSCR, leverage, watchlist)"),
            ("CI_LGD_v2", "LGD", "C&I", 2, "2024-02-01", "2023-12-31", "ACTIVE", "C&I LGD"),
            ("LEASE_LGD_v1", "LGD", "C&I", 1, "2023-08-01", "2023-06-30", "ACTIVE", "Equipment lease LGD"),
            ("CRE_PD_v3", "PD", "CRE", 3, "2023-10-01", "2023-08-31", "RETIRED", "Superseded by CRE_PD_v4 in 2025-11"),
            ("CRE_PD_v4", "PD", "CRE", 4, "2025-11-01", "2025-09-30", "ACTIVE", "CRE PD (rating, DSCR, LTV, watchlist, CRE price index)"),
            ("CRE_PD_v5", "PD", "CRE", 5, None, "2026-06-30", "IN_VALIDATION", "Candidate CRE PD model; not approved for production use"),
            ("CRE_LGD_v3", "LGD", "CRE", 3, "2025-11-01", "2025-09-30", "ACTIVE", "CRE LGD, LTV-driven with CRE price sensitivity"),
            ("EAD_BAL_v1", "EAD", "ALL", 1, "2023-01-01", None, "ACTIVE", "Contractual balance EAD"),
            ("EAD_CCF_v2", "EAD", "Credit Card / C&I", 2, "2024-01-01", "2023-11-30", "ACTIVE", "CCF EAD for revolving products"),
        ]
        metadata = pd.DataFrame(meta, columns=["model_id", "model_type", "portfolio", "version", "deployment_date",
                                               "calibration_date", "status", "description"])
        assign = []
        for d in (self.d0, self.d1):
            for p, spec in prods.items():
                assign.append({"as_of_date": d, "product_type": p, "portfolio": spec["portfolio"],
                               "pd_model_id": spec["pd_model"], "lgd_model_id": spec["lgd_model"],
                               "ead_model_id": spec["ead_model"], "model_run_date": d + pd.Timedelta(days=4)})
        params = []
        for kind in ("pd_models", "lgd_models"):
            for mid, spec in self.models[kind].items():
                for k in ("intercept", "base", "floor", "cap"):
                    if k in spec:
                        params.append({"model_id": mid, "parameter": k, "value": spec[k]})
                for f, fs in spec.get("features", {}).items():
                    params.append({"model_id": mid, "parameter": f"{f}.coef", "value": fs["coef"]})
                    params.append({"model_id": mid, "parameter": f"{f}.center", "value": fs["center"]})
                for v, b in spec.get("macro_sensitivity", {}).items():
                    params.append({"model_id": mid, "parameter": f"macro.{v}", "value": b})
        return metadata, pd.DataFrame(assign), pd.DataFrame(params)

    # ---------------------------------------------------------------- main
    def generate(self) -> GeneratedData:
        july, jcps = self.build_july()
        aug, acps, events = self.evolve_to_august(july, jcps)
        loans = pd.concat([july, aug], ignore_index=True).drop(columns=["_age"])
        cps = pd.concat([jcps, acps], ignore_index=True)
        meta, assign, params = self.build_model_reference()
        weights = self.story["scenario_weights"]
        # intended outcome of the story; a variant states its own under `design:` (checked against the observed run)
        intent = self.story.get("design") or {"primary_driver": "macroeconomic", "primary_portfolio": "CRE",
                                              "primary_macro_variable": "unemployment_rate"}
        gt = {
            "seed": self.seed,
            "story_variant": config.story_variant(),
            "previous_date": f"{self.d0:%Y-%m-%d}",
            "current_date": f"{self.d1:%Y-%m-%d}",
            "design": {
                **intent,
                "model_changed": False,
                "scenario_weights_changed": weights[f"{self.d0:%Y-%m-%d}"] != weights[f"{self.d1:%Y-%m-%d}"],
                "overlays_changed": False,
                "largest_borrower": self.cohort["largest"],
                "deteriorated_cre_borrowers": list(self.cohort["ids"]),
                "new_cre_exposure": self.story["new_cre_exposure"],
            },
            "events": events,
        }
        return GeneratedData(loans=loans, counterparties=cps, macro=self.build_macro(),
                             scenario_weights=self.build_weights(), overlays=self.build_overlays(),
                             model_metadata=meta, model_assignments=assign, model_parameters=params,
                             ground_truth=gt)


def to_jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
