"""Credit Agent tools: borrower-level deterioration, PD / LGD / rating migration.

PD has two flavours in the results:
* ``pd_base_annual`` — scorecard PD with no macro adjustment (borrower credit quality only)
* ``pd_12m`` — scenario-weighted 12-month PD including the macro forecast

Comparing both lets the agent separate "the borrower got worse" from "the
economy got worse". Feature-level explanations use the scorecard's logit
contributions, which are exactly additive in logit space.
"""
from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.cecl.engine import pd_logit_components
from src.tools.base import (Framework, ToolError, driver_labels, driver_of_column, ew_mean, get_context,
                            pct_change, tool)

CREDIT_AGENTS = ["orchestrator", "credit_agent"]
Driver = Literal["total", "credit_deterioration", "macroeconomic", "portfolio_growth", "paydowns_runoff",
                 "model_methodology"]
ATTR_FIELDS = ["current_balance", "remaining_term_months", "fico", "dti", "ltv", "dscr", "leverage", "utilization", "rating_notch",
               "internal_rating", "watchlist_flag", "days_past_due", "delinquency_bucket", "default_flag"]


def _continuing(previous_date: str, current_date: str, framework: str, portfolio: str | None):
    ctx = get_context()
    r0 = ctx.results(previous_date, framework).set_index("loan_id")
    r1 = ctx.results(current_date, framework).set_index("loan_id")
    ids = r0.index.intersection(r1.index)
    if portfolio:
        ids = ids[r0.loc[ids, "portfolio"].eq(portfolio).to_numpy()]
        if len(ids) == 0:
            raise ToolError(f"No continuing loans in portfolio {portfolio}", hint={"portfolios": ctx.portfolios})
    return r0.loc[ids], r1.loc[ids]


def _missing(v: Any) -> bool:
    return v is None or (not isinstance(v, str) and bool(pd.isna(v)))


def reason_codes(prev: pd.Series | None, curr: pd.Series | None) -> list[str]:
    """Deterministic, machine-readable explanation of what changed on a loan."""
    if prev is None:
        return ["NEW_LOAN"]
    if curr is None:
        return ["LOAN_EXITED"]
    codes: list[str] = []

    def num(s, k):
        v = s.get(k)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    n0, n1 = num(prev, "rating_notch"), num(curr, "rating_notch")
    if n0 is not None and n1 is not None and n1 != n0:
        codes.append(f"RATING_{'DOWNGRADE' if n1 > n0 else 'UPGRADE'}_{abs(int(n1 - n0))}_NOTCH"
                     f"({prev.get('internal_rating')}->{curr.get('internal_rating')})")
    for k, label, thr, worse_if_down in (("dscr", "DSCR", 0.02, True), ("ltv", "LTV", 0.01, False),
                                         ("fico", "FICO", 10, True), ("dti", "DTI", 0.02, False),
                                         ("utilization", "UTILIZATION", 0.05, False), ("leverage", "LEVERAGE", 0.2, False)):
        a, b = num(prev, k), num(curr, k)
        if a is not None and b is not None and abs(b - a) >= thr:
            worse = (b < a) if worse_if_down else (b > a)
            codes.append(f"{label}_{'WORSE' if worse else 'BETTER'}({a:.2f}->{b:.2f})")
    if (num(prev, "watchlist_flag") or 0) == 0 and (num(curr, "watchlist_flag") or 0) == 1:
        codes.append("ADDED_TO_WATCHLIST")
    if (num(prev, "watchlist_flag") or 0) == 1 and (num(curr, "watchlist_flag") or 0) == 0:
        codes.append("REMOVED_FROM_WATCHLIST")
    d0, d1 = num(prev, "days_past_due") or 0, num(curr, "days_past_due") or 0
    if prev.get("delinquency_bucket") != curr.get("delinquency_bucket"):
        codes.append(f"DELINQUENCY_{'WORSE' if d1 > d0 else 'BETTER'}({prev.get('delinquency_bucket')}->{curr.get('delinquency_bucket')})")
    if (num(prev, "default_flag") or 0) == 0 and (num(curr, "default_flag") or 0) == 1:
        codes.append("DEFAULTED")
    t0, t1 = num(prev, "remaining_term_months"), num(curr, "remaining_term_months")
    if t0 is not None and t1 is not None and t1 > t0:
        codes.append(f"TERM_EXTENDED_RENEWAL({t0:.0f}->{t1:.0f}_MONTHS)")
    b0, b1 = num(prev, "current_balance") or 0, num(curr, "current_balance") or 0
    if b0 and abs(b1 - b0) / b0 >= 0.10:
        codes.append(f"BALANCE_{'UP' if b1 > b0 else 'DOWN'}_{abs(100 * (b1 - b0) / b0):.0f}PCT")
    return codes or ["NO_ATTRIBUTE_CHANGE"]


def merge_reason_codes(per_loan: list[list[str]]) -> list[str]:
    """Combine loan-level codes for a borrower: one entry per code type, noting how many loans it hit."""
    seen: dict[str, tuple[str, int]] = {}
    for codes in per_loan:
        for c in dict.fromkeys(codes):
            key = re.split(r"_\d|\(", c)[0]          # e.g. BALANCE_UP_17PCT -> BALANCE_UP
            first, k = seen.get(key, (c, 0))
            seen[key] = (first, k + 1)
    out = [first if k == 1 else f"{first} [{k} loans]" for first, k in seen.values()]
    if len(out) > 1 and "NO_ATTRIBUTE_CHANGE" in seen:
        out = [c for c in out if not c.startswith("NO_ATTRIBUTE_CHANGE")]
    return out


@tool("credit", CREDIT_AGENTS)
def calculate_pd_migration(previous_date: str, current_date: str, framework: Framework = "CECL",
                           portfolio: str | None = None) -> dict[str, Any]:
    """PD migration on continuing loans by portfolio: 12-month PD (with macro) and base scorecard PD (credit quality only), EAD-weighted, plus counts of loans whose PD rose materially."""
    r0, r1 = _continuing(previous_date, current_date, framework, portfolio)
    w = r0["ead"]                                   # previous EAD weights: mix-neutral comparison

    def stats(idx) -> dict[str, Any]:
        a, b, ww = r0.loc[idx], r1.loc[idx], w.loc[idx]
        pd0, pd1 = ew_mean(a["pd_12m"], ww), ew_mean(b["pd_12m"], ww)
        bp0, bp1 = ew_mean(a["pd_base_annual"], ww), ew_mean(b["pd_base_annual"], ww)
        ratio = b["pd_12m"] / a["pd_12m"].replace(0, np.nan)
        return {"loans": int(len(idx)), "ead": float(ww.sum()),
                "pd_12m_previous": pd0, "pd_12m_current": pd1,
                "pd_12m_change_bps": None if pd0 is None else 1e4 * (pd1 - pd0),
                "base_pd_previous": bp0, "base_pd_current": bp1,
                "base_pd_change_bps": None if bp0 is None else 1e4 * (bp1 - bp0),
                "loans_pd_up_50pct": int((ratio >= 1.5).sum()), "loans_pd_doubled": int((ratio >= 2.0).sum()),
                "loans_pd_down": int((ratio < 1.0).sum())}

    rows = [{"portfolio": port, **stats(g.index)} for port, g in r0.groupby("portfolio")]
    rows.sort(key=lambda r: -(r["pd_12m_change_bps"] or 0))
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date,
            "portfolio_filter": portfolio, "population": "continuing loans (in both snapshots)",
            "weighting": "previous-period EAD for both dates (isolates PD from mix changes)",
            "total": stats(r0.index), "by_portfolio": rows,
            "largest_pd_increase_portfolio": rows[0]["portfolio"] if rows else None,
            "interpretation_guide": {"base_pd": "scorecard PD without macro: moves only with borrower attributes",
                                     "pd_12m": "scenario-weighted PD including the macro forecast",
                                     "macro_effect_hint": "pd_12m change much larger than base_pd change => macro-driven"}}


@tool("credit", CREDIT_AGENTS)
def calculate_lgd_migration(previous_date: str, current_date: str, framework: Framework = "CECL",
                            portfolio: str | None = None) -> dict[str, Any]:
    """LGD migration on continuing loans by portfolio (EAD-weighted, first-year scenario-weighted LGD), with the change in balance-weighted LTV for secured portfolios."""
    ctx = get_context()
    r0, r1 = _continuing(previous_date, current_date, framework, portfolio)
    L0 = ctx.store.loans_at(previous_date).set_index("loan_id").loc[r0.index]
    L1 = ctx.store.loans_at(current_date).set_index("loan_id").loc[r0.index]
    w = r0["ead"]
    rows = []
    for port, g in r0.groupby("portfolio"):
        i = g.index
        l0, l1 = ew_mean(r0.loc[i, "lgd"], w.loc[i]), ew_mean(r1.loc[i, "lgd"], w.loc[i])
        t0, t1 = ew_mean(L0.loc[i, "ltv"], L0.loc[i, "current_balance"]), ew_mean(L1.loc[i, "ltv"], L0.loc[i, "current_balance"])
        rows.append({"portfolio": port, "loans": len(i), "lgd_previous": l0, "lgd_current": l1,
                     "lgd_change_pp": None if l0 is None else 100 * (l1 - l0),
                     "ltv_previous": t0, "ltv_current": t1,
                     "ltv_change_pp": None if t0 is None else 100 * (t1 - t0),
                     "lgd_model_ids": sorted(set(r1.loc[i, "lgd_model_id"]))})
    rows.sort(key=lambda r: -(r["lgd_change_pp"] or 0))
    l0, l1 = ew_mean(r0["lgd"], w), ew_mean(r1["lgd"], w)
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date,
            "portfolio_filter": portfolio, "population": "continuing loans",
            "total": {"lgd_previous": l0, "lgd_current": l1, "lgd_change_pp": 100 * (l1 - l0)},
            "by_portfolio": rows,
            "notes": "LGD moves with LTV (collateral re-appraisals = credit) and with HPI / CRE price paths (= macro)."}


@tool("credit", CREDIT_AGENTS)
def calculate_rating_migration(previous_date: str, current_date: str, portfolio: str | None = None,
                               n: int = 10) -> dict[str, Any]:
    """Internal rating migration for commercial loans (C&I, CRE): upgrades / downgrades with counts and balance, the rating transition matrix (non-zero cells) and the largest downgrades."""
    ctx = get_context()
    P = ctx.store.loans_at(previous_date).set_index("loan_id")
    C = ctx.store.loans_at(current_date).set_index("loan_id")
    ids = P.index.intersection(C.index)
    P, C = P.loc[ids], C.loc[ids]
    m = P["rating_notch"].notna() & C["rating_notch"].notna()
    if portfolio:
        m &= P["portfolio"].eq(portfolio)
    if not m.any():
        raise ToolError("No rated (commercial) continuing loans for this filter",
                        hint={"rated_portfolios": sorted(P.loc[P['rating_notch'].notna(), 'portfolio'].unique())})
    P, C = P[m], C[m]
    delta = (C["rating_notch"] - P["rating_notch"]).astype(int)
    bal = C["current_balance"]
    cells = (pd.DataFrame({"from": P["internal_rating"], "to": C["internal_rating"],
                           "from_notch": P["rating_notch"], "to_notch": C["rating_notch"], "balance": bal})
             .groupby(["from_notch", "from", "to_notch", "to"]).agg(loans=("balance", "size"), balance=("balance", "sum"))
             .reset_index())
    by_port = []
    for port in sorted(P["portfolio"].unique()):
        k = P["portfolio"].eq(port)
        d = delta[k]
        by_port.append({"portfolio": port, "rated_loans": int(k.sum()),
                        "downgrades": int((d > 0).sum()), "upgrades": int((d < 0).sum()),
                        "downgraded_balance": float(bal[k][d > 0].sum()), "upgraded_balance": float(bal[k][d < 0].sum()),
                        "downgrades_2plus_notches": int((d >= 2).sum()),
                        "net_notches_balance_weighted": ew_mean(d[k].astype(float), bal[k])})
    down = (pd.DataFrame({"counterparty_id": C["counterparty_id"], "portfolio": C["portfolio"],
                          "from": P["internal_rating"], "to": C["internal_rating"], "notches": delta, "balance": bal})
            [delta > 0].sort_values(["notches", "balance"], ascending=False).head(n))
    return {"previous_date": previous_date, "current_date": current_date, "portfolio_filter": portfolio,
            "scale": "internal notch 1 = AAA ... 17 = CCC; positive notches = downgrade",
            "summary": {"rated_loans": int(len(delta)), "downgrades": int((delta > 0).sum()),
                        "upgrades": int((delta < 0).sum()), "unchanged": int((delta == 0).sum()),
                        "downgraded_balance": float(bal[delta > 0].sum())},
            "by_portfolio": by_port,
            "transition_cells": cells.sort_values(["from_notch", "to_notch"]).to_dict("records"),
            "largest_downgrades": [{"loan_id": i, **r.to_dict()} for i, r in down.iterrows()]}


def _contribution_frame(previous_date: str, current_date: str, framework: str) -> pd.DataFrame:
    ctx = get_context()
    attr = ctx.attribution(previous_date, current_date, framework)
    lc = attr.loan_contributions.copy()
    col_driver = driver_of_column(attr)
    for d in dict.fromkeys(col_driver.values()):
        lc[f"drv_{d}"] = lc[[c for c, dd in col_driver.items() if dd == d and c in lc.columns]].sum(axis=1)
    return lc


@tool("credit", CREDIT_AGENTS + ["portfolio_agent"])
def identify_top_contributors(previous_date: str, current_date: str, framework: Framework = "CECL",
                              n: int = 10, level: Literal["counterparty", "loan"] = "counterparty",
                              driver: Driver = "total", portfolio: str | None = None,
                              direction: Literal["increase", "decrease"] = "increase") -> dict[str, Any]:
    """Rank borrowers (or loans) by allowance change, optionally by one driver (e.g. credit_deterioration), with balance, ECL, PD and LGD before/after, the driver split and machine-readable reason codes."""
    ctx = get_context()
    n = max(1, min(int(n), 50))
    lc = _contribution_frame(previous_date, current_date, framework)
    if portfolio:
        lc = lc[lc["portfolio"] == portfolio]
    r0 = ctx.results(previous_date, framework).set_index("loan_id")
    r1 = ctx.results(current_date, framework).set_index("loan_id")
    L0 = ctx.store.loans_at(previous_date).set_index("loan_id")
    L1 = ctx.store.loans_at(current_date).set_index("loan_id")
    drivers = list(driver_labels())
    metric = "ecl_change" if driver == "total" else f"drv_{driver}"
    lc = lc.assign(bal0=r0["current_balance"].reindex(lc.index), bal1=r1["current_balance"].reindex(lc.index),
                   ead0=r0["ead"].reindex(lc.index), ead1=r1["ead"].reindex(lc.index),
                   pd0=r0["pd_12m"].reindex(lc.index), pd1=r1["pd_12m"].reindex(lc.index),
                   lgd0=r0["lgd"].reindex(lc.index), lgd1=r1["lgd"].reindex(lc.index))
    sums = ["bal0", "bal1", "ecl_previous", "ecl_current", "ecl_change"] + [f"drv_{d}" for d in drivers]
    if level == "counterparty":
        agg = lc.groupby("counterparty_id")[sums].sum()
    else:
        agg = lc[sums].copy()
    order = agg[metric].sort_values(ascending=(direction == "decrease"))
    top = agg.loc[order.index[:n]]
    cps = ctx.store.counterparties_at(current_date).set_index("counterparty_id")
    cps0 = ctx.store.counterparties_at(previous_date).set_index("counterparty_id")
    total_metric = float(lc[metric].sum())
    by_cp = lc.index.to_series().groupby(lc["counterparty_id"]).agg(list) if level == "counterparty" else None
    rows = []
    for key_id, row in top.iterrows():
        loan_ids = by_cp[key_id] if level == "counterparty" else [key_id]
        x = lc.loc[loan_ids]
        cp_id = key_id if level == "counterparty" else x["counterparty_id"].iloc[0]
        name = cps["customer_name"].get(cp_id, cps0["customer_name"].get(cp_id))
        codes = merge_reason_codes([reason_codes(L0.loc[lid] if lid in L0.index else None,
                                                 L1.loc[lid] if lid in L1.index else None) for lid in loan_ids])
        drv = {d: float(row[f"drv_{d}"]) for d in drivers}
        ptypes = [L1.loc[l, "property_type"] for l in loan_ids if l in L1.index and isinstance(L1.loc[l, "property_type"], str)]
        rows.append({
            "rank": len(rows) + 1, "id": key_id, "counterparty_id": cp_id, "customer_name": name,
            "portfolio": ", ".join(sorted(set(x["portfolio"]))), "loans": int(len(x)),
            "property_type": ptypes[0] if ptypes else None,
            "balance_previous": row["bal0"], "balance_current": row["bal1"],
            "ecl_previous": row["ecl_previous"], "ecl_current": row["ecl_current"], "ecl_change": row["ecl_change"],
            "pd_12m_previous": ew_mean(x["pd0"], x["ead0"]), "pd_12m_current": ew_mean(x["pd1"], x["ead1"]),
            "lgd_previous": ew_mean(x["lgd0"], x["ead0"]), "lgd_current": ew_mean(x["lgd1"], x["ead1"]),
            "driver_split": drv, "main_driver": max(drv, key=lambda d: abs(drv[d])),
            "share_of_ranked_metric_pct": 100 * row[metric] / total_metric if total_metric else None,
            "reason_codes": codes,
        })
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date,
            "level": level, "ranked_by": metric, "direction": direction, "portfolio_filter": portfolio,
            "currency": "USD", "population_total": total_metric,
            "top_n_total": float(top[metric].sum()), "top_n_share_pct": 100 * float(top[metric].sum()) / total_metric if total_metric else None,
            "contributors": rows}


def _loan_dates(loan_id: str, as_of_date: str | None) -> list[str]:
    ctx = get_context()
    return [as_of_date] if as_of_date else ctx.dates


@tool("credit", CREDIT_AGENTS + ["portfolio_agent"])
def get_loan_details(loan_id: str, as_of_date: str | None = None, framework: Framework = "CECL") -> dict[str, Any]:
    """Attributes and ECL results for one loan on each reporting date (or one date): balances, risk fields, PD, LGD, EAD, ECL by scenario and model IDs."""
    ctx = get_context()
    loan_id = loan_id.strip().upper()
    snaps = []
    for d in _loan_dates(loan_id, as_of_date):
        l = ctx.store.loans_at(d)
        row = l[l["loan_id"] == loan_id]
        if row.empty:
            snaps.append({"as_of_date": d, "on_book": False})
            continue
        r = ctx.results(d, framework)
        res = r[r["loan_id"] == loan_id].iloc[0]
        keep = ["counterparty_id", "portfolio", "product_type", "property_type", "industry", "state",
                "origination_date", "maturity_date", "remaining_term_months", "original_balance", "credit_limit",
                "interest_rate", "collateral_value"] + ATTR_FIELDS + ["pd_model_id", "lgd_model_id", "ead_model_id"]
        snaps.append({"as_of_date": d, "on_book": True, **row.iloc[0][[k for k in keep if k in row.columns]].to_dict(),
                      "results": res[["ead", "pd_base_annual", "pd_12m", "lifetime_pd", "lgd", "horizon_years",
                                      "ecl", "ecl_baseline", "ecl_adverse", "ecl_severe_adverse", "ifrs9_stage"]].to_dict()})
    if not any(s["on_book"] for s in snaps):
        raise ToolError(f"Loan {loan_id} not found", "not_found",
                        {"example_ids": ctx.store.loans_at(ctx.dates[-1])["loan_id"].head(3).tolist()})
    return {"loan_id": loan_id, "framework": framework, "snapshots": snaps}


@tool("credit", CREDIT_AGENTS + ["portfolio_agent"])
def get_counterparty_details(counterparty_id: str, as_of_date: str | None = None,
                             framework: Framework = "CECL") -> dict[str, Any]:
    """Borrower profile (name, type, industry, financials or FICO/income, ratings, watchlist) and its loans with balance, PD, LGD and ECL on each reporting date."""
    ctx = get_context()
    cp_id = counterparty_id.strip().upper()
    snaps = []
    for d in _loan_dates(cp_id, as_of_date):
        cps = ctx.store.counterparties_at(d)
        row = cps[cps["counterparty_id"] == cp_id]
        r = ctx.results(d, framework)
        loans = r[r["counterparty_id"] == cp_id]
        if row.empty and loans.empty:
            snaps.append({"as_of_date": d, "found": False})
            continue
        prof = row.iloc[0].drop(labels=["as_of_date"]).to_dict() if not row.empty else {}
        snaps.append({"as_of_date": d, "found": True, "profile": prof,
                      "total_balance": float(loans["current_balance"].sum()), "total_ecl": float(loans["ecl"].sum()),
                      "loans": loans[["loan_id", "portfolio", "product_type", "current_balance", "ead", "pd_12m", "lgd",
                                      "ecl", "pd_model_id"]].to_dict("records")})
    if not any(s["found"] for s in snaps):
        raise ToolError(f"Counterparty {cp_id} not found", "not_found",
                        {"hint": "use identify_top_contributors to find borrower IDs"})
    return {"counterparty_id": cp_id, "framework": framework, "snapshots": snaps}


@tool("credit", CREDIT_AGENTS)
def compare_borrower_periods(counterparty_id: str, previous_date: str, current_date: str,
                             framework: Framework = "CECL") -> dict[str, Any]:
    """Explain one borrower's allowance change: attribute changes per loan, the PD scorecard logit change by feature (exactly additive), base vs 12-month PD, LGD, ECL and the attribution-step split."""
    ctx = get_context()
    cp_id = counterparty_id.strip().upper()
    L0 = ctx.store.loans_at(previous_date)
    L1 = ctx.store.loans_at(current_date)
    l0, l1 = L0[L0["counterparty_id"] == cp_id].set_index("loan_id"), L1[L1["counterparty_id"] == cp_id].set_index("loan_id")
    if l0.empty and l1.empty:
        raise ToolError(f"Counterparty {cp_id} has no loans on these dates", "not_found",
                        {"hint": "use identify_top_contributors to find borrower IDs"})
    r0 = ctx.results(previous_date, framework).set_index("loan_id")
    r1 = ctx.results(current_date, framework).set_index("loan_id")
    lc = _contribution_frame(previous_date, current_date, framework)
    drivers = list(driver_labels())
    comps0 = pd_logit_components(l0) if len(l0) else pd.DataFrame()
    comps1 = pd_logit_components(l1) if len(l1) else pd.DataFrame()
    loans = []
    for lid in l0.index.union(l1.index):
        p = l0.loc[lid] if lid in l0.index else None
        c = l1.loc[lid] if lid in l1.index else None
        attrs = {}
        for k in ATTR_FIELDS:
            a = None if p is None else p.get(k)
            b = None if c is None else c.get(k)
            if not (_missing(a) and _missing(b)):        # skip fields that don't apply to this product
                attrs[k] = {"previous": a, "current": b}
        logit = None
        if p is not None and c is not None:
            a, b = comps0.loc[lid], comps1.loc[lid]
            feats = [f for f in a.index if f != "intercept"]
            logit = {"total_change": float(b.sum() - a.sum()),
                     "by_feature": sorted(({"feature": f, "previous": float(a[f]), "current": float(b[f]),
                                            "change": float(b[f] - a[f])} for f in feats),
                                          key=lambda x: -abs(x["change"])),
                     "pd_model_previous": p["pd_model_id"], "pd_model_current": c["pd_model_id"]}
        res = {}
        for tag, r in (("previous", r0), ("current", r1)):
            if lid in r.index:
                res[tag] = r.loc[lid, ["ead", "pd_base_annual", "pd_12m", "lgd", "ecl"]].to_dict()
        lrow = lc.loc[lid] if lid in lc.index else None
        loans.append({"loan_id": lid, "portfolio": (c if c is not None else p)["portfolio"],
                      "product_type": (c if c is not None else p)["product_type"],
                      "status": "CONTINUING" if p is not None and c is not None else ("NEW" if p is None else "EXITED"),
                      "attributes": attrs, "pd_logit_change": logit, "results": res,
                      "ecl_change": None if lrow is None else float(lrow["ecl_change"]),
                      "driver_split": None if lrow is None else {d: float(lrow[f"drv_{d}"]) for d in drivers},
                      "reason_codes": reason_codes(p, c)})
    tot = {d: sum((x["driver_split"] or {}).get(d, 0.0) for x in loans) for d in drivers}
    cps = ctx.store.counterparties_at(current_date)
    name = cps.loc[cps["counterparty_id"] == cp_id, "customer_name"]
    return {"counterparty_id": cp_id, "customer_name": name.iloc[0] if len(name) else None,
            "framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "ecl_previous": float(r0.loc[r0.index.intersection(l0.index), "ecl"].sum()),
            "ecl_current": float(r1.loc[r1.index.intersection(l1.index), "ecl"].sum()),
            "ecl_change": float(sum(x["ecl_change"] or 0.0 for x in loans)),
            "driver_split": tot, "main_driver": max(tot, key=lambda d: abs(tot[d])) if tot else None,
            "loans": loans,
            "notes": "pd_logit_change.by_feature: scorecard contribution coef*(x-center) per feature; changes sum to the base-PD logit change. Macro effects are not in the base logit."}
