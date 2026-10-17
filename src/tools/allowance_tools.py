"""Allowance-level tools: official allowance, period comparison, engine re-run."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import config
from src.tools.base import Framework, ToolError, ew_mean, get_context, pct_change, tool

ALL = ["orchestrator", "portfolio_agent", "credit_agent", "macro_agent", "model_risk_agent", "reconciliation_agent"]


def _by_portfolio(as_of_date: str, framework: str) -> pd.DataFrame:
    ctx = get_context()
    r = ctx.results(as_of_date, framework)
    g = r.groupby("portfolio")
    out = pd.DataFrame({
        "loan_count": g.size(),
        "balance": g["current_balance"].sum(),
        "ead": g["ead"].sum(),
        "modeled_ecl": g["ecl"].sum(),
    })
    out["ead_weighted_pd_12m"] = g.apply(lambda x: ew_mean(x["pd_12m"], x["ead"]), include_groups=False)
    out["ead_weighted_lgd"] = g.apply(lambda x: ew_mean(x["lgd"], x["ead"]), include_groups=False)
    ov = ctx.store.overlays_at(as_of_date).groupby("portfolio")["amount"].sum()
    out["overlay"] = ov.reindex(out.index).fillna(0.0)
    out["allowance"] = out["modeled_ecl"] + out["overlay"]
    out["coverage_pct"] = 100.0 * out["allowance"] / out["balance"]
    return out


@tool("allowance", ALL)
def get_reporting_context() -> dict[str, Any]:
    """List what can be analysed: reporting dates, frameworks, portfolios, macro variables, scenarios and model IDs.

    Call this first if you are unsure which dates or names are valid.
    """
    ctx = get_context()
    return {"reporting_dates": ctx.dates, "frameworks": ["CECL", "IFRS9"], "default_framework": "CECL",
            "portfolios": ctx.portfolios, "product_types": ctx.product_types,
            "macro_variables": ctx.macro_variables, "scenarios": ctx.scenarios, "model_ids": ctx.model_ids,
            "currency": "USD"}


@tool("allowance", ALL)
def get_allowance_summary(as_of_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Official allowance for one reporting date: total, modeled ECL, overlays, balance, coverage, average PD/LGD, scenario ECL and a portfolio breakdown."""
    ctx = get_context()
    row = ctx.summary_row(as_of_date, framework)
    scen = [s for s in ctx.scenarios if f"scenario_ecl_{s}" in row]
    return {
        "as_of_date": as_of_date, "framework": framework, "run_id": row["run_id"],
        "scenario_set_id": row["scenario_set_id"], "currency": "USD",
        "allowance": row["allowance"], "modeled_ecl": row["modeled_ecl"], "overlay_total": row["overlay_total"],
        "total_balance": row["total_balance"], "total_ead": row["total_ead"], "loan_count": row["loan_count"],
        "coverage_pct": row["allowance_rate_pct"],
        "ead_weighted_pd_12m": row["ead_weighted_pd_12m"], "ead_weighted_lgd": row["ead_weighted_lgd"],
        "scenarios": [{"scenario": s, "weight": row[f"weight_{s}"], "unweighted_ecl": row[f"scenario_ecl_{s}"]}
                      for s in scen],
        "by_portfolio": _by_portfolio(as_of_date, framework).reset_index().to_dict("records"),
    }


@tool("allowance", ALL)
def compare_allowance(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Compare the official allowance between two reporting dates: dollar and percent change in total and by portfolio, plus balance, coverage, average PD and LGD."""
    ctx = get_context()
    p, c = ctx.summary_row(previous_date, framework), ctx.summary_row(current_date, framework)
    bp, bc = _by_portfolio(previous_date, framework), _by_portfolio(current_date, framework)
    ports = bp.index.union(bc.index)
    rows = []
    for port in ports:
        a0 = float(bp["allowance"].get(port, 0.0))
        a1 = float(bc["allowance"].get(port, 0.0))
        rows.append({"portfolio": port, "previous_allowance": a0, "current_allowance": a1, "change": a1 - a0,
                     "change_pct": pct_change(a0, a1),
                     "previous_coverage_pct": bp["coverage_pct"].get(port), "current_coverage_pct": bc["coverage_pct"].get(port)})
    change = float(c["allowance"] - p["allowance"])
    for r in rows:
        r["share_of_total_change_pct"] = 100.0 * r["change"] / change if change else None
    rows.sort(key=lambda r: -abs(r["change"]))
    return {
        "framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
        "previous_allowance": p["allowance"], "current_allowance": c["allowance"],
        "change": change, "change_pct": pct_change(p["allowance"], c["allowance"]),
        "direction": "increase" if change > 0 else "decrease" if change < 0 else "unchanged",
        "previous_balance": p["total_balance"], "current_balance": c["total_balance"],
        "previous_coverage_pct": p["allowance_rate_pct"], "current_coverage_pct": c["allowance_rate_pct"],
        "previous_ead_weighted_pd_12m": p["ead_weighted_pd_12m"], "current_ead_weighted_pd_12m": c["ead_weighted_pd_12m"],
        "previous_ead_weighted_lgd": p["ead_weighted_lgd"], "current_ead_weighted_lgd": c["ead_weighted_lgd"],
        "previous_loan_count": p["loan_count"], "current_loan_count": c["loan_count"],
        "by_portfolio": rows,
        "largest_portfolio_change": rows[0]["portfolio"] if rows else None,
        "source": {"previous_run_id": p["run_id"], "current_run_id": c["run_id"]},
    }


@tool("allowance", ["orchestrator", "reconciliation_agent", "model_risk_agent"])
def calculate_ecl(as_of_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Re-run the ECL engine from the stored inputs for one date and check it reproduces the official reported allowance."""
    ctx = get_context()
    res = ctx.engine.run(ctx.store.engine_inputs(as_of_date), framework)
    row = ctx.summary_row(as_of_date, framework)
    diff = float(res.allowance - row["allowance"])
    tol = float(config.reconciliation()["absolute_tolerance"])
    return {
        "as_of_date": as_of_date, "framework": framework, "currency": "USD",
        "recomputed_allowance": res.allowance, "recomputed_modeled_ecl": res.total_ecl,
        "recomputed_overlay_total": res.overlay_total, "reported_allowance": row["allowance"],
        "difference": diff, "reproduces_reported": abs(diff) <= tol, "tolerance": tol,
        "scenario_ecl": res.scenario_ecl, "scenario_weights": res.weights,
        "loan_count": int(len(res.loans)), "reported_run_id": row["run_id"],
    }


@tool("allowance", ["orchestrator", "portfolio_agent", "credit_agent"])
def get_portfolio_snapshot(as_of_date: str, portfolio: str | None = None) -> dict[str, Any]:
    """Book composition on one date: loans, balance, undrawn, EAD, risk-attribute averages (FICO, LTV, DSCR, rating) and delinquency mix by portfolio."""
    ctx = get_context()
    l = ctx.store.loans_at(as_of_date)
    r = ctx.results(as_of_date, "CECL")[["loan_id", "ead"]]
    l = l.merge(r, on="loan_id", how="left")
    if portfolio:
        l = l[l["portfolio"] == portfolio]
        if l.empty:
            raise ToolError(f"No loans in portfolio {portfolio} on {as_of_date}", hint={"portfolios": ctx.portfolios})
    rows = []
    for port, x in l.groupby("portfolio"):
        bal = x["current_balance"]
        rows.append({
            "portfolio": port, "loan_count": len(x), "balance": bal.sum(), "undrawn": x["undrawn_amount"].fillna(0).sum(),
            "ead": x["ead"].sum(), "avg_balance": bal.mean(),
            "balance_weighted_fico": ew_mean(x["fico"], bal), "balance_weighted_ltv": ew_mean(x["ltv"], bal),
            "balance_weighted_dscr": ew_mean(x["dscr"], bal), "balance_weighted_rating_notch": ew_mean(x["rating_notch"], bal),
            "balance_weighted_utilization": ew_mean(x["utilization"], bal),
            "watchlist_loans": int(x["watchlist_flag"].fillna(0).sum()),
            "defaulted_loans": int(x["default_flag"].fillna(0).sum()),
            "delinquency_mix": {k: int(v) for k, v in x["delinquency_bucket"].value_counts().items()},
        })
    rows.sort(key=lambda r: -r["balance"])
    return {"as_of_date": as_of_date, "portfolio_filter": portfolio, "currency": "USD",
            "total_loans": int(len(l)), "total_balance": float(l["current_balance"].sum()),
            "total_ead": float(l["ead"].sum()), "by_portfolio": rows,
            "notes": {"rating_notch": "1 = AAA ... 17 = CCC (higher is worse)",
                      "fields_not_applicable": "FICO/DTI apply to retail; DSCR/rating to commercial; null when not applicable"}}
