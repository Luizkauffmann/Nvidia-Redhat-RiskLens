"""Portfolio Agent tools: exposure movements and the allowance attribution views.

Exposure facts come from the loan snapshots; allowance impacts come from the
sequential counterfactual attribution (src/cecl/attribution.py), so every
dollar quoted here is part of the same reconciled walk.
"""
from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from src.tools.base import (Framework, ToolError, driver_labels, driver_of_column, ew_mean, get_context,
                            pct_change, tool)

PORTFOLIO_AGENTS = ["orchestrator", "portfolio_agent"]


def _loan_sets(previous_date: str, current_date: str, portfolio: str | None = None):
    ctx = get_context()
    P = ctx.store.loans_at(previous_date).set_index("loan_id")
    C = ctx.store.loans_at(current_date).set_index("loan_id")
    if portfolio:
        P, C = P[P["portfolio"] == portfolio], C[C["portfolio"] == portfolio]
        if P.empty and C.empty:
            raise ToolError(f"No loans in portfolio {portfolio}", hint={"portfolios": ctx.portfolios})
    cont = P.index.intersection(C.index)
    return P, C, cont, P.index.difference(C.index), C.index.difference(P.index)


def _exit_reason(P: pd.DataFrame, exits: pd.Index, current_date: str) -> pd.Series:
    """Inferred from the previous snapshot: matured by the current date, defaulted (charge-off) or paid off early."""
    x = P.loc[exits]
    matured = x["maturity_date"] <= pd.Timestamp(current_date)
    defaulted = x["default_flag"].fillna(0).eq(1)
    return pd.Series(["charged_off" if d else "matured" if m else "paid_off"
                      for m, d in zip(matured, defaulted)], index=exits)


@tool("portfolio", PORTFOLIO_AGENTS)
def compare_portfolios(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Side-by-side portfolio comparison: loan count, balance, EAD, allowance and coverage for both dates, with changes."""
    ctx = get_context()
    out = []
    for d, tag in ((previous_date, "previous"), (current_date, "current")):
        r = ctx.results(d, framework)
        ov = ctx.store.overlays_at(d).groupby("portfolio")["amount"].sum()
        g = r.groupby("portfolio").agg(loan_count=("loan_id", "size"), balance=("current_balance", "sum"),
                                       ead=("ead", "sum"), modeled_ecl=("ecl", "sum"))
        g["allowance"] = g["modeled_ecl"] + ov.reindex(g.index).fillna(0.0)
        out.append(g.add_prefix(f"{tag}_"))
    m = out[0].join(out[1], how="outer").fillna(0.0)
    rows = []
    for port, x in m.iterrows():
        row = {"portfolio": port, **x.to_dict()}
        for k in ("loan_count", "balance", "ead", "allowance"):
            row[f"{k}_change"] = x[f"current_{k}"] - x[f"previous_{k}"]
        row["balance_change_pct"] = pct_change(x["previous_balance"], x["current_balance"])
        row["allowance_change_pct"] = pct_change(x["previous_allowance"], x["current_allowance"])
        row["previous_coverage_pct"] = 100 * x["previous_allowance"] / x["previous_balance"] if x["previous_balance"] else None
        row["current_coverage_pct"] = 100 * x["current_allowance"] / x["current_balance"] if x["current_balance"] else None
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["allowance_change"]))
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "by_portfolio": rows,
            "largest_balance_growth": max(rows, key=lambda r: r["balance_change"])["portfolio"],
            "largest_allowance_increase": max(rows, key=lambda r: r["allowance_change"])["portfolio"]}


@tool("portfolio", PORTFOLIO_AGENTS)
def calculate_balance_change(previous_date: str, current_date: str, portfolio: str | None = None) -> dict[str, Any]:
    """Exact balance bridge: previous balance - exits (matured / paid off / charged off) + net change on continuing loans + new originations = current balance."""
    P, C, cont, exits, new = _loan_sets(previous_date, current_date, portfolio)
    reason = _exit_reason(P, exits, current_date)

    def bridge(port: str | None) -> dict[str, Any]:
        sel = (lambda f: f[f["portfolio"] == port]) if port else (lambda f: f)
        p, c = sel(P), sel(C)
        k = p.index.intersection(c.index)
        e = p.index.difference(c.index)
        n = c.index.difference(p.index)
        b0, b1 = float(p["current_balance"].sum()), float(c["current_balance"].sum())
        delta_cont = c.loc[k, "current_balance"] - p.loc[k, "current_balance"]
        exit_bal = p.loc[e, "current_balance"].groupby(reason.reindex(e)).sum()
        res = {
            "previous_balance": b0,
            "exits_matured": -float(exit_bal.get("matured", 0.0)),
            "exits_paid_off": -float(exit_bal.get("paid_off", 0.0)),
            "exits_charged_off": -float(exit_bal.get("charged_off", 0.0)),
            "continuing_paydowns": float(delta_cont[delta_cont < 0].sum()),
            "continuing_increases": float(delta_cont[delta_cont > 0].sum()),
            "new_originations": float(c.loc[n, "current_balance"].sum()),
            "current_balance": b1,
        }
        res["bridge_check"] = (res["previous_balance"] + sum(v for kk, v in res.items() if kk.startswith(("exits", "continuing", "new")))) - b1
        res["net_change"] = b1 - b0
        res["net_change_pct"] = pct_change(b0, b1)
        res["counts"] = {"previous": len(p), "current": len(c), "continuing": len(k), "exited": len(e), "new": len(n)}
        return res

    ports = sorted(set(P["portfolio"]).union(C["portfolio"]))
    return {"previous_date": previous_date, "current_date": current_date, "portfolio_filter": portfolio,
            "currency": "USD", "total": bridge(None),
            "by_portfolio": [{"portfolio": port, **bridge(port)} for port in ports] if not portfolio else [],
            "notes": "Exit reasons are inferred from the previous snapshot (maturity date passed / defaulted / otherwise paid off). bridge_check should be 0."}


@tool("portfolio", PORTFOLIO_AGENTS)
def calculate_new_originations(previous_date: str, current_date: str, framework: Framework = "CECL",
                               portfolio: str | None = None, n: int = 10) -> dict[str, Any]:
    """New loans booked between the two dates: count, balance, average PD and their allowance impact (the 'new originations' attribution step), by portfolio, with the largest new loans."""
    ctx = get_context()
    _, C, _, _, new = _loan_sets(previous_date, current_date, portfolio)
    attr = ctx.attribution(previous_date, current_date, framework)
    lc = attr.loan_contributions.loc[new]
    res = ctx.results(current_date, framework).set_index("loan_id").loc[new]
    x = C.loc[new].join(res[["ead", "pd_12m", "lgd", "ecl"]]).join(lc[["new_originations"]])
    rows = []
    for port, g in x.groupby("portfolio"):
        rows.append({"portfolio": port, "loan_count": len(g), "balance": g["current_balance"].sum(),
                     "ead": g["ead"].sum(), "ead_weighted_pd_12m": ew_mean(g["pd_12m"], g["ead"]),
                     "ecl_current": g["ecl"].sum(), "allowance_impact": g["new_originations"].sum()})
    rows.sort(key=lambda r: -r["balance"])
    top = x.sort_values("current_balance", ascending=False).head(n)
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "portfolio_filter": portfolio, "loan_count": int(len(x)), "balance": float(x["current_balance"].sum()),
            "allowance_impact": float(x["new_originations"].sum()), "by_portfolio": rows,
            "largest_new_loans": [{"loan_id": i, "counterparty_id": r["counterparty_id"], "portfolio": r["portfolio"],
                                   "product_type": r["product_type"], "property_type": r.get("property_type"),
                                   "balance": r["current_balance"], "pd_12m": r["pd_12m"], "ecl": r["ecl"]}
                                  for i, r in top.iterrows()],
            "attribution_step": "new_originations",
            "notes": "allowance_impact is measured at July macro and scenario weights (sequential walk); later steps capture how macro changes affect these loans."}


@tool("portfolio", PORTFOLIO_AGENTS)
def calculate_paydowns(previous_date: str, current_date: str, framework: Framework = "CECL",
                       portfolio: str | None = None, n: int = 10) -> dict[str, Any]:
    """Runoff: loans that exited (matured, paid off, charged off) and paydowns on continuing loans, with their allowance impact (attribution steps 'runoff_exits' and 'balance_term')."""
    ctx = get_context()
    P, C, cont, exits, _ = _loan_sets(previous_date, current_date, portfolio)
    attr = ctx.attribution(previous_date, current_date, framework)
    lc = attr.loan_contributions
    reason = _exit_reason(P, exits, current_date)
    ex = P.loc[exits, ["counterparty_id", "portfolio", "product_type", "current_balance"]].assign(
        reason=reason, allowance_impact=lc.loc[exits, "runoff_exits"])
    ct = pd.DataFrame({"portfolio": C.loc[cont, "portfolio"],
                       "balance_change": C.loc[cont, "current_balance"] - P.loc[cont, "current_balance"],
                       "allowance_impact": lc.loc[cont, "balance_term"]})
    by_port = []
    for port in sorted(set(ex["portfolio"]).union(ct["portfolio"])):
        e, k = ex[ex["portfolio"] == port], ct[ct["portfolio"] == port]
        by_port.append({"portfolio": port, "exited_loans": len(e), "exited_balance": e["current_balance"].sum(),
                        "exit_allowance_impact": e["allowance_impact"].sum(),
                        "continuing_net_balance_change": k["balance_change"].sum(),
                        "balance_term_allowance_impact": k["allowance_impact"].sum()})
    by_reason = [{"reason": r, "loans": len(g), "balance": g["current_balance"].sum(),
                  "allowance_impact": g["allowance_impact"].sum()} for r, g in ex.groupby("reason")]
    top = ex.sort_values("current_balance", ascending=False).head(n)
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "portfolio_filter": portfolio,
            "exits": {"loans": int(len(ex)), "balance": float(ex["current_balance"].sum()),
                      "allowance_impact": float(ex["allowance_impact"].sum()), "by_reason": by_reason},
            "continuing_loans": {"net_balance_change": float(ct["balance_change"].sum()),
                                 "allowance_impact": float(ct["allowance_impact"].sum())},
            "total_runoff_allowance_impact": float(ex["allowance_impact"].sum() + ct["allowance_impact"].sum()),
            "by_portfolio": by_port,
            "largest_exits": [{"loan_id": i, **r.to_dict()} for i, r in top.iterrows()],
            "notes": "balance_term also includes term/renewal effects: renewed commercial loans get a longer remaining life, which can raise lifetime ECL."}


@tool("portfolio", PORTFOLIO_AGENTS + ["credit_agent"])
def calculate_ead_change(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Exposure at default by portfolio for both dates, split into drawn balance and credit-converted undrawn commitments."""
    ctx = get_context()
    rows = {}
    for d, tag in ((previous_date, "previous"), (current_date, "current")):
        r = ctx.results(d, framework)
        g = r.groupby("portfolio")
        rows[tag] = pd.DataFrame({"ead": g["ead"].sum(), "drawn": g["current_balance"].sum(),
                                  "undrawn_total": g["undrawn_amount"].sum()})
        rows[tag]["ccf_undrawn_ead"] = rows[tag]["ead"] - rows[tag]["drawn"]
    m = rows["previous"].add_prefix("previous_").join(rows["current"].add_prefix("current_"), how="outer").fillna(0.0)
    out = []
    for port, x in m.iterrows():
        out.append({"portfolio": port, **x.to_dict(), "ead_change": x["current_ead"] - x["previous_ead"],
                    "ead_change_pct": pct_change(x["previous_ead"], x["current_ead"]),
                    "drawn_change": x["current_drawn"] - x["previous_drawn"],
                    "ccf_undrawn_change": x["current_ccf_undrawn_ead"] - x["previous_ccf_undrawn_ead"]})
    out.sort(key=lambda r: -abs(r["ead_change"]))
    tot0, tot1 = float(m["previous_ead"].sum()), float(m["current_ead"].sum())
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "previous_ead": tot0, "current_ead": tot1, "ead_change": tot1 - tot0, "ead_change_pct": pct_change(tot0, tot1),
            "by_portfolio": out,
            "notes": "Under CECL, unconditionally cancellable card lines carry no CCF (undrawn card limits are not reserved)."}


def _segment_matrix(previous_date: str, current_date: str, framework: str, segment_by: str) -> pd.DataFrame:
    ctx = get_context()
    attr = ctx.attribution(previous_date, current_date, framework)
    lc = attr.loan_contributions
    col_driver = driver_of_column(attr)
    cols = [c for c in col_driver if c in lc.columns and c != "overlay"]
    by_drv = pd.DataFrame({d: lc[[c for c in cols if col_driver[c] == d]].sum(axis=1)
                           for d in dict.fromkeys(col_driver.values())})
    mat = by_drv.groupby(lc[segment_by]).sum()
    ov = attr.portfolio_contributions["overlay"]
    if ov.abs().sum() > 0:
        drv = col_driver["overlay"]
        if segment_by == "portfolio":
            mat[drv] = mat[drv].add(ov.reindex(mat.index).fillna(0.0), fill_value=0.0)
        else:
            mat.loc["Portfolio-level overlays", :] = 0.0
            mat.loc["Portfolio-level overlays", drv] = float(ov.sum())
    mat["total"] = mat.sum(axis=1)
    return mat


@tool("portfolio", PORTFOLIO_AGENTS + ["credit_agent", "macro_agent", "reconciliation_agent"])
def calculate_segment_attribution(previous_date: str, current_date: str, framework: Framework = "CECL",
                                  segment_by: Literal["portfolio", "product_type"] = "portfolio") -> dict[str, Any]:
    """Allowance change by segment x driver (macro, credit, growth, runoff, model/judgment). Rows sum to each segment's change; columns sum to the driver totals."""
    ctx = get_context()
    mat = _segment_matrix(previous_date, current_date, framework, segment_by)
    attr = ctx.attribution(previous_date, current_date, framework)
    change = attr.summary["change"]
    labels = driver_labels()
    rows = []
    for seg, x in mat.sort_values("total", ascending=False).iterrows():
        drv = {d: float(x[d]) for d in labels if d in x}
        rows.append({"segment": seg, "total_change": float(x["total"]),
                     "share_of_total_change_pct": 100 * x["total"] / change if change else None,
                     "drivers": drv, "largest_driver": max(drv, key=lambda d: abs(drv[d]))})
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date,
            "segment_by": segment_by, "currency": "USD", "total_change": change,
            "driver_labels": labels, "segments": rows,
            "largest_contributing_segment": rows[0]["segment"] if rows else None,
            "column_totals": {d: float(mat[d].sum()) for d in labels if d in mat},
            "method": attr.summary["method"]}


@tool("portfolio", ["orchestrator", "portfolio_agent", "credit_agent", "macro_agent", "model_risk_agent",
                    "reconciliation_agent"])
def calculate_portfolio_attribution(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Main allowance waterfall: previous allowance -> drivers (runoff, growth, credit, macro, scenario weights, models, overlays) -> current allowance, with each step's dollar impact and share."""
    ctx = get_context()
    attr = ctx.attribution(previous_date, current_date, framework)
    s = attr.summary
    steps = []
    for st in attr.steps:
        steps.append({"step_id": st["id"], "label": st["label"], "driver": st["driver"], "impact": st["impact"],
                      "allowance_after": st["allowance_after"]})
    return {
        "framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
        "previous_allowance": s["previous_allowance"], "current_allowance": s["current_allowance"],
        "change": s["change"], "change_pct": s["change_pct"],
        "drivers": s["drivers"], "primary_driver": s["drivers"][0]["driver"] if s["drivers"] else None,
        "steps": steps, "explained_change": s["explained_change"], "residual": s["residual"],
        "loan_counts": s["loan_counts"], "method": s["method"],
        "notes": "Sequential counterfactual walk: each step re-runs the engine with one more input group switched to the current period. Run reconcile_allowance_change for controls and order sensitivity.",
    }
