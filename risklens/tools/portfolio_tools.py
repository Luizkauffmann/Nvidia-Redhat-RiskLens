"""Portfolio Agent tools: exposure, runoff, new business, segment attribution."""
from __future__ import annotations

from risklens.cecl import attribution as A
from risklens.tools.common import check_order, driver_labels, loans, portfolios, store
from risklens.tools.registry import tool


def _split(prev: str, cur: str, portfolio: str | None):
    P = loans(prev, portfolio).set_index("loan_id")
    C = loans(cur, portfolio).set_index("loan_id")
    surv = P.index.intersection(C.index)
    return P, C, surv, P.index.difference(C.index), C.index.difference(P.index)


@tool("portfolio")
def calculate_balance_change(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """Exact balance bridge: previous balance - exits +/- changes on existing loans + new originations = current balance."""
    check_order(previous_date, current_date)
    P, C, surv, exited, new = _split(previous_date, current_date, portfolio)
    pb, cb = float(P["current_balance"].sum()), float(C["current_balance"].sum())
    exits = float(P.loc[exited, "current_balance"].sum())
    existing = float(C.loc[surv, "current_balance"].sum() - P.loc[surv, "current_balance"].sum())
    newb = float(C.loc[new, "current_balance"].sum())
    return {"portfolio": portfolio or "ALL", "previous_balance": pb, "exits_paid_off_matured_charged_off": -exits,
            "change_on_existing_loans": existing, "new_originations": newb, "current_balance": cb,
            "net_change": cb - pb, "bridge_check": round(pb - exits + existing + newb - cb, 2),
            "loan_counts": {"previous": len(P), "exited": len(exited), "new": len(new), "current": len(C)}}


@tool("portfolio")
def calculate_new_originations(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """New loans booked between the two dates: count, balance and allowance, by portfolio, plus the largest ones."""
    check_order(previous_date, current_date)
    _, C, _, _, new = _split(previous_date, current_date, portfolio)
    N = C.loc[new]
    by = N.groupby("portfolio").agg(loans=("product_type", "size"), balance=("current_balance", "sum"), ecl=("ecl", "sum"))
    top = N.nlargest(5, "current_balance")[["counterparty_id", "portfolio", "product_type", "property_type", "current_balance", "ecl"]]
    w = A.waterfall(previous_date, current_date)
    growth = next(r for r in w["steps"] if r["step"] == "growth")
    return {"portfolio": portfolio or "ALL", "new_loans": int(len(N)), "new_balance": float(N["current_balance"].sum()),
            "new_loan_ecl": float(N["ecl"].sum()), "by_portfolio": by.to_dict("index"),
            "largest_new_loans": top.reset_index().to_dict("records"),
            "allowance_impact_portfolio_growth": growth["impact_by_portfolio"].get(portfolio, 0.0) if portfolio else growth["impact"],
            "impact_note": "allowance impact is the sequential-waterfall 'growth' step (new loans at previous macro and weights)"}


@tool("portfolio")
def calculate_paydowns(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """Loans that paid off, matured or were charged off, amortisation on existing loans, and the allowance released."""
    check_order(previous_date, current_date)
    P, C, surv, exited, _ = _split(previous_date, current_date, portfolio)
    X = P.loc[exited]
    by = X.groupby("portfolio").agg(loans=("product_type", "size"), balance=("current_balance", "sum"), ecl_released=("ecl", "sum"))
    w = A.waterfall(previous_date, current_date)
    run = next(r for r in w["steps"] if r["step"] == "runoff")
    amort = float(C.loc[surv, "current_balance"].sum() - P.loc[surv, "current_balance"].sum())
    return {"portfolio": portfolio or "ALL", "exited_loans": int(len(X)), "exited_balance": float(X["current_balance"].sum()),
            "allowance_on_exited_loans_previous": float(X["ecl"].sum()), "by_portfolio": by.to_dict("index"),
            "balance_change_on_existing_loans": amort,
            "allowance_impact_paydowns_runoff": run["impact_by_portfolio"].get(portfolio, 0.0) if portfolio else run["impact"]}


@tool("portfolio")
def calculate_ead_change(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """Exposure at default in both periods, total and by portfolio."""
    check_order(previous_date, current_date)
    P, C = loans(previous_date, portfolio), loans(current_date, portfolio)
    a, b = P.groupby("portfolio")["ead"].sum(), C.groupby("portfolio")["ead"].sum()
    by = {p: {"previous": float(a.get(p, 0)), "current": float(b.get(p, 0)), "change": float(b.get(p, 0) - a.get(p, 0))}
          for p in (portfolios() if portfolio is None else [portfolio])}
    return {"portfolio": portfolio or "ALL", "previous_ead": float(P["ead"].sum()), "current_ead": float(C["ead"].sum()),
            "change": float(C["ead"].sum() - P["ead"].sum()), "by_portfolio": by}


@tool("portfolio")
def calculate_segment_attribution(previous_date: str, current_date: str) -> dict:
    """Allowance change by portfolio, split into drivers (sequential waterfall). Shows which segment drove the change."""
    check_order(previous_date, current_date)
    w = A.waterfall(previous_date, current_date)
    labels = driver_labels()
    table = {}
    for p in portfolios():
        row = {}
        for r in w["steps"]:
            row[labels[r["driver"]]] = row.get(labels[r["driver"]], 0.0) + r["impact_by_portfolio"].get(p, 0.0)
        row["total_change"] = sum(row.values())
        table[p] = row
    ranked = sorted(table, key=lambda p: table[p]["total_change"], reverse=True)
    total = sum(t["total_change"] for t in table.values())
    return {"previous_date": previous_date, "current_date": current_date, "by_portfolio": table,
            "ranking": ranked, "largest_contributor": ranked[0],
            "largest_contributor_share_pct": table[ranked[0]]["total_change"] / total * 100 if total else None,
            "total_change": total}


@tool("portfolio")
def calculate_portfolio_attribution(previous_date: str, current_date: str, portfolio: str) -> dict:
    """Driver split of the allowance change for one portfolio (sequential waterfall)."""
    check_order(previous_date, current_date)
    w = A.waterfall(previous_date, current_date)
    labels = driver_labels()
    steps = [{"step": r["step"], "driver": labels[r["driver"]], "impact": r["impact_by_portfolio"].get(portfolio, 0.0)} for r in w["steps"]]
    return {"portfolio": portfolio, "steps": steps, "total_change": sum(s["impact"] for s in steps),
            "primary_driver": max(steps, key=lambda s: s["impact"])["driver"]}
