"""Allowance-level tools shared by every agent."""
from __future__ import annotations

from risklens import config
from risklens.tools.common import check_order, loans, pct, portfolios, store, wavg
from risklens.tools.registry import tool


@tool("shared")
def get_reporting_context() -> dict:
    """List available reporting dates, portfolios, scenarios and macro variables. Call this first if unsure."""
    s = store()
    return {
        "framework": config.settings()["framework"],
        "available_dates": s.dates,
        "default_comparison": {"previous_date": s.dates[0], "current_date": s.dates[-1]},
        "portfolios": portfolios(),
        "scenarios": ["baseline", "adverse", "severe_adverse"],
        "macro_variables": config.methodology()["macro"]["variables"],
        "attribution_drivers": config.methodology()["attribution"]["drivers"],
        "currency": "USD",
    }


def _summary(as_of: str, portfolio: str | None = None) -> dict:
    df = loans(as_of, portfolio)
    run = store().run(as_of)
    by_pf = run.by_portfolio()
    overlay = run.overlay_by_portfolio.get(portfolio, 0.0) if portfolio else run.overlay_total
    modeled = float(df["ecl"].sum())
    allowance = modeled + overlay if portfolio else store().reported_allowance(as_of)
    bal = float(df["current_balance"].sum())
    out = {
        "as_of_date": as_of, "portfolio": portfolio or "ALL", "allowance": allowance,
        "modeled_ecl": modeled, "management_overlay": overlay, "loan_count": int(len(df)),
        "balance": bal, "ead": float(df["ead"].sum()),
        "coverage_pct": allowance / bal * 100 if bal else None,
        "avg_pd_12m_pct": (wavg(df["pd_12m"], df["ead"]) or 0) * 100,
        "avg_lgd_pct": (wavg(df["lgd"], df["ead"]) or 0) * 100,
        "averages_are": "EAD-weighted",
    }
    if portfolio is None:
        out["run_id"] = store().manifest[as_of]["run_id"]
        out["by_portfolio"] = {p: {"allowance": by_pf[p], "balance": float(df.loc[df["portfolio"] == p, "current_balance"].sum())}
                               for p in portfolios()}
    return out


@tool("shared")
def get_allowance_summary(as_of_date: str, portfolio: str = None) -> dict:
    """Allowance, balance, coverage, average PD and LGD for one reporting date (optionally one portfolio)."""
    return _summary(as_of_date, portfolio)


@tool("shared")
def compare_allowance(previous_date: str, current_date: str) -> dict:
    """Booked allowance in two periods: change in dollars and percent, total and by portfolio."""
    check_order(previous_date, current_date)
    a, b = _summary(previous_date), _summary(current_date)
    by = {p: {"previous": a["by_portfolio"][p]["allowance"], "current": b["by_portfolio"][p]["allowance"],
              "change": b["by_portfolio"][p]["allowance"] - a["by_portfolio"][p]["allowance"]} for p in portfolios()}
    largest = max(by, key=lambda p: by[p]["change"])
    return {
        "previous_date": previous_date, "current_date": current_date,
        "previous_allowance": a["allowance"], "current_allowance": b["allowance"],
        "change": b["allowance"] - a["allowance"], "change_pct": pct(a["allowance"], b["allowance"]),
        "previous_balance": a["balance"], "current_balance": b["balance"], "balance_change": b["balance"] - a["balance"],
        "previous_coverage_pct": a["coverage_pct"], "current_coverage_pct": b["coverage_pct"],
        "by_portfolio": by, "largest_increase_portfolio": largest,
        "source": "booked official runs (run_manifest.json)",
    }


@tool("shared")
def get_portfolio_snapshot(as_of_date: str, portfolio: str = None) -> dict:
    """Composition of the book on one date: products, delinquency mix and rating mix with balances and ECL."""
    df = loans(as_of_date, portfolio)
    prod = df.groupby("product_type").agg(loans=("loan_id", "size"), balance=("current_balance", "sum"), ecl=("ecl", "sum"))
    dq = df.groupby("delinquency_bucket").agg(loans=("loan_id", "size"), balance=("current_balance", "sum"))
    out = {"as_of_date": as_of_date, "portfolio": portfolio or "ALL", "loan_count": int(len(df)),
           "balance": float(df["current_balance"].sum()), "ecl": float(df["ecl"].sum()),
           "by_product": prod.to_dict("index"), "delinquency": dq.to_dict("index"),
           "default_count": int(df["default_flag"].sum())}
    com = df[df["internal_rating"].notna()]
    if len(com):
        out["rating_mix_balance"] = com.groupby("internal_rating")["current_balance"].sum().to_dict()
        out["watchlist"] = {"loans": int(com["watchlist_flag"].sum()),
                            "balance": float(com.loc[com["watchlist_flag"], "current_balance"].sum())}
    return out
