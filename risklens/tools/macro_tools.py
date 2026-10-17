"""Macro Scenario Agent tools."""
from __future__ import annotations

import numpy as np

from risklens import config
from risklens.cecl import attribution as A
from risklens.cecl.engine import SCENARIOS, RunInputs, run_cecl
from risklens.tools.common import check_order, portfolios, store
from risklens.tools.registry import ToolError, tool


@tool("macro")
def compare_macro_scenarios(previous_date: str, current_date: str, scenario: str = None) -> dict:
    """Macro forecast in both vintages: average over the 8-quarter reasonable & supportable period, per variable and scenario."""
    check_order(previous_date, current_date)
    m = config.methodology()["macro"]
    rs = m["reasonable_supportable_quarters"]
    df = store().macro
    out = {}
    worse = []
    for s in ([scenario] if scenario else list(SCENARIOS)):
        out[s] = {}
        for v in m["variables"]:
            a = df[(df["as_of_date"] == previous_date) & (df["scenario"] == s) & (df["forecast_quarter"] <= rs)][v].mean()
            b = df[(df["as_of_date"] == current_date) & (df["scenario"] == s) & (df["forecast_quarter"] <= rs)][v].mean()
            out[s][v] = {"previous": a, "current": b, "change": b - a}
            risk_dir = np.sign(m["pd_beta"]["Credit Card"][v] or m["pd_beta"]["CRE"][v] or m["pd_beta"]["Mortgage"][v] or 0)
            if (b - a) * risk_dir > 1e-9:
                worse.append(f"{s}.{v}")
    return {"horizon": f"average of forecast quarters 1-{rs} (reasonable & supportable)", "units": "percent / percent growth",
            "by_scenario": out, "deteriorated": worse}


@tool("macro")
def compare_scenario_weights(previous_date: str, current_date: str) -> dict:
    """Scenario weights used in each period and whether they changed."""
    check_order(previous_date, current_date)
    a, b = store().weights(previous_date), store().weights(current_date)
    return {"previous": a, "current": b, "change": {s: b[s] - a[s] for s in a},
            "changed": any(abs(b[s] - a[s]) > 1e-12 for s in a),
            "shift_toward": max(a, key=lambda s: b[s] - a[s]) if any(abs(b[s] - a[s]) > 1e-12 for s in a) else None}


@tool("macro")
def calculate_macro_attribution(previous_date: str, current_date: str) -> dict:
    """Allowance impact of the new macro scenario paths (sequential waterfall), total and by portfolio, plus the scenario-weight effect."""
    check_order(previous_date, current_date)
    w = A.waterfall(previous_date, current_date)
    macro = next(r for r in w["steps"] if r["step"] == "macro")
    meth = next(r for r in w["steps"] if r["step"] == "methodology")
    total = w["end_allowance"] - w["start_allowance"]
    return {"macro_paths_impact": macro["impact"], "share_of_total_change_pct": macro["impact"] / total * 100 if total else None,
            "by_portfolio": macro["impact_by_portfolio"],
            "largest_portfolio": max(macro["impact_by_portfolio"], key=macro["impact_by_portfolio"].get),
            "scenario_weights_and_methodology_impact": meth["impact"],
            "note": "macro paths measured after runoff, growth and credit steps; scenario-weight changes are in the methodology step"}


@tool("macro")
def calculate_variable_attribution(previous_date: str, current_date: str, portfolio: str = None) -> dict:
    """Split the macro impact by variable (unemployment, GDP, rates, HPI, CRE prices, inflation) using exact Shapley values."""
    check_order(previous_date, current_date)
    res = A.macro_variable_shapley(previous_date, current_date)
    if portfolio:
        vals = {v: res["by_variable_and_portfolio"][v].get(portfolio, 0.0) for v in res["by_variable"]}
    else:
        vals = res["by_variable"]
    ranked = sorted(vals, key=vals.get, reverse=True)
    total = sum(vals.values())
    return {"portfolio": portfolio or "ALL", "by_variable": vals, "ranking": ranked, "top_variable": ranked[0],
            "top_variable_share_pct": vals[ranked[0]] / total * 100 if total else None, "macro_total": total,
            "method": "exact Shapley over the 6 variables; sums to the macro step of the waterfall"}


@tool("macro")
def run_scenario_analysis(as_of_date: str, baseline_weight: float = None, adverse_weight: float = None,
                          severe_adverse_weight: float = None) -> dict:
    """Allowance under each scenario at 100% weight, under the booked weights, and under optional what-if weights."""
    s = store()
    inp = s.inputs(as_of_date)
    res = run_cecl(inp)
    single = {sc: float(res.ecl_by_scenario[i].sum() + res.overlay_total) for i, sc in enumerate(SCENARIOS)}
    out = {"as_of_date": as_of_date, "booked_weights": inp.weights, "booked_allowance": res.allowance,
           "allowance_if_100pct": single}
    given = {"baseline": baseline_weight, "adverse": adverse_weight, "severe_adverse": severe_adverse_weight}
    if any(v is not None for v in given.values()):
        w = {k: (v if v is not None else inp.weights[k]) for k, v in given.items()}
        if any(x > 1 for x in w.values()):
            w = {k: x / 100 for k, x in w.items()}
        if abs(sum(w.values()) - 1) > 1e-6:
            raise ToolError(f"What-if weights must sum to 1 (got {sum(w.values()):.4f}).", {"booked_weights": inp.weights})
        # Overlays are added once, not per scenario.
        what_if = sum(w[k] * (single[k] - res.overlay_total) for k in SCENARIOS) + res.overlay_total
        out["what_if"] = {"weights": w, "allowance": what_if, "change_vs_booked": what_if - res.allowance,
                          "note": "sandbox calculation; not a booked figure"}
    return out


@tool("macro")
def calculate_segment_macro_sensitivity(as_of_date: str, variable: str = "unemployment_rate", shock: float = 1.0) -> dict:
    """Allowance change by portfolio if one macro variable is shocked by `shock` units in every scenario and quarter.

    variable: unemployment_rate, gdp_growth, interest_rate, hpi_growth, cre_price_growth or inflation
    shock: size of the shock in the variable's units (default +1.0)
    """
    names = config.methodology()["macro"]["variables"]
    if variable not in names:
        raise ToolError(f"Unknown macro variable '{variable}'.", {"valid_variables": names})
    s = store()
    inp = s.inputs(as_of_date)
    base = run_cecl(inp)
    dev = inp.macro_dev.copy()
    dev[:, names.index(variable), :] += shock
    shocked = run_cecl(RunInputs(as_of_date, inp.loans, dev, inp.weights, inp.overlays))
    a, b = base.by_portfolio(), shocked.by_portfolio()
    by = {p: {"impact": b[p] - a[p], "impact_pct_of_portfolio_allowance": (b[p] - a[p]) / a[p] * 100 if a[p] else None} for p in portfolios()}
    return {"as_of_date": as_of_date, "variable": variable, "shock": shock, "total_impact": shocked.allowance - base.allowance,
            "by_portfolio": by, "most_sensitive_portfolio": max(by, key=lambda p: by[p]["impact"]),
            "most_sensitive_portfolio_relative": max(by, key=lambda p: by[p]["impact_pct_of_portfolio_allowance"] or 0)}
