"""Macro Scenario Agent tools: forecast paths, scenario weights, macro attribution.

Variable-level attribution has two methods:
* ``sequential`` — part of the main waterfall (one variable at a time, in config order)
* ``shapley``    — exact Shapley over the 6 variables (64 projections); order independent
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from src.cecl.engine import EngineInputs
from src.tools.base import Framework, ToolError, get_context, pct_change, tool
from src.utils import config

MACRO_AGENTS = ["orchestrator", "macro_agent"]

# Direction in which each variable hurts credit (used for labels only, never for numbers).
ADVERSE_DIRECTION = {"unemployment_rate": "up", "gdp_growth": "down", "interest_rate": "up",
                     "hpi_growth": "down", "cre_price_growth": "down", "inflation": "up"}
UNITS = {"unemployment_rate": "% of labour force", "gdp_growth": "% annualised real growth",
         "interest_rate": "% policy rate", "hpi_growth": "% YoY house prices", "cre_price_growth": "% YoY CRE prices",
         "inflation": "% YoY CPI"}


def _rs_quarters() -> int:
    return int(config.methodology()["macro"]["reasonable_supportable_quarters"])


@tool("macro", MACRO_AGENTS + ["model_risk_agent"])
def compare_macro_scenarios(previous_date: str, current_date: str, scenario: str | None = None,
                            include_paths: bool = False) -> dict[str, Any]:
    """Compare the macro forecast vintages: for each scenario and variable, the average over the reasonable & supportable period, the worst quarter and the change, flagged worse/better for credit. Set include_paths for quarterly paths (charts)."""
    ctx = get_context()
    m0, m1 = ctx.store.macro_at(previous_date), ctx.store.macro_at(current_date)
    rs = _rs_quarters()
    anchors = config.methodology()["macro"]["long_run_anchor"]
    scen = [scenario] if scenario else ctx.scenarios
    out, worsened = [], {}
    for s in scen:
        a = m0[m0["scenario"] == s].sort_values("forecast_quarter").head(rs)
        b = m1[m1["scenario"] == s].sort_values("forecast_quarter").head(rs)
        vars_ = []
        for v in ctx.macro_variables:
            adv = ADVERSE_DIRECTION[v]
            worst = (lambda x: x.max()) if adv == "up" else (lambda x: x.min())
            avg0, avg1 = float(a[v].mean()), float(b[v].mean())
            chg = avg1 - avg0
            direction = "worse" if (chg > 0) == (adv == "up") and abs(chg) > 1e-9 else ("better" if abs(chg) > 1e-9 else "unchanged")
            vars_.append({"variable": v, "units": UNITS[v], "long_run_anchor": anchors[v],
                          "rs_average_previous": avg0, "rs_average_current": avg1, "change": chg,
                          "worst_quarter_previous": float(worst(a[v])), "worst_quarter_current": float(worst(b[v])),
                          "credit_direction": direction})
            if direction == "worse":
                worsened.setdefault(s, []).append(v)
        out.append({"scenario": s, "variables": vars_})
    path_scen = scenario or "baseline"
    paths = {}
    if include_paths:
        for tag, m in (("previous", m0), ("current", m1)):
            x = m[m["scenario"] == path_scen].sort_values("forecast_quarter").head(rs)
            paths[tag] = x[["forecast_quarter", "quarter_label"] + ctx.macro_variables].to_dict("records")
    units = {v: UNITS[v] for v in ctx.macro_variables}
    for sc in out:
        for v in sc["variables"]:
            v.pop("units")
    return {"units": units, "previous_date": previous_date, "current_date": current_date,
            "scenario_set_previous": m0["scenario_set_id"].iloc[0], "scenario_set_current": m1["scenario_set_id"].iloc[0],
            "reasonable_supportable_quarters": rs,
            "reversion": f"straight-line to long-run anchor over {config.methodology()['macro']['reversion_quarters']} quarters",
            "scenarios": out, "variables_worse_for_credit": worsened,
            **({"paths": {"scenario": path_scen, **paths}} if include_paths else {})}


@tool("macro", MACRO_AGENTS + ["model_risk_agent"])
def compare_scenario_weights(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Scenario probability weights on both dates (with approver), unweighted scenario ECL, and the allowance impact of the weight change from the attribution walk."""
    ctx = get_context()
    w0, w1 = ctx.store.weights_at(previous_date), ctx.store.weights_at(current_date)
    meta0 = ctx.store.weights[pd.to_datetime(ctx.store.weights["as_of_date"]) == pd.Timestamp(previous_date)]
    meta1 = ctx.store.weights[pd.to_datetime(ctx.store.weights["as_of_date"]) == pd.Timestamp(current_date)]
    s0, s1 = ctx.summary_row(previous_date, framework), ctx.summary_row(current_date, framework)
    attr = ctx.attribution(previous_date, current_date, framework)
    impact = sum(st["impact"] for st in attr.steps if st["step"] == "scenario_weights")
    rows = []
    for s in ctx.scenarios:
        rows.append({"scenario": s, "weight_previous": w0.get(s), "weight_current": w1.get(s),
                     "weight_change_pp": 100 * (w1.get(s, 0) - w0.get(s, 0)),
                     "unweighted_ecl_previous": s0.get(f"scenario_ecl_{s}"),
                     "unweighted_ecl_current": s1.get(f"scenario_ecl_{s}")})
    changed = any(abs(r["weight_change_pp"]) > 1e-9 for r in rows)
    shift = max(rows, key=lambda r: r["weight_change_pp"])["scenario"] if changed else None
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "weights_changed": changed, "weight_shifted_toward": shift, "scenarios": rows,
            "approval_previous": {"approved_by": meta0["approved_by"].iloc[0], "approval_date": meta0["approval_date"].iloc[0]},
            "approval_current": {"approved_by": meta1["approved_by"].iloc[0], "approval_date": meta1["approval_date"].iloc[0]},
            "allowance_impact_of_weight_change": float(impact),
            "notes": "Impact measured in the attribution walk after macro paths moved to the current vintage (sequential method)."}


@tool("macro", MACRO_AGENTS + ["reconciliation_agent"])
def calculate_macro_attribution(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Allowance impact of the macro forecast change: total, share of the allowance change, split by portfolio and by variable (sequential walk)."""
    ctx = get_context()
    attr = ctx.attribution(previous_date, current_date, framework)
    lc = attr.loan_contributions
    cols = [c for c in lc.columns if c.startswith("macro:")]
    total = float(sum(st["impact"] for st in attr.steps if st["step"] == "macro"))
    change = attr.summary["change"]
    by_port = lc[cols].sum(axis=1).groupby(lc["portfolio"]).sum().sort_values(ascending=False)
    by_var = sorted(({"variable": st["id"].split(":", 1)[1], "impact": st["impact"]}
                     for st in attr.steps if st["step"] == "macro"), key=lambda x: -abs(x["impact"]))
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "macro_impact": total, "total_change": change,
            "macro_share_of_change_pct": 100 * total / change if change else None,
            "by_portfolio": [{"portfolio": p, "impact": float(v), "share_of_macro_pct": 100 * v / total if total else None}
                             for p, v in by_port.items()],
            "by_variable_sequential": by_var,
            "top_variable": by_var[0]["variable"] if by_var else None,
            "most_affected_portfolio": by_port.index[0] if len(by_port) else None,
            "method": {"type": "sequential", "variable_order": attr.summary["method"]["macro_variable_order"],
                       "note": "Use calculate_variable_attribution(method='shapley') for an order-independent split."}}


@tool("macro", MACRO_AGENTS + ["reconciliation_agent"], slow=True)
def calculate_variable_attribution(previous_date: str, current_date: str, framework: Framework = "CECL",
                                   method: Literal["shapley", "sequential"] = "shapley") -> dict[str, Any]:
    """Which macro variable drove the allowance change: impact per variable (exact Shapley by default, or sequential) and the variable x portfolio matrix."""
    ctx = get_context()
    attr = ctx.attribution(previous_date, current_date, framework)
    seq = {st["id"].split(":", 1)[1]: st["impact"] for st in attr.steps if st["step"] == "macro"}
    if method == "sequential":
        lc = attr.loan_contributions
        cols = [c for c in lc.columns if c.startswith("macro:")]
        mat = lc[cols].groupby(lc["portfolio"]).sum().rename(columns=lambda c: c.split(":", 1)[1])
        variables = sorted(({"variable": v, "impact": i} for v, i in seq.items()), key=lambda x: -abs(x["impact"]))
        total = float(sum(seq.values()))
        runs = None
    else:
        sh = ctx.macro_shapley(previous_date, current_date, framework)
        mat = sh["by_portfolio"]
        variables = sh["variables"]
        total = sh["macro_total"]
        runs = sh["engine_runs"]
    for v in variables:
        v["share_of_macro_pct"] = 100 * v["impact"] / total if total else None
        v["sequential_impact"] = seq.get(v["variable"])
        v["adverse_direction"] = ADVERSE_DIRECTION[v["variable"]]
    seq_rank = sorted(seq, key=lambda k: -abs(seq[k]))
    matrix = []
    for port, row in mat.iterrows():
        matrix.append({"portfolio": port, **{k: float(x) for k, x in row.items()},
                       "top_variable": max(row.index, key=lambda k: abs(row[k]))})
    matrix.sort(key=lambda r: -sum(abs(v) for k, v in r.items() if k in seq))
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "method": method, "engine_runs": runs, "macro_total": total, "variables": variables,
            "top_variable": variables[0]["variable"] if variables else None,
            "top_variable_agrees_with_sequential": bool(variables) and variables[0]["variable"] == seq_rank[0],
            "by_portfolio": matrix,
            "notes": "Shapley splits interaction effects evenly across variables; values sum exactly to the macro total."}


@tool("macro", MACRO_AGENTS + ["model_risk_agent"])
def run_scenario_analysis(as_of_date: str, framework: Framework = "CECL",
                          weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Allowance if each scenario were 100% weighted, and optionally under custom weights (e.g. {"baseline":0.5,"adverse":0.3,"severe_adverse":0.2}), versus the official result; also by portfolio."""
    ctx = get_context()
    r = ctx.results(as_of_date, framework)
    ov = float(ctx.store.overlays_at(as_of_date)["amount"].sum())
    official_w = ctx.store.weights_at(as_of_date)
    row = ctx.summary_row(as_of_date, framework)
    scen = {s: float(r[f"ecl_{s}"].sum()) for s in ctx.scenarios}
    out = {"framework": framework, "as_of_date": as_of_date, "currency": "USD",
           "official_allowance": row["allowance"], "official_weights": official_w, "overlay_total": ov,
           "single_scenario_allowance": {s: v + ov for s, v in scen.items()},
           "by_portfolio_unweighted_ecl": r.groupby("portfolio")[[f"ecl_{s}" for s in ctx.scenarios]].sum()
           .rename(columns=lambda c: c[4:]).reset_index().to_dict("records")}
    if weights:
        unknown = set(weights) - set(ctx.scenarios)
        if unknown:
            raise ToolError(f"Unknown scenario(s) {sorted(unknown)}", hint={"scenarios": ctx.scenarios})
        w = {s: float(weights.get(s, 0.0)) for s in ctx.scenarios}
        if any(v < 0 for v in w.values()) or abs(sum(w.values()) - 1.0) > 1e-6:
            raise ToolError(f"Weights must be non-negative and sum to 1 (got {sum(w.values()):.4f})",
                            hint={"example": official_w})
        alt = sum(w[s] * scen[s] for s in scen) + ov
        out["custom"] = {"weights": w, "allowance": alt, "difference_vs_official": alt - row["allowance"],
                         "difference_pct": pct_change(row["allowance"], alt)}
    out["notes"] = ("Weighted ECL is linear in the weights, so these figures are exact re-weightings of the official "
                    "scenario results (no re-estimation). Results are what-ifs, not the official allowance.")
    return out


@tool("macro", MACRO_AGENTS, slow=True)
def calculate_segment_macro_sensitivity(as_of_date: str, variable: str = "unemployment_rate", shock: float = 1.0,
                                        framework: Framework = "CECL", scenario: str | None = None) -> dict[str, Any]:
    """Which portfolio is most sensitive to a macro variable: allowance change by portfolio if the variable's forecast is shifted by `shock` units over the reasonable & supportable period (then reverts)."""
    ctx = get_context()
    if not np.isfinite(shock) or abs(shock) > 20:
        raise ToolError("shock must be a finite number of the variable's units, |shock| <= 20",
                        hint={"example": {"variable": "unemployment_rate", "shock": 1.0}})
    inp: EngineInputs = ctx.store.engine_inputs(as_of_date)
    base = inp.macro
    shocked = base.copy()
    rows_sel = shocked["scenario"].eq(scenario) if scenario else np.ones(len(shocked), dtype=bool)
    shocked.loc[rows_sel, variable] = shocked.loc[rows_sel, variable] + shock
    res = ctx.engine.macro_sweep(inp, [base, shocked], framework)
    port = inp.loans["portfolio"].to_numpy()
    d = pd.Series(res[1]["loan_ecl"] - res[0]["loan_ecl"]).groupby(port).sum()
    b = pd.Series(res[0]["loan_ecl"]).groupby(port).sum()
    rows = [{"portfolio": p, "base_ecl": float(b[p]), "ecl_change": float(d[p]),
             "ecl_change_pct": pct_change(float(b[p]), float(b[p] + d[p]))} for p in d.index]
    rows.sort(key=lambda r: -abs(r["ecl_change"]))
    total = float(d.sum())
    return {"framework": framework, "as_of_date": as_of_date, "variable": variable, "units": UNITS[variable],
            "shock": shock, "scenarios_shocked": [scenario] if scenario else ctx.scenarios, "currency": "USD",
            "allowance_base": res[0]["allowance"], "allowance_shocked": res[1]["allowance"], "allowance_change": total,
            "by_portfolio": rows,
            "most_sensitive_portfolio_dollars": rows[0]["portfolio"] if rows else None,
            "most_sensitive_portfolio_pct": max(rows, key=lambda r: abs(r["ecl_change_pct"] or 0))["portfolio"] if rows else None,
            "notes": "What-if sensitivity on the official inputs; not part of the reported allowance."}
