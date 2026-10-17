"""The attribution pack as data for the offline demo app.

Same nine tool calls and the same eight sheets as `build_attribution_workbook` (src/tools/reporting_tools.py),
but returned as JSON so the browser can render it without Excel. Every cell carries its kind, matching the
workbook's colour code:

* ``input``   tool output (blue)        -> ``src`` names the tool call it came from
* ``formula`` derived figure (black)    -> ``x`` holds the formula, computed here in Python
* ``link``    copied from another sheet (green) -> ``x`` names the source cell

Derived figures are simple arithmetic on tool outputs (sums, differences, shares), the same formulas the
Excel pack writes. Nothing here models or estimates anything.
"""
from __future__ import annotations

from typing import Any

from src.tools.base import ToolError, call_tool

SHEETS = ["Summary", "Waterfall", "Portfolio x driver", "Macro variables", "Top borrowers", "Models and weights",
          "Reconciliation", "Sources"]
TOP_COLUMNS = ["rank", "counterparty_id", "customer_name", "portfolio", "balance_previous", "balance_current",
               "ecl_previous", "ecl_current", "ecl_change", "pd_12m_previous", "pd_12m_current", "main_driver",
               "reason_codes"]


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def cell(v: Any, k: str = "text", f: str | None = None, x: str | None = None, src: str | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"v": v, "k": k}
    if f:
        c["f"] = f
    if x:
        c["x"] = x
    if src:
        c["src"] = src
    return c


def text(v: Any) -> dict[str, Any]:
    return cell(v)


def build_pack(previous_date: str, current_date: str, framework: str = "CECL", top_n: int = 25) -> dict[str, Any]:
    pair = {"previous_date": previous_date, "current_date": current_date, "framework": framework}
    calls: list[dict[str, Any]] = []

    def run(name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        rec = call_tool(name, args, caller="demo_app", log=None)
        if rec.status != "ok":
            raise ToolError(f"{name} failed while building the pack: {rec.error['message']}", "dependency_error",
                            rec.error.get("hint"))
        calls.append({"call_id": rec.call_id, "tool_name": name, "parameters": rec.parameters,
                      "timestamp": rec.timestamp, "result_sha256": rec.result_sha256})
        return rec.result, rec.call_id

    cmp_, c_cmp = run("compare_allowance", pair)
    wf, c_wf = run("calculate_portfolio_attribution", pair)
    seg, c_seg = run("calculate_segment_attribution", pair)
    mv, c_mv = run("calculate_variable_attribution", pair)
    top, c_top = run("identify_top_contributors", {**pair, "n": top_n})
    models, c_mod = run("compare_model_versions", pair)
    weights, c_w = run("compare_scenario_weights", pair)
    overlays, c_ov = run("compare_management_overlays", pair)
    rec, c_rec = run("reconcile_allowance_change", pair)

    prev, cur = cmp_["previous_allowance"], cmp_["current_allowance"]
    change = cur - prev
    explained = sum(d["impact"] for d in wf["drivers"])
    primary = next(d["label"] for d in wf["drivers"] if d["driver"] == wf["primary_driver"])

    summary_rows = [
        [text("Previous allowance"), cell(prev, "input", "money", src=c_cmp)],
        [text("Current allowance"), cell(cur, "input", "money", src=c_cmp)],
        [text("Change"), cell(change, "formula", "money", "Current allowance − Previous allowance")],
        [text("Change %"), cell(_div(change, prev), "formula", "pct", "Change ÷ Previous allowance")],
        [text("Explained by attribution"), cell(explained, "link", "money", "Waterfall › Total explained")],
        [text("Residual (change − explained)"), cell(change - explained, "formula", "money", "Change − Explained by attribution")],
        [text("Previous balance"), cell(cmp_["previous_balance"], "input", "money", src=c_cmp)],
        [text("Current balance"), cell(cmp_["current_balance"], "input", "money", src=c_cmp)],
        [text("Reconciliation status (tool)"), cell(rec["overall_status"], "input", src=c_rec)],
        [text("Primary driver (tool)"), cell(primary, "input", src=c_wf)],
        [text("Any model version changed (tool)"), cell("Yes" if models["any_model_version_changed"] else "No", "input", src=c_mod)],
        [text("Scenario weights changed (tool)"), cell("Yes" if weights["weights_changed"] else "No", "input", src=c_w)],
    ]

    wf_rows = [[text(f"Allowance {previous_date}"), cell(prev, "link", "money", "Summary › Previous allowance"), text("")]]
    for d in wf["drivers"]:
        wf_rows.append([text(d["label"]), cell(d["impact"], "input", "money", src=c_wf),
                        cell(_div(d["impact"], change), "formula", "pct", "Impact ÷ Summary › Change")])
    wf_total = [text("Total explained"), cell(explained, "formula", "money", "Sum of driver impacts"),
                cell(_div(explained, change), "formula", "pct", "Total explained ÷ Summary › Change")]
    walk = prev + explained
    wf_tail = [
        [text(f"Allowance {current_date} (walk)"), cell(walk, "formula", "money", "Previous allowance + Total explained"), text("")],
        [text("Check: walk minus reported current"), cell(walk - cur, "formula", "money", "Walk − Summary › Current allowance"), text("")],
    ]

    keys = [d["driver"] for d in wf["drivers"]]
    labels = seg.get("driver_labels") or {d["driver"]: d["label"] for d in wf["drivers"]}
    segs = [x for x in seg["segments"] if "segment" in x]
    pxd_rows, col_tot = [], [0.0] * (len(keys) + 1)
    for x in segs:
        vals = [x["drivers"].get(k, 0.0) for k in keys]
        row_total = sum(vals)
        for j, v in enumerate(vals + [row_total]):
            col_tot[j] += v
        pxd_rows.append([text(x["segment"]), *[cell(v, "input", "money", src=c_seg) for v in vals],
                         cell(row_total, "formula", "money", "Sum across drivers")])
    pxd_total = [text("Total"), *[cell(v, "formula", "money", "Sum down portfolios") for v in col_tot]]

    vs = mv["variables"]
    macro_total = sum(v["impact"] for v in vs)
    seq_vals = [v.get("sequential_impact") for v in vs]
    mv_rows = [[text(v["variable"]), cell(v["impact"], "input", "money", src=c_mv),
                cell(v.get("sequential_impact"), "input", "money", src=c_mv),
                cell(_div(v["impact"], macro_total), "formula", "pct", "Shapley impact ÷ Total macro effect")] for v in vs]
    mv_total = [text("Total macro effect"), cell(macro_total, "formula", "money", "Sum of Shapley impacts"),
                cell(sum(s for s in seq_vals if s is not None), "formula", "money", "Sum of sequential impacts"), text("")]

    contributors = [c for c in top["contributors"] if "rank" in c]
    top_rows = []
    for c in contributors:
        row = []
        for col in TOP_COLUMNS:
            if col == "ecl_change":
                row.append(cell(c["ecl_current"] - c["ecl_previous"], "formula", "money", "ECL current − ECL previous"))
            elif col == "reason_codes":
                row.append(text("; ".join(c.get(col) or [])))
            elif col.startswith(("balance", "ecl")):
                row.append(cell(c.get(col), "input", "money", src=c_top))
            elif col.startswith("pd_"):
                row.append(cell(c.get(col), "input", "pct2", src=c_top))
            else:
                row.append(text(c.get(col)))
        top_rows.append(row)

    sc = weights["scenarios"]
    w_rows = [[text(x["scenario"]), cell(x["weight_previous"], "input", "pct", src=c_w),
               cell(x["weight_current"], "input", "pct", src=c_w),
               cell((x["weight_current"] - x["weight_previous"]) * 100, "formula", "pp", "(Current − Previous) × 100")]
              for x in sc]
    w_total = [text("Total"), cell(sum(x["weight_previous"] for x in sc), "formula", "pct", "Sum of weights"),
               cell(sum(x["weight_current"] for x in sc), "formula", "pct", "Sum of weights"), text("")]
    model_rows = [[text(x["product_type"]), *[cell(x.get(k), "input", src=c_mod) for k in (
        "pd_model_previous", "pd_model_current", "lgd_model_previous", "lgd_model_current",
        "ead_model_previous", "ead_model_current")]] for x in models["by_product"]]

    rec_rows = [[text(c.get("control")), cell(c.get("status"), "input", src=c_rec),
                 cell(c.get("value") if isinstance(c.get("value"), (int, float)) else str(c.get("value")), "input", src=c_rec)]
                for c in rec.get("controls", [])]

    src_rows = [[text(c["call_id"]), text(c["tool_name"]), text(str(c["parameters"])), text(c["timestamp"]),
                 text(c["result_sha256"])] for c in calls]

    sheets = [
        {"name": "Summary", "title": f"{framework} allowance attribution: {previous_date} to {current_date}",
         "tables": [{"columns": ["Item", "Value"], "rows": summary_rows}]},
        {"name": "Waterfall", "title": "Attribution waterfall (sequential counterfactual; residual measured, never forced)",
         "chart": {"start": prev, "end": cur, "steps": [{"label": d["label"], "impact": d["impact"]} for d in wf["drivers"]]},
         "tables": [{"columns": ["Driver", "Impact ($)", "Share of change"], "rows": wf_rows,
                     "totals": [wf_total], "tail": wf_tail}]},
        {"name": "Portfolio x driver", "title": "Allowance change by portfolio and driver",
         "tables": [{"columns": ["Portfolio", *[labels.get(k, k) for k in keys], "Total"], "rows": pxd_rows,
                     "totals": [pxd_total]}]},
        {"name": "Macro variables", "title": "Macro effect by variable (Shapley, with sequential for comparison)",
         "tables": [{"columns": ["Variable", "Shapley impact ($)", "Sequential impact ($)", "Share of macro"],
                     "rows": mv_rows, "totals": [mv_total]}]},
        {"name": "Top borrowers", "title": f"Top {len(contributors)} borrowers by allowance change",
         "tables": [{"columns": [c.replace("_", " ") for c in TOP_COLUMNS], "rows": top_rows}]},
        {"name": "Models and weights", "title": "Scenario weights, overlays and model versions",
         "tables": [
             {"caption": "Scenario weights", "columns": ["Scenario", f"Weight {previous_date}", f"Weight {current_date}", "Change (pp)"],
              "rows": w_rows, "totals": [w_total]},
             {"caption": "Weight change and overlays", "columns": ["Item", "Previous", "Current"], "rows": [
                 [text("Allowance impact of the weight change (tool)"),
                  cell(weights["allowance_impact_of_weight_change"], "input", "money", src=c_w), text("")],
                 [text("Management overlays (tool)"), cell(overlays["overlay_total_previous"], "input", "money", src=c_ov),
                  cell(overlays["overlay_total_current"], "input", "money", src=c_ov)]]},
             {"caption": "Model versions by product",
              "columns": ["Product", "PD previous", "PD current", "LGD previous", "LGD current", "EAD previous", "EAD current"],
              "rows": model_rows}]},
        {"name": "Reconciliation", "title": f"Reconciliation controls: {rec['overall_status']}",
         "tables": [{"columns": ["Control", "Status", "Value"], "rows": rec_rows}]},
        {"name": "Sources", "title": "Every tool call behind the numbers",
         "tables": [{"columns": ["call_id", "tool", "parameters", "timestamp (UTC)", "result sha256"], "rows": src_rows}]},
    ]
    assert [s["name"] for s in sheets] == SHEETS
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "sheets": sheets,
            "headline": {"previous_allowance": prev, "current_allowance": cur, "change": change,
                         "explained_change": explained, "residual": change - explained, "primary_driver": wf["primary_driver"],
                         "reconciliation": rec["overall_status"],
                         "top_borrower": contributors[0]["counterparty_id"] if contributors else None},
            "source_calls": calls}
