"""Reconciliation Agent tools: driver attribution and the reconciliation control."""
from __future__ import annotations

from risklens import config
from risklens.cecl import attribution as A
from risklens.tools.common import check_order, driver_labels, store
from risklens.tools.registry import tool


def _drivers(prev: str, cur: str, method: str) -> dict[str, float]:
    if method == "shapley":
        g = A.shapley_groups(prev, cur)
        steps = config.methodology()["attribution"]["steps"]
        out: dict[str, float] = {}
        for s in steps:
            out[s["driver"]] = out.get(s["driver"], 0.0) + g[s["id"]]
        return out
    order = None
    if method == "reverse":
        order = tuple(reversed([s["id"] for s in config.methodology()["attribution"]["steps"]]))
    return A.driver_totals(A.waterfall(prev, cur, order=order)["steps"])


@tool("reconciliation", enums={"method": ["sequential", "reverse", "shapley"]})
def calculate_allowance_attribution(previous_date: str, current_date: str, method: str = "sequential") -> dict:
    """Allowance change split into drivers: macro, credit deterioration, portfolio growth, model/methodology, paydowns/runoff.

    method: 'sequential' (reported waterfall), 'reverse' (opposite order) or 'shapley' (average over all orders)
    """
    check_order(previous_date, current_date)
    labels = driver_labels()
    d = _drivers(previous_date, current_date, method)
    total = sum(d.values())
    rows = sorted(({"driver": labels[k], "driver_id": k, "impact": v, "share_pct": v / total * 100 if total else None}
                   for k, v in d.items()), key=lambda r: r["impact"], reverse=True)
    out = {"method": method, "previous_date": previous_date, "current_date": current_date,
           "drivers": rows, "total_explained": total, "primary_driver": rows[0]["driver"]}
    if method == "sequential":
        out["order"] = [s["label"] for s in config.methodology()["attribution"]["steps"]]
    return out


@tool("reconciliation")
def reconcile_allowance_change(previous_date: str, current_date: str) -> dict:
    """Check that the attributed drivers add up to the booked allowance change: residual, % explained, PASS/WARNING/FAIL, robustness."""
    check_order(previous_date, current_date)
    s = store()
    cfg = config.load("reconciliation")
    rep_prev, rep_cur = s.reported_allowance(previous_date), s.reported_allowance(current_date)
    reported = rep_cur - rep_prev
    w = A.waterfall(previous_date, current_date)
    labels = driver_labels()
    seq = A.driver_totals(w["steps"])
    explained = sum(seq.values())
    residual = reported - explained
    res_pct = abs(residual) / abs(reported) * 100 if reported else 0.0
    status = "PASS" if res_pct <= cfg["pass_max_pct"] else "WARNING" if res_pct <= cfg["warning_max_pct"] else "FAIL"
    methods = {m: _drivers(previous_date, current_date, m) for m in ("sequential", "reverse", "shapley")}
    primary = {m: labels[max(d, key=d.get)] for m, d in methods.items()}
    stable = len(set(primary.values())) == 1
    if cfg.get("require_stable_primary_driver") and not stable and status == "PASS":
        status = "WARNING"
    return {
        "previous_date": previous_date, "current_date": current_date,
        "reported_previous_allowance": rep_prev, "reported_current_allowance": rep_cur, "reported_change": reported,
        "explained_change": explained, "residual": residual, "residual_pct_of_change": res_pct,
        "pct_explained": explained / reported * 100 if reported else None,
        "components": {labels[k]: v for k, v in seq.items()},
        "opening_ties_to_booked": abs(w["start_allowance"] - rep_prev) < 0.01,
        "closing_ties_to_booked": abs(w["end_allowance"] - rep_cur) < 0.01,
        "thresholds_pct": {"pass": cfg["pass_max_pct"], "warning": cfg["warning_max_pct"]},
        "status": status,
        "robustness": {"by_method": {m: {labels[k]: v for k, v in d.items()} for m, d in methods.items()},
                       "primary_driver_by_method": primary, "primary_driver_stable": stable},
        "method": "sequential counterfactual waterfall; engine re-run after each step; residual measured against booked runs",
    }
