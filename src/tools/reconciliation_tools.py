"""Reconciliation Agent tools: does the explanation add up to the *reported* change?

The reported allowance comes from the stored official runs (what Finance books).
The explanation comes from the attribution walk, which re-runs the engine.
They are independent, so a mismatch is a genuine finding — unlike the legacy
Dataiku workbench, which rescaled components to fit the total.

Controls
1. Engine reproduces the reported previous and current allowance.
2. Residual = reported change - sum of attributed drivers  -> PASS / WARNING / FAIL
   using config/reconciliation.yaml thresholds (% of |reported change|).
3. Loan-level ties: per-loan contributions (+ portfolio overlays) add up to the drivers.
4. Portfolio ties: attributed change per portfolio = reported change per portfolio.
5. Order robustness (optional): the primary driver is the same under the
   sequential walk, the reversed walk and exact Shapley over drivers.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.tools.base import Framework, driver_labels, get_context, tool
from src.utils import config

RECON_AGENTS = ["orchestrator", "reconciliation_agent"]
_RANK = {"PASS": 0, "WARNING": 1, "FAIL": 2}


def classify(residual: float, change: float, cfg: dict[str, Any] | None = None) -> tuple[str, float | None]:
    """PASS / WARNING / FAIL for a residual, using thresholds from config/reconciliation.yaml."""
    cfg = cfg or config.reconciliation()
    th = cfg["thresholds"]
    if abs(residual) <= float(cfg["absolute_tolerance"]):
        return "PASS", (100.0 * abs(residual) / abs(change)) if change else 0.0
    if not change:
        return "FAIL", None
    pct = 100.0 * abs(residual) / abs(change)
    if pct <= th["pass_max_pct"]:
        return "PASS", pct
    if pct <= th["warning_max_pct"]:
        return "WARNING", pct
    return "FAIL", pct


def _reported_by_portfolio(as_of_date: str, framework: str) -> pd.Series:
    ctx = get_context()
    r = ctx.results(as_of_date, framework).groupby("portfolio")["ecl"].sum()
    ov = ctx.store.overlays_at(as_of_date).groupby("portfolio")["amount"].sum()
    return r.add(ov, fill_value=0.0)


@tool("reconciliation", RECON_AGENTS, slow=True)
def reconcile_allowance_change(previous_date: str, current_date: str, framework: Framework = "CECL",
                               check_order_sensitivity: bool = True) -> dict[str, Any]:
    """Verify the attribution reconciles to the reported allowance change: explained vs reported, residual, % explained, PASS/WARNING/FAIL, loan- and portfolio-level ties and (optionally) whether the primary driver is robust to attribution order."""
    ctx = get_context()
    cfg = config.reconciliation()
    tol = float(cfg["absolute_tolerance"])
    rep0 = float(ctx.summary_row(previous_date, framework)["allowance"])
    rep1 = float(ctx.summary_row(current_date, framework)["allowance"])
    reported_change = rep1 - rep0
    attr = ctx.attribution(previous_date, current_date, framework)
    s = attr.summary
    explained = float(sum(d["impact"] for d in s["drivers"]))
    residual = reported_change - explained
    status, residual_pct = classify(residual, reported_change, cfg)

    controls: list[dict[str, Any]] = []

    def control(name: str, value: float, ok: bool | None = None, st: str | None = None, detail: Any = None):
        st = st or ("PASS" if ok else "FAIL")
        controls.append({"control": name, "value": value, "status": st, **({"detail": detail} if detail is not None else {})})

    control("engine_reproduces_previous_allowance", s["previous_allowance"] - rep0, abs(s["previous_allowance"] - rep0) <= tol)
    control("engine_reproduces_current_allowance", s["current_allowance"] - rep1, abs(s["current_allowance"] - rep1) <= tol)
    control("residual_vs_reported_change", residual, st=status,
            detail={"residual_pct_of_change": residual_pct, "thresholds_pct": cfg["thresholds"]})

    lc = attr.loan_contributions
    step_cols = [st["id"] for st in attr.steps if st["id"] in lc.columns]
    overlay_delta = float(attr.portfolio_contributions["overlay"].sum())
    loan_sum = float(lc[step_cols].to_numpy().sum()) + overlay_delta
    control("loan_contributions_sum_to_explained", loan_sum - explained, abs(loan_sum - explained) <= tol)
    loan_tie = float((lc["ecl_change"] - lc[step_cols].sum(axis=1)).abs().max())
    control("each_loan_steps_sum_to_its_ecl_change", loan_tie, loan_tie <= 1e-4,
            detail="max absolute per-loan gap between step impacts and ECL change")

    rp0, rp1 = _reported_by_portfolio(previous_date, framework), _reported_by_portfolio(current_date, framework)
    rep_port = rp1.sub(rp0, fill_value=0.0)
    att_port = attr.portfolio_contributions["total"]
    gaps = rep_port.sub(att_port, fill_value=0.0)
    port_rows = [{"portfolio": p, "reported_change": float(rep_port.get(p, 0.0)), "attributed_change": float(att_port.get(p, 0.0)),
                  "gap": float(gaps[p])} for p in gaps.index]
    control("portfolio_attribution_ties_to_reported", float(gaps.abs().max()), float(gaps.abs().max()) <= tol)

    robustness = None
    if check_order_sensitivity:
        rev = ctx.attribution(previous_date, current_date, framework, order="reverse").summary
        shap = ctx.driver_shapley(previous_date, current_date, framework)
        labels = driver_labels()
        methods = {"sequential": {d["driver"]: d["impact"] for d in s["drivers"]},
                   "reverse_sequential": {d["driver"]: d["impact"] for d in rev["drivers"]},
                   "shapley": {d["driver"]: d["impact"] for d in shap["drivers"]}}
        table = [{"driver": d, "label": labels[d], **{m: v.get(d, 0.0) for m, v in methods.items()},
                  "max_spread": max(v.get(d, 0.0) for v in methods.values()) - min(v.get(d, 0.0) for v in methods.values())}
                 for d in labels]
        table.sort(key=lambda r: -abs(r["sequential"]))
        primary = {m: max(v, key=lambda d: v[d]) for m, v in methods.items()}
        ranking = {m: [d for d in sorted(v, key=lambda d: -abs(v[d]))] for m, v in methods.items()}
        robust = len(set(primary.values())) == 1
        robustness = {"primary_driver_by_method": primary, "primary_driver_robust": robust,
                      "ranking_by_method": ranking,
                      "ranking_identical": len({tuple(r) for r in ranking.values()}) == 1,
                      "drivers": table, "shapley_engine_runs": shap["engine_runs"],
                      "interpretation_guide": "Spread between methods = interaction effects (e.g. credit deterioration costs more under a worse macro path). Robust primary driver => safe to state it as the main cause."}
        control("primary_driver_robust_to_order", float(len(set(primary.values()))), st="PASS" if robust else "WARNING",
                detail=primary)

    overall = max((c["status"] for c in controls), key=lambda x: _RANK[x])
    return {
        "framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
        "reported_previous_allowance": rep0, "reported_current_allowance": rep1,
        "reported_change": reported_change, "explained_change": explained, "residual": residual,
        "residual_pct_of_change": residual_pct,
        "pct_explained": 100.0 * explained / reported_change if reported_change else None,
        "status": status, "overall_status": overall, "controls": controls,
        "drivers": s["drivers"], "by_portfolio": port_rows, "order_robustness": robustness,
        "thresholds": {**cfg["thresholds"], "absolute_tolerance": tol},
        "source": {"reported": "official run summaries (allowance_summary)",
                   "explained": "sequential counterfactual attribution (engine re-runs)", "method": s["method"]},
    }
