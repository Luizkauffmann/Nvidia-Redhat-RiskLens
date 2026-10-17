"""Model Risk Agent tools: model versions, parameters, overlays and the model registry."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.tools.base import Framework, ToolError, get_context, tool
from src.utils import config

MODEL_AGENTS = ["orchestrator", "model_risk_agent"]
MODEL_COLS = ["pd_model_id", "lgd_model_id", "ead_model_id"]


def _params(model_id: str) -> dict[str, float]:
    mp = get_context().store.model_parameters
    x = mp[mp["model_id"] == model_id]
    return dict(zip(x["parameter"], x["value"].astype(float)))


def _attr_step(previous_date: str, current_date: str, framework: str, step: str) -> float:
    attr = get_context().attribution(previous_date, current_date, framework)
    return float(sum(s["impact"] for s in attr.steps if s["step"] == step))


@tool("model", MODEL_AGENTS + ["reconciliation_agent"])
def compare_model_versions(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Which PD / LGD / EAD model versions each product used on both dates, whether any changed, and models awaiting validation that are NOT in production."""
    ctx = get_context()
    a0 = ctx.store.model_assignments_at(previous_date).set_index("product_type")
    a1 = ctx.store.model_assignments_at(current_date).set_index("product_type")
    meta = ctx.store.model_metadata.set_index("model_id")
    rows, changes = [], []
    for prod in a0.index.union(a1.index):
        row = {"product_type": prod, "portfolio": (a1 if prod in a1.index else a0).loc[prod, "portfolio"]}
        for col in MODEL_COLS:
            p = a0[col].get(prod)
            c = a1[col].get(prod)
            row[f"{col[:-3]}_previous"], row[f"{col[:-3]}_current"] = p, c
            if p != c:
                changes.append({"product_type": prod, "model_type": col.split("_")[0].upper(), "previous": p, "current": c})
        rows.append(row)
    used = sorted(set(a0[MODEL_COLS].stack()).union(a1[MODEL_COLS].stack()))
    pending = meta[meta["status"].isin(["IN_VALIDATION", "PENDING_APPROVAL"])]
    loans1 = ctx.store.loans_at(current_date)
    loan_level = {col: sorted(loans1[col].dropna().unique()) for col in MODEL_COLS}
    return {"previous_date": previous_date, "current_date": current_date,
            "any_model_version_changed": bool(changes), "changes": changes, "by_product": rows,
            "models_in_use": [{"model_id": m, **meta.loc[m, ["model_type", "version", "status", "deployment_date",
                                                              "calibration_date"]].to_dict()} for m in used if m in meta.index],
            "candidate_models_not_in_production": [{"model_id": m, **r[["model_type", "portfolio", "status", "description"]].to_dict()}
                                                   for m, r in pending.iterrows()],
            "loan_level_model_ids_current": loan_level,
            "run_dates": {"previous": a0["model_run_date"].iloc[0], "current": a1["model_run_date"].iloc[0]},
            "allowance_impact_of_model_change": _attr_step(previous_date, current_date, framework, "model_change")}


@tool("model", MODEL_AGENTS)
def compare_model_parameters(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Compare the parameters (coefficients, centers, macro sensitivities, LGD floors/caps) of the models used on both dates, plus the methodology settings fingerprint."""
    ctx = get_context()
    a0 = ctx.store.model_assignments_at(previous_date).set_index("product_type")
    a1 = ctx.store.model_assignments_at(current_date).set_index("product_type")
    diffs, compared = [], []
    for prod in a0.index.intersection(a1.index):
        for col in MODEL_COLS:
            m0, m1 = a0.loc[prod, col], a1.loc[prod, col]
            p0, p1 = _params(m0), _params(m1)
            delta = [{"parameter": k, "previous": p0.get(k), "current": p1.get(k)}
                     for k in sorted(set(p0) | set(p1)) if p0.get(k) != p1.get(k)]
            compared.append({"product_type": prod, "model_type": col.split("_")[0].upper(), "previous_model": m0,
                             "current_model": m1, "parameters_compared": len(set(p0) | set(p1)),
                             "parameters_changed": len(delta)})
            if delta:
                diffs.append({"product_type": prod, "previous_model": m0, "current_model": m1, "changes": delta})
    method_hash = hashlib.sha256((config.config_dir() / "methodology.yaml").read_bytes()).hexdigest()[:12]
    m = config.methodology()
    return {"previous_date": previous_date, "current_date": current_date,
            "any_parameter_changed": bool(diffs), "parameter_changes": diffs, "comparisons": compared,
            "methodology": {"fingerprint": method_hash, "applies_to": "both runs (single methodology version in this dataset)",
                            "reasonable_supportable_quarters": m["macro"]["reasonable_supportable_quarters"],
                            "reversion_quarters": m["macro"]["reversion_quarters"],
                            "discounting": m["calculation"]["discount"]},
            "allowance_impact_of_model_change": _attr_step(previous_date, current_date, framework, "model_change"),
            "notes": "Parameters are versioned with the model ID; identical IDs mean identical parameters (verified field by field above)."}


@tool("model", MODEL_AGENTS + ["reconciliation_agent"])
def compare_management_overlays(previous_date: str, current_date: str, framework: Framework = "CECL") -> dict[str, Any]:
    """Management (qualitative) overlays on both dates by overlay ID and portfolio, with reasons, the total change and its allowance impact."""
    ctx = get_context()
    o0 = ctx.store.overlays_at(previous_date).set_index("overlay_id")
    o1 = ctx.store.overlays_at(current_date).set_index("overlay_id")
    rows = []
    for oid in o0.index.union(o1.index):
        a = float(o0["amount"].get(oid, 0.0))
        b = float(o1["amount"].get(oid, 0.0))
        src = o1 if oid in o1.index else o0
        rows.append({"overlay_id": oid, "portfolio": src.loc[oid, "portfolio"], "reason": src.loc[oid, "reason"],
                     "amount_previous": a, "amount_current": b, "change": b - a,
                     "status": "NEW" if oid not in o0.index else "REMOVED" if oid not in o1.index else
                     ("CHANGED" if a != b else "UNCHANGED")})
    t0, t1 = float(o0["amount"].sum()), float(o1["amount"].sum())
    return {"framework": framework, "previous_date": previous_date, "current_date": current_date, "currency": "USD",
            "overlay_total_previous": t0, "overlay_total_current": t1, "change": t1 - t0,
            "overlays_changed": any(r["status"] != "UNCHANGED" for r in rows), "overlays": rows,
            "allowance_impact_of_overlay_change": _attr_step(previous_date, current_date, framework, "overlay")}


@tool("model", MODEL_AGENTS + ["credit_agent", "macro_agent"])
def get_model_metadata(model_id: str) -> dict[str, Any]:
    """Registry entry for one model: type, portfolio, version, status, deployment and calibration dates, description, parameters and which products use it on each date."""
    ctx = get_context()
    meta = ctx.store.model_metadata.set_index("model_id")
    key = model_id.strip()
    match = [m for m in meta.index if m.lower() == key.lower()]
    if not match:
        close = [m for m in meta.index if key.split("_")[0].lower() in m.lower()]
        raise ToolError(f"Unknown model_id '{model_id}'", "not_found",
                        {"did_you_mean": close[:6], "model_ids": list(meta.index)})
    mid = match[0]
    asg = ctx.store.model_assignments
    used = []
    for col in MODEL_COLS:
        x = asg[asg[col] == mid]
        used += [{"as_of_date": d, "product_type": p, "role": col.split("_")[0].upper()}
                 for d, p in zip(x["as_of_date"], x["product_type"])]
    return {"model_id": mid, **meta.loc[mid].to_dict(), "parameters": _params(mid), "used_by": used,
            "in_production_on": sorted({u["as_of_date"].strftime("%Y-%m-%d") for u in used})}
