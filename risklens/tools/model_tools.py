"""Model Risk Agent tools: versions, parameters, overlays, metadata."""
from __future__ import annotations

from risklens import config
from risklens.tools.common import check_order, store
from risklens.tools.registry import ToolError, tool


def _models_used(as_of: str) -> dict:
    df = store().loans(as_of)
    out = {}
    for prod, g in df.groupby("product_type"):
        out[prod] = {k: sorted(g[f"{k}_model_id"].unique().tolist()) for k in ("pd", "lgd", "ead")}
    return out


@tool("model")
def compare_model_versions(previous_date: str, current_date: str) -> dict:
    """PD, LGD and EAD model versions used for each product in both periods, and whether any changed."""
    check_order(previous_date, current_date)
    a, b = _models_used(previous_date), _models_used(current_date)
    rows, changed = {}, []
    for prod in sorted(set(a) | set(b)):
        rows[prod] = {}
        for k in ("pd", "lgd", "ead"):
            pa, pb = a.get(prod, {}).get(k), b.get(prod, {}).get(k)
            rows[prod][k] = {"previous": pa, "current": pb, "changed": pa != pb}
            if pa != pb:
                changed.append(f"{prod} {k.upper()}")
    meta = store().model_metadata
    pending = meta[meta["status"] != "PRODUCTION"][["model_id", "status", "description"]].to_dict("records")
    return {"by_product": rows, "any_model_changed": bool(changed), "changed": changed,
            "models_not_in_production": pending, "source": "model ids stamped on each loan in the booked runs"}


@tool("model")
def compare_model_parameters(previous_date: str, current_date: str) -> dict:
    """Parameters of every model used in either period, and whether parameters differ between the periods."""
    check_order(previous_date, current_date)
    a, b = _models_used(previous_date), _models_used(current_date)
    cfg = config.models()
    used = {m for d in (a, b) for p in d.values() for k in p.values() for m in k}
    params = {m: cfg["pd_models"].get(m) or cfg["lgd_models"].get(m) or cfg["ead_models"].get(m) for m in sorted(used)}
    same_ids = a == b
    return {"models": params, "parameters_changed": not same_ids,
            "note": "parameters are versioned by model id; identical ids in both periods mean identical parameters"}


@tool("model")
def compare_management_overlays(previous_date: str, current_date: str) -> dict:
    """Management overlays (qualitative adjustments) in both periods and the change."""
    check_order(previous_date, current_date)
    a = {o["overlay_id"]: o for o in store().overlay_list(previous_date)}
    b = {o["overlay_id"]: o for o in store().overlay_list(current_date)}
    rows = {k: {"portfolio": (b.get(k) or a.get(k))["portfolio"], "previous": a.get(k, {}).get("amount", 0.0),
                "current": b.get(k, {}).get("amount", 0.0), "reason": (b.get(k) or a.get(k))["reason"]} for k in sorted(set(a) | set(b))}
    for r in rows.values():
        r["change"] = r["current"] - r["previous"]
    return {"overlays": rows, "previous_total": sum(r["previous"] for r in rows.values()),
            "current_total": sum(r["current"] for r in rows.values()),
            "change": sum(r["change"] for r in rows.values()), "changed": any(r["change"] for r in rows.values())}


@tool("model")
def get_model_metadata(model_id: str) -> dict:
    """Registry entry and parameters for one model id (e.g. CRE_PD_v4)."""
    meta = store().model_metadata
    row = meta[meta["model_id"].str.upper() == str(model_id).upper()]
    if row.empty:
        raise ToolError(f"Model '{model_id}' not found.", {"available_models": meta["model_id"].tolist()})
    r = row.iloc[0].to_dict()
    cfg = config.models()
    mid = r["model_id"]
    r["parameters"] = cfg["pd_models"].get(mid) or cfg["lgd_models"].get(mid) or cfg["ead_models"].get(mid)
    r["used_in_dates"] = [d for d in store().dates if any(mid in v for p in _models_used(d).values() for v in p.values())]
    return r
