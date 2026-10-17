"""Phase 2: deterministic tools.

* every tool is registered, documented, schema-valid and returns JSON-safe output
* the demo questions (README "Demo success criteria") are answered correctly
  *through the tools alone*, checked against the hidden ground truth
* the tool layer recovers from the mistakes small models make
* reconciliation genuinely detects a mismatch (it is not plugged)
* tools never read the ground truth
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.tools import REGISTRY, ToolCallLog, ToolContext, call_tool, list_tools, set_context, tool_schemas
from src.tools.reconciliation_tools import classify
from tests.conftest import CURR, PREV

ROOT = Path(__file__).resolve().parents[1]
PAIR = {"previous_date": PREV, "current_date": CURR}


@pytest.fixture(scope="module")
def ctx(generated, attribution, tmp_path_factory):
    c = ToolContext(data_dir=generated[0], cache_dir=tmp_path_factory.mktemp("cache"), disk_cache=False)
    c._mem[f"attr_default_CECL_{PREV}_{CURR}"] = attribution      # reuse the session's attribution run
    set_context(c)
    yield c
    set_context(None)


def run(ctx, name, **args):
    rec = call_tool(name, args, ctx=ctx, log=None)
    assert rec.status == "ok", rec.error
    json.dumps(rec.result)                                           # JSON-safe
    return rec.result


# --------------------------------------------------------------------------- registry
def test_registry_has_all_spec_tools():
    spec = {"get_portfolio_snapshot", "get_allowance_summary", "compare_allowance", "calculate_ecl",
            "calculate_segment_attribution", "calculate_portfolio_attribution", "calculate_pd_migration",
            "calculate_lgd_migration", "calculate_ead_change", "calculate_rating_migration", "calculate_macro_attribution",
            "calculate_variable_attribution", "compare_macro_scenarios", "compare_scenario_weights",
            "identify_top_contributors", "get_loan_details", "get_counterparty_details", "compare_model_versions",
            "get_model_metadata", "reconcile_allowance_change", "compare_portfolios", "calculate_balance_change",
            "calculate_new_originations", "calculate_paydowns", "compare_borrower_periods", "run_scenario_analysis",
            "calculate_segment_macro_sensitivity", "compare_model_parameters", "compare_management_overlays"}
    assert spec <= set(REGISTRY)


def test_schemas_are_valid_and_documented(ctx):
    for s in tool_schemas(ctx=ctx):
        f = s["function"]
        assert f["description"] and len(f["description"]) < 400, f["name"]
        props, req = f["parameters"]["properties"], f["parameters"]["required"]
        assert set(req) <= set(props)
        json.dumps(s)
        for p in ("previous_date", "current_date", "as_of_date"):
            if p in props:
                assert [d for d in props[p]["enum"] if d is not None] == ctx.dates


def test_agents_config_matches_registry_tags():
    cfg = yaml.safe_load((ROOT / "config" / "agents.yaml").read_text())["agents"]
    for agent, spec in cfg.items():
        assert set(spec["tools"]) <= set(REGISTRY), agent
        assert spec["tools"] == list_tools(agent), f"{agent}: regenerate config/agents.yaml"
    # specialists get a focused subset
    assert all(len(cfg[a]["tools"]) < len(REGISTRY) for a in cfg if a != "orchestrator")


def test_tools_never_read_ground_truth():
    for path in (ROOT / "src" / "tools").glob("*.py"):
        text = path.read_text()
        assert "read_ground_truth" not in text and "ground_truth" not in re.sub(r'""".*?"""', "", text, flags=re.S), path


# --------------------------------------------------------------------------- demo questions vs ground truth
def test_q1_to_q4_allowance_levels_and_change(ctx, store):
    r = run(ctx, "compare_allowance", **PAIR)
    s = store.summary[store.summary.framework == "CECL"].set_index(pd.to_datetime(store.summary.as_of_date[store.summary.framework == "CECL"]))
    assert r["previous_allowance"] == pytest.approx(s.loc[PREV, "allowance"], abs=0.01)
    assert r["current_allowance"] == pytest.approx(s.loc[CURR, "allowance"], abs=0.01)
    assert r["change"] == pytest.approx(r["current_allowance"] - r["previous_allowance"], abs=0.01)
    assert r["change_pct"] == pytest.approx(100 * r["change"] / r["previous_allowance"], rel=1e-6)
    assert r["direction"] == "increase"
    assert sum(p["change"] for p in r["by_portfolio"]) == pytest.approx(r["change"], abs=1.0)


def test_q5_major_drivers_and_primary_driver(ctx, ground_truth):
    r = run(ctx, "calculate_portfolio_attribution", **PAIR)
    assert r["primary_driver"] == ground_truth["design"]["primary_driver"]
    assert r["residual"] == pytest.approx(0.0, abs=1e-6)
    assert sum(d["impact"] for d in r["drivers"]) == pytest.approx(r["change"], abs=1e-6)


def test_q6_portfolio_contributing_most(ctx, ground_truth):
    r = run(ctx, "calculate_segment_attribution", **PAIR)
    assert r["largest_contributing_segment"] == ground_truth["design"]["primary_portfolio"]
    assert sum(s["total_change"] for s in r["segments"]) == pytest.approx(r["total_change"], abs=1.0)
    r2 = run(ctx, "compare_allowance", **PAIR)
    assert r2["largest_portfolio_change"] == ground_truth["design"]["primary_portfolio"]


def test_q7_top_borrowers(ctx, ground_truth):
    r = run(ctx, "identify_top_contributors", **PAIR, n=10)
    ids = [c["counterparty_id"] for c in r["contributors"]]
    assert ids[0] == ground_truth["design"]["largest_borrower"]
    assert len(set(ids) & set(ground_truth["design"]["deteriorated_cre_borrowers"])) >= 6
    top = r["contributors"][0]
    assert top["main_driver"] == "credit_deterioration"
    assert any(c.startswith("RATING_DOWNGRADE") for c in top["reason_codes"])
    assert top["pd_12m_current"] > 2 * top["pd_12m_previous"]
    credit = run(ctx, "identify_top_contributors", **PAIR, n=20, driver="credit_deterioration")
    cohort = set(ground_truth["design"]["deteriorated_cre_borrowers"])
    assert len({c["counterparty_id"] for c in credit["contributors"]} & cohort) >= 12


def test_q8_q9_pd_and_lgd_increased(ctx):
    pdm = run(ctx, "calculate_pd_migration", **PAIR)
    assert pdm["total"]["pd_12m_current"] > pdm["total"]["pd_12m_previous"]
    cre = next(p for p in pdm["by_portfolio"] if p["portfolio"] == "CRE")
    assert cre["base_pd_change_bps"] > 0                    # borrower deterioration, not only macro
    lgd = run(ctx, "calculate_lgd_migration", **PAIR)
    assert lgd["total"]["lgd_change_pp"] > 0
    assert lgd["by_portfolio"][0]["portfolio"] == "CRE"


def test_q10_exposure_grew_and_bridge_ties(ctx, ground_truth):
    b = run(ctx, "calculate_balance_change", **PAIR)
    assert b["total"]["net_change"] > 0
    assert b["total"]["bridge_check"] == pytest.approx(0.0, abs=0.05)
    assert all(abs(p["bridge_check"]) < 0.05 for p in b["by_portfolio"])
    new = run(ctx, "calculate_new_originations", **PAIR, portfolio="CRE")
    assert new["balance"] == pytest.approx(ground_truth["design"]["new_cre_exposure"], rel=0.01)
    assert run(ctx, "calculate_ead_change", **PAIR)["ead_change"] > 0


def test_q11_macro_variable_with_largest_impact(ctx, ground_truth):
    want = ground_truth["design"]["primary_macro_variable"]
    seq = run(ctx, "calculate_macro_attribution", **PAIR)
    assert seq["top_variable"] == want
    sh = run(ctx, "calculate_variable_attribution", **PAIR)            # exact Shapley
    assert sh["top_variable"] == want and sh["top_variable_agrees_with_sequential"]
    assert sum(v["impact"] for v in sh["variables"]) == pytest.approx(seq["macro_impact"], abs=1e-3)
    cre = next(p for p in sh["by_portfolio"] if p["portfolio"] == "CRE")
    assert cre["top_variable"] == "cre_price_growth"


def test_q12_scenario_weights_changed(ctx, ground_truth):
    r = run(ctx, "compare_scenario_weights", **PAIR)
    assert r["weights_changed"] is ground_truth["design"]["scenario_weights_changed"] is True
    assert r["weight_shifted_toward"] == "severe_adverse"
    assert r["allowance_impact_of_weight_change"] > 0


def test_q13_model_versions_unchanged(ctx, ground_truth):
    r = run(ctx, "compare_model_versions", **PAIR)
    assert r["any_model_version_changed"] is ground_truth["design"]["model_changed"] is False
    cre = next(p for p in r["by_product"] if p["product_type"] == "Commercial Real Estate")
    assert cre["pd_model_previous"] == cre["pd_model_current"] == "CRE_PD_v4"
    assert "CRE_PD_v5" in {m["model_id"] for m in r["candidate_models_not_in_production"]}
    assert r["allowance_impact_of_model_change"] == 0.0
    assert run(ctx, "compare_model_parameters", **PAIR)["any_parameter_changed"] is False
    ov = run(ctx, "compare_management_overlays", **PAIR)
    assert ov["overlays_changed"] is ground_truth["design"]["overlays_changed"]


def test_q14_cre_story(ctx, ground_truth):
    lead = ground_truth["design"]["largest_borrower"]
    b = run(ctx, "compare_borrower_periods", counterparty_id=lead, **PAIR)
    loan = b["loans"][0]
    feats = loan["pd_logit_change"]["by_feature"]
    assert sum(f["change"] for f in feats) == pytest.approx(loan["pd_logit_change"]["total_change"], abs=1e-9)
    assert {f["feature"] for f in feats if f["change"] > 0} >= {"rating_notch", "dscr", "ltv"}
    assert b["main_driver"] == "credit_deterioration"
    rm = run(ctx, "calculate_rating_migration", **PAIR, portfolio="CRE")
    assert rm["summary"]["downgrades"] >= 20


def test_q15_reconciliation_passes_and_is_robust(ctx):
    r = run(ctx, "reconcile_allowance_change", **PAIR)
    assert r["status"] == "PASS" and r["overall_status"] == "PASS"
    assert r["pct_explained"] == pytest.approx(100.0, abs=0.1)
    assert all(c["status"] == "PASS" for c in r["controls"])
    rob = r["order_robustness"]
    assert rob["primary_driver_robust"] and set(rob["primary_driver_by_method"].values()) == {"macroeconomic"}
    shap = {d["driver"]: d["shapley"] for d in rob["drivers"]}
    assert sum(shap.values()) == pytest.approx(r["reported_change"], abs=1e-3)


def test_engine_reproduces_official_runs(ctx):
    for d in (PREV, CURR):
        assert run(ctx, "calculate_ecl", as_of_date=d)["reproduces_reported"]


def test_scenario_analysis_is_exact_reweighting(ctx):
    r = run(ctx, "run_scenario_analysis", as_of_date=CURR,
            weights={"baseline": 0.58, "adverse": 0.30, "severe_adverse": 0.12})
    assert r["custom"]["difference_vs_official"] == pytest.approx(0.0, abs=0.01)
    s = r["single_scenario_allowance"]
    assert s["baseline"] < r["official_allowance"] < s["severe_adverse"]


def test_unemployment_sensitivity_is_positive_everywhere(ctx):
    r = run(ctx, "calculate_segment_macro_sensitivity", as_of_date=CURR, variable="unemployment_rate", shock=1.0)
    assert r["allowance_change"] > 0 and all(p["ecl_change"] > 0 for p in r["by_portfolio"])


@pytest.mark.parametrize("name,args", [
    ("get_reporting_context", {}), ("get_allowance_summary", {"as_of_date": CURR}),
    ("get_portfolio_snapshot", {"as_of_date": CURR}), ("compare_portfolios", PAIR),
    ("calculate_paydowns", PAIR), ("compare_macro_scenarios", {**PAIR, "include_paths": True}),
    ("get_loan_details", {"loan_id": "LN_0000001"}), ("get_model_metadata", {"model_id": "CRE_PD_v4"}),
])
def test_other_tools_run(ctx, name, args):
    run(ctx, name, **args)


# --------------------------------------------------------------------------- robustness for small models
def test_argument_repair(ctx):
    rec = call_tool("compare_allowance", {"previous_date": "July 2026", "current_date": "2026-08", "framework": "cecl"},
                    ctx=ctx, log=None)
    assert rec.status == "ok" and rec.normalized
    assert rec.parameters == {"previous_date": PREV, "current_date": CURR, "framework": "CECL"}
    rec = call_tool("calculate_pd_migration", json.dumps({**PAIR, "portfolio": "commercial real estate"}), ctx=ctx, log=None)
    assert rec.status == "ok" and rec.parameters["portfolio"] == "CRE"
    rec = call_tool("identify_top_contributors", {**PAIR, "n": "3"}, ctx=ctx, log=None)
    assert rec.status == "ok" and len(rec.result["contributors"]) == 3


@pytest.mark.parametrize("name,args,error_type,hint_key", [
    ("compare_allowanse", PAIR, "unknown_tool", "did_you_mean"),
    ("compare_allowance", {"previous_date": "2025-01-31", "current_date": CURR}, "invalid_arguments", "available_dates"),
    ("compare_allowance", {"current_date": CURR}, "invalid_arguments", "required"),
    ("compare_allowance", {"previous_date": CURR, "current_date": PREV}, "invalid_arguments", "available_dates"),
    ("compare_allowance", {**PAIR, "colour": "red"}, "invalid_arguments", "allowed_arguments"),
    ("calculate_pd_migration", {**PAIR, "portfolio": "Boats"}, "invalid_arguments", "allowed_portfolios"),
    ("identify_top_contributors", {**PAIR, "n": "lots"}, "invalid_arguments", "expected_type"),
    ("get_model_metadata", {"model_id": "CRE_PD_v9"}, "not_found", "did_you_mean"),
    ("run_scenario_analysis", {"as_of_date": CURR, "weights": {"baseline": 0.5}}, "invalid_arguments", "example"),
])
def test_structured_errors_with_recovery_hints(ctx, name, args, error_type, hint_key):
    rec = call_tool(name, args, ctx=ctx, log=None)
    assert rec.status == "error"
    assert rec.error["error_type"] == error_type
    assert hint_key in (rec.error["hint"] or {})


def test_malformed_json_arguments(ctx):
    rec = call_tool("compare_allowance", "{previous_date: July", ctx=ctx, log=None)
    assert rec.status == "error" and rec.error["error_type"] == "malformed_arguments"


def test_provenance_and_log(ctx, tmp_path):
    log = ToolCallLog(tmp_path / "calls.jsonl")
    ok = call_tool("compare_allowance", PAIR, ctx=ctx, log=log, caller="test")
    call_tool("nope", {}, ctx=ctx, log=log)
    p = ok.provenance()
    assert p["tool_name"] == "compare_allowance" and p["parameters"]["framework"] == "CECL"
    assert p["call_id"] and p["timestamp"] and p["result_sha256"]
    again = call_tool("compare_allowance", PAIR, ctx=ctx, log=None)
    assert again.result_sha256 == ok.result_sha256                 # deterministic
    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and "result" not in json.loads(lines[0])
    st = log.stats()
    assert st["calls"] == 2 and st["failed"] == 1 and st["errors_by_type"] == {"unknown_tool": 1}


# --------------------------------------------------------------------------- reconciliation is a real control
def test_classify_thresholds():
    assert classify(0.5, 1e6)[0] == "PASS"                      # under absolute tolerance
    assert classify(500.0, 1e6)[0] == "PASS"                    # 0.05%
    assert classify(5_000.0, 1e6)[0] == "WARNING"               # 0.5%
    assert classify(50_000.0, 1e6)[0] == "FAIL"                 # 5%


@pytest.mark.parametrize("bump,expected", [(5_000.0, "PASS"), (60_000.0, "WARNING"), (1_000_000.0, "FAIL")])
def test_reconciliation_detects_reported_mismatch(ctx, bump, expected):
    """If the booked August allowance differs from what the engine explains, the control must say so."""
    original = ctx.store.summary
    tampered = original.copy()
    m = (tampered["framework"] == "CECL") & (pd.to_datetime(tampered["as_of_date"]) == pd.Timestamp(CURR))
    tampered.loc[m, "allowance"] += bump
    ctx.store.__dict__["summary"] = tampered
    try:
        r = run(ctx, "reconcile_allowance_change", **PAIR, check_order_sensitivity=False)
    finally:
        ctx.store.__dict__["summary"] = original
    assert r["status"] == expected
    assert r["residual"] == pytest.approx(bump, abs=0.01)
    failing = {c["control"] for c in r["controls"] if c["status"] != "PASS"}
    assert "engine_reproduces_current_allowance" in failing
