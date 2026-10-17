import json
from pathlib import Path

import pytest

from risklens import config
from risklens.tools import TOOLS, call_tool, tool_schemas
from risklens.tools.registry import _load_tools

_load_tools()


def test_every_tool_runs_with_defaults():
    needs = {"get_allowance_summary": {"as_of_date": "2026-08-31"}, "get_portfolio_snapshot": {"as_of_date": "2026-08-31"},
             "calculate_portfolio_attribution": {"portfolio": "CRE"}, "get_loan_details": {"loan_id": "LN_0000001"},
             "get_counterparty_details": {"counterparty_id": "CP_000001"}, "get_model_metadata": {"model_id": "CRE_PD_v4"},
             "compare_borrower_periods": {"counterparty_id": None}, "run_scenario_analysis": {"as_of_date": "2026-08-31"},
             "calculate_segment_macro_sensitivity": {"as_of_date": "2026-08-31"}}
    top = call_tool("identify_top_contributors", {"n": 1}).result["rows"][0]["counterparty_id"]
    needs["compare_borrower_periods"]["counterparty_id"] = top
    for name in TOOLS:
        r = call_tool(name, needs.get(name, {}))
        assert r.ok, (name, r.error)
        json.dumps(r.result)     # JSON-serialisable
        assert isinstance(r.result, dict)
        assert r.provenance()["result_sha256"]


def test_schemas_are_valid_openai_functions():
    for s in tool_schemas():
        f = s["function"]
        assert s["type"] == "function" and f["name"] and f["description"]
        assert f["parameters"]["type"] == "object"
        for p in f["parameters"]["required"]:
            assert p in f["parameters"]["properties"]


@pytest.mark.parametrize("args,expected", [
    ({"previous_date": "July 2026", "current_date": "Aug"}, ("2026-07-31", "2026-08-31")),
    ({"previous_date": "2026-07", "current_date": "current"}, ("2026-07-31", "2026-08-31")),
    ({"prev_date": "2026-07-31"}, ("2026-07-31", "2026-08-31")),
])
def test_argument_repair(args, expected):
    r = call_tool("compare_allowance", args)
    assert r.ok and (r.arguments["previous_date"], r.arguments["current_date"]) == expected


def test_portfolio_aliases():
    for alias in ("cre", "Commercial Real Estate", "real estate"):
        r = call_tool("calculate_portfolio_attribution", {"portfolio": alias})
        assert r.ok and r.arguments["portfolio"] == "CRE"


@pytest.mark.parametrize("name,args,hint_key", [
    ("get_allowance_summary", {"as_of_date": "2025-01-31"}, "available_dates"),
    ("compare_allowance", {"previous_date": "2026-08-31", "current_date": "2026-07-31"}, "swap_to"),
    ("calculate_portfolio_attribution", {"portfolio": "Boats"}, "valid_portfolios"),
    ("compare_allowanse", {}, "did_you_mean"),
    ("get_model_metadata", {"model_id": "XYZ"}, "available_models"),
])
def test_structured_errors_with_hints(name, args, hint_key):
    r = call_tool(name, args)
    assert not r.ok and r.error
    assert hint_key in r.hint


def test_malformed_json_arguments():
    r = call_tool("compare_allowance", "{not json")
    assert not r.ok and "expected_schema" in r.hint
    r = call_tool("compare_allowance", '{"previous_date": "2026-07-31"}')
    assert r.ok and "parsed JSON string arguments" in r.repairs


def test_agent_allow_lists_reference_real_tools():
    for agent, names in config.load("agents")["agents"].items():
        if names != "all":
            assert set(names) <= set(TOOLS), agent


def test_tools_never_read_ground_truth():
    src = Path(__file__).resolve().parents[1] / "risklens"
    gt_name = config.settings()["ground_truth_file"]
    for f in list((src / "tools").glob("*.py")) + list((src / "agents").glob("*.py")) + [src / "data" / "store.py"]:
        text = f.read_text()
        assert gt_name not in text and "ground_truth" not in text, f
