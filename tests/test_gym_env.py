"""NeMo Gym environment: task bank, verifier (reward), dataset rows and the resources server."""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from src.evaluation import verifier
from src.evaluation.gym_dataset import build_rows, responses_tools
from src.evaluation.tasks import build_tasks, load_ground_truth
from src.evaluation.verifier import ToolCallTrace, score
from src.tools import ToolContext, call_tool, set_context
from src.utils import config
from tests.conftest import CURR, PREV

PAIR = {"previous_date": "2026-07-31", "current_date": "2026-08-31"}


@pytest.fixture(scope="module", autouse=True)
def ctx(generated, attribution, tmp_path_factory):
    """Tools read the session's generated data; the task bank reads its ground truth."""
    c = ToolContext(data_dir=generated[0], cache_dir=tmp_path_factory.mktemp("cache"), disk_cache=False)
    c._mem[f"attr_default_CECL_{PREV}_{CURR}"] = attribution
    set_context(c)
    old = os.environ.get("RISKLENS_GROUND_TRUTH_DIR")
    os.environ["RISKLENS_GROUND_TRUTH_DIR"] = str(generated[1])
    config.reload()
    yield c
    set_context(None)
    if old is None:
        os.environ.pop("RISKLENS_GROUND_TRUTH_DIR", None)
    else:
        os.environ["RISKLENS_GROUND_TRUTH_DIR"] = old
    config.reload()


@pytest.fixture(scope="module")
def tasks():
    return {t.task_id: t.to_dict() for t in build_tasks()}


def _trace(name, args):
    rec = call_tool(name, args)
    return ToolCallTrace(name=name, arguments=args, output={"status": rec.status, "result": rec.result, "error": rec.error})


def test_task_bank_covers_the_demo_questions(tasks):
    canonical = [t for t in tasks.values() if t["paraphrase_of"] is None]
    assert len(canonical) == 15
    for t in tasks.values():
        assert t["checks"] and t["required_tools"]


def test_expected_facts_come_from_ground_truth(tasks):
    gt = load_ground_truth()
    borrower_check = tasks["q07_top_borrowers"]["checks"][0]
    assert gt["design"]["largest_borrower"] in borrower_check["any_of"][0]
    change = next(c for c in tasks["q03_dollar_change"]["checks"] if c["type"] == "number")
    assert change["value"] == pytest.approx(gt["observed_cecl"]["change"])


def test_correct_grounded_answer_scores_high(tasks):
    t = tasks["q03_dollar_change"]
    calls = [_trace("compare_allowance", PAIR)]
    c = calls[0].result
    answer = f"The allowance increased by ${c['change'] / 1e6:.2f}M ({c['change_pct']:.1f}%). [compare_allowance]"
    res = score(answer, calls, t)
    assert res.components["correctness"] == 1.0
    assert res.components["grounding"] == 1.0
    assert res.reward == pytest.approx(1.0)


def test_hallucinated_numbers_lose_reward(tasks):
    t = tasks["q03_dollar_change"]
    calls = [_trace("compare_allowance", PAIR)]
    res = score("The allowance increased by $25.17M, mostly from new loans worth $412.9M.", calls, t)
    assert res.components["grounding"] == 0.0
    assert res.components["correctness"] < 1.0
    assert res.reward < 0.6


def test_answer_without_tools_is_penalised(tasks):
    t = tasks["q07_top_borrowers"]
    res = score("Investigate CP_040408 first.", [], t)
    assert res.components["correctness"] == 1.0
    assert res.components["tool_selection"] == 0.0
    assert res.components["call_validity"] == 0.0
    assert res.reward < 0.3          # a fact recalled without any tool evidence is not an investigation


def test_no_answer_scores_zero(tasks):
    res = score("", [_trace("compare_allowance", PAIR)], tasks["q01_previous_allowance"])
    assert res.reward == 0.0 and not res.has_answer


def test_failed_calls_and_overspending_reduce_reward(tasks):
    t = tasks["q15_reconcile"]
    bad = _trace("reconcile_allowance_change", {"previous_date": "2025-01-31", "current_date": "2026-08-31"})
    assert not bad.ok
    good = _trace("reconcile_allowance_change", PAIR)
    r = good.result
    answer = f"Reconciliation {r['overall_status']}: reported ${r['reported_change'] / 1e6:.2f}M, residual ${r['residual']:.2f}."
    clean = score(answer, [good], t)
    messy = score(answer, [bad] + [good] * 4, t)
    assert clean.reward > messy.reward
    assert messy.components["call_validity"] == pytest.approx(0.8)
    assert messy.components["efficiency"] == pytest.approx(3 / 5)


def test_model_change_question_rejects_wrong_claim(tasks):
    t = tasks["q13_model_change"]
    calls = [_trace("compare_model_versions", PAIR)]
    right = score("No. The same model versions were used in both months (CRE_PD_v4).", calls, t)
    wrong = score("Yes, model versions changed: yes, a new model was deployed.", calls, t)
    assert right.components["correctness"] == 1.0
    assert wrong.components["correctness"] < 1.0


def test_responses_output_adapter():
    output = [
        {"type": "function_call", "call_id": "c1", "name": "compare_allowance", "arguments": json.dumps(PAIR)},
        {"type": "function_call_output", "call_id": "c1", "output": json.dumps({"status": "ok", "result": {"change": 1.0}})},
        {"type": "function_call", "call_id": "c2", "name": "nope", "arguments": "{bad json"},
        {"type": "function_call_output", "call_id": "c2", "output": json.dumps({"status": "error", "error": {}})},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Final answer."}]},
    ]
    answer, calls = verifier.from_responses_output(output)
    assert answer == "Final answer."
    assert [c.name for c in calls] == ["compare_allowance", "nope"]
    assert calls[0].ok and calls[0].result == {"change": 1.0}
    assert not calls[1].ok


def test_dataset_rows_are_gym_format():
    splits = build_rows()
    assert len(splits["example"]) == 15
    assert set(r["id"] for r in splits["validation"]).isdisjoint(r["id"] for r in splits["train"])
    row = splits["example"][0]
    params = row["responses_create_params"]
    assert params["input"][0]["role"] == "system" and params["input"][1]["role"] == "user"
    assert params["parallel_tool_calls"] is False
    tool = params["tools"][0]
    assert tool["type"] == "function" and {"name", "description", "parameters"} <= tool.keys()
    assert "checks" in row["verifier_metadata"]
    json.dumps(row)


def test_specialist_tool_lists():
    names = {t["name"] for t in responses_tools("credit_agent")}
    assert "identify_top_contributors" in names
    assert "calculate_variable_attribution" not in names


@pytest.mark.skipif(importlib.util.find_spec("nemo_gym") is None, reason="nemo-gym not installed")
def test_resources_server_endpoints():
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient
    from nemo_gym.server_utils import ServerClient

    from resources_servers.risklens_cecl.app import RiskLensResourcesServer, RiskLensResourcesServerConfig

    cfg = RiskLensResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="risklens_cecl",
                                        warm_cache=False)
    server = RiskLensResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())

    ok = client.post("/compare_allowance", json={"previous_date": "July 2026", "current_date": "aug"}).json()
    assert ok["status"] == "ok" and ok["result"]["current_date"] == "2026-08-31"
    bad = client.post("/compare_allowance", content=b"{previous_date: july").json()
    assert bad["status"] == "error" and bad["error"]["error_type"] == "malformed_arguments"

    row = build_rows()["example"][2]          # q03 dollar change
    change = ok["result"]["change"]
    response = {"id": "r1", "created_at": 0, "model": "m", "object": "response", "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": [],
                "output": [{"type": "function_call", "call_id": "c1", "name": "compare_allowance", "id": "f1",
                            "arguments": json.dumps(PAIR), "status": "completed"},
                           {"type": "function_call_output", "call_id": "c1", "output": json.dumps(ok)},
                           {"type": "message", "role": "assistant", "id": "m1", "status": "completed",
                            "content": [{"type": "output_text", "annotations": [],
                                         "text": f"The allowance increased by ${change / 1e6:.2f}M."}]}]}
    out = client.post("/verify", json={**row, "response": response})
    assert out.status_code == 200, out.text
    body = out.json()
    assert body["reward"] == pytest.approx(1.0)
    assert set(body["reward_components"]) == {"correctness", "grounding", "tool_selection", "call_validity", "efficiency"}


def test_rollouts_to_sft(tmp_path):
    from src.evaluation.trajectories import rollouts_to_sft

    row = build_rows()["example"][0]
    good = {**row, "reward": 0.95, "response": {"output": [
        {"type": "function_call", "call_id": "c1", "name": "get_allowance_summary", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "{\"status\": \"ok\"}"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "July was $99.06M."}]}]}}
    bad = {**good, "reward": 0.2}
    src = tmp_path / "r.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in (good, bad, good)) + "\n")
    out = rollouts_to_sft([src], min_reward=0.8, out=tmp_path / "sft.jsonl")
    assert len(out) == 1                                   # low reward dropped, duplicate removed
    roles = [m["role"] for m in out[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert out[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "get_allowance_summary"
    assert out[0]["tools"][0]["type"] == "function"


def test_mentions_match_through_unicode_spacing(tasks):
    """Nemotron writes narrow no-break spaces and non-breaking hyphens; checks use plain ones."""
    check = {"type": "mentions", "any_of": [r"cre price|commercial real estate price"], "desc": "x"}
    for text in ("CRE\u202fprice growth led", "commercial\u2011real\u2011estate price growth led", "CRE price"):
        assert verifier._check(check, text, 0.015)[0], text
    assert not verifier._check(check, "unemployment led", 0.015)[0]
    absent = {"type": "not_mentions", "none_of": [r"model versions? changed: yes"], "desc": "y"}
    assert not verifier._check(absent, "Model\u00a0versions changed: yes", 0.015)[0]


def test_rollouts_to_sft_can_rescore(tmp_path):
    from src.evaluation.trajectories import rollouts_to_sft

    row = build_rows()["example"][0]                       # q01: July allowance
    july = load_ground_truth()["observed_cecl"]["previous_allowance"]
    stale = {**row, "reward": 0.1, "response": {"output": [
        {"type": "function_call", "call_id": "c1", "name": "get_allowance_summary", "arguments": json.dumps({"as_of_date": PREV})},
        {"type": "function_call_output", "call_id": "c1", "output": json.dumps({"status": "ok", "result": {"total_allowance": july}})},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"July was ${july / 1e6:.2f}\u202fM."}]}]}}
    src = tmp_path / "r.jsonl"
    src.write_text(json.dumps(stale) + "\n")
    assert rollouts_to_sft([src], min_reward=0.8) == []    # stored score
    out = rollouts_to_sft([src], min_reward=0.8, rescore=True)
    assert len(out) == 1 and out[0]["metadata"]["reward"] >= 0.8
