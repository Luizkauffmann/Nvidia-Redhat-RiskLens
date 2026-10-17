"""Phase 3: the orchestrating agent, the LLM provider layer and the grounding check.

No network: the mock planner and a scripted fake OpenAI client stand in for
Nemotron. The fake client reproduces the failure modes small models show
(bad arguments, tool calls written as text, invented numbers) so the recovery
paths are tested, not just the happy path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from src.agents.base import ToolAgent, compact
from src.agents.grounding import check_grounding, extract_numbers
from src.agents.orchestrator import ORCHESTRATOR_PROMPT, build_orchestrator
from src.llm.mock_provider import MockProvider
from src.llm.openai_compat import OpenAICompatibleProvider, parse_text_tool_calls
from src.llm.provider import get_provider, llm_settings
from src.tools import ToolCallLog, ToolContext, list_tools, set_context
from tests.conftest import CURR, PREV


@pytest.fixture(scope="module", autouse=True)
def ctx(generated, attribution, tmp_path_factory):
    c = ToolContext(data_dir=generated[0], cache_dir=tmp_path_factory.mktemp("cache"), disk_cache=False)
    c._mem[f"attr_default_CECL_{PREV}_{CURR}"] = attribution
    set_context(c)
    yield c
    set_context(None)


# --------------------------------------------------------------------------- mock planner end-to-end
@pytest.mark.parametrize("question,must_use,must_say", [
    ("Why did our allowance increase in August?", {"calculate_portfolio_attribution"}, "Macroeconomic"),
    ("Which macro variable had the largest impact?", {"calculate_variable_attribution"}, "unemployment_rate"),
    ("Which borrowers should I investigate?", {"identify_top_contributors"}, "CP_040408"),
    ("Could this just be because we changed models?", {"compare_model_versions"}, "Model versions changed: no"),
    ("Did scenario weights change?", {"compare_scenario_weights"}, "severe_adverse"),
    ("Does everything reconcile?", {"reconcile_allowance_change"}, "PASS"),
])
def test_mock_agent_answers_demo_questions(question, must_use, must_say, ground_truth):
    run = build_orchestrator(MockProvider()).run(question)
    assert run.status == "answered"
    assert must_use <= set(run.tools_used)
    assert must_say in run.answer
    assert run.grounding["status"] in ("PASS", "NO_NUMBERS")
    assert all(r.status == "ok" for r in run.tool_records)
    for r in run.tool_records:                                   # provenance on every call
        p = r.provenance()
        assert p["call_id"] and p["result_sha256"] and p["parameters"]["previous_date"] == PREV


def test_run_record_is_serialisable():
    run = build_orchestrator(MockProvider()).run("Why did the allowance increase?")
    d = run.to_dict()
    json.dumps(d, default=str)
    assert d["metrics"]["tool_calls"] == 2 and d["metrics"]["grounding"] == "PASS"
    assert d["tool_calls"][0]["tool_name"] == "compare_allowance"


def test_prompt_has_dates_and_rules():
    agent = build_orchestrator(MockProvider())
    assert f"previous_date={PREV}" in agent.system_prompt and f"current_date={CURR}" in agent.system_prompt
    assert "Never calculate" in ORCHESTRATOR_PROMPT
    assert len(agent.tools) == len(list_tools("orchestrator")) == 39   # 33 analytics + 4 regulatory + 2 factor tools


# --------------------------------------------------------------------------- scripted "small model"
def _msg(content="", tool_calls=None, reasoning=None):
    tcs = [NS(id=f"c{i}", type="function", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(tool_calls or [])]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tcs or None, reasoning_content=reasoning),
                          finish_reason="tool_calls" if tcs else "stop")],
              usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120))


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **params):
        self.requests.append(params)
        return self.script.pop(0)


def fake_provider(script, **kw):
    client = FakeClient(script)
    return OpenAICompatibleProvider(name="vllm", base_url="http://fake/v1", model="nemotron-test", api_key="x",
                                    client=client, **kw), client


def test_agent_recovers_from_bad_args_text_tool_calls_and_invented_numbers():
    xml_call = ("<tool_call>\n<function=calculate_portfolio_attribution>\n<parameter=previous_date>\n2026-07-31\n"
                "</parameter>\n<parameter=current_date>\n2026-08-31\n</parameter>\n</function>\n</tool_call>")
    script = [
        _msg(tool_calls=[("compare_allowance", json.dumps({"previous_date": "2025-06-30", "current_date": CURR}))],
             reasoning="I should compare the allowance."),                            # bad date
        _msg(tool_calls=[("compare_allowance", json.dumps({"previous_date": "July 2026", "current_date": "Aug 2026"}))]),  # fixed (repaired)
        _msg(content=xml_call),                                                        # tool call as text
        _msg(content="Allowance rose $19.24M; macro added $12.50M."),                  # $12.50M is invented
        _msg(content="Allowance rose $19.24M (19.4%); macroeconomic changes added $10.34M [calculate_portfolio_attribution]."),
    ]
    provider, client = fake_provider(script, enable_thinking=True)
    run = build_orchestrator(provider).run("Why did the allowance increase?")
    m = run.metrics()
    assert run.status == "answered"
    assert run.tools_used == ["compare_allowance", "compare_allowance", "calculate_portfolio_attribution"]
    assert m["tool_errors"] == 1 and m["recovered_after_error"] == 1 and m["repaired_arguments"] >= 1
    assert m["text_fallback_tool_calls"] == 1
    assert m["revisions"] == 1 and run.grounding["status"] == "PASS"
    assert "$10.34M" in run.answer
    # the error went back to the model with a recovery hint
    tool_msgs = [json.loads(x["content"]) for x in client.requests[1]["messages"] if x["role"] == "tool"]
    assert tool_msgs[0]["error"]["hint"]["available_dates"] == [PREV, CURR]
    # grounding feedback named the invented number
    assert "$12.50M" in client.requests[4]["messages"][-1]["content"]
    # request shape: tools sent, thinking flag passed through
    assert client.requests[0]["tools"] and client.requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert m["total_tokens"] == 5 * 120


def test_duplicate_calls_are_served_from_the_run():
    args = json.dumps({"previous_date": PREV, "current_date": CURR})
    script = [_msg(tool_calls=[("compare_allowance", args)]), _msg(tool_calls=[("compare_allowance", args)]),
              _msg(content="Done.")]
    provider, _ = fake_provider(script)
    run = build_orchestrator(provider).run("q")
    assert len(run.tool_records) == 1 and run.metrics()["duplicate_calls_served"] == 1


def test_specialist_cannot_call_tools_outside_its_list():
    script = [_msg(tool_calls=[("reconcile_allowance_change", json.dumps({"previous_date": PREV, "current_date": CURR}))]),
              _msg(content="I cannot reconcile.")]
    provider, client = fake_provider(script)
    agent = ToolAgent("macro_agent", "You are the macro agent.", provider)
    run = agent.run("Does it reconcile?")
    assert run.tool_records == []
    names = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert "reconcile_allowance_change" not in names and "calculate_variable_attribution" in names
    err = json.loads([x for x in client.requests[1]["messages"] if x["role"] == "tool"][0]["content"])
    assert err["error"]["error_type"] == "tool_not_allowed"


def test_step_limit_forces_an_answer():
    args = lambda d: json.dumps({"as_of_date": d})                                    # noqa: E731
    script = [_msg(tool_calls=[("get_allowance_summary", args(PREV))]), _msg(tool_calls=[("get_allowance_summary", args(CURR))]),
              _msg(content="Summary written from what I have.")]
    provider, client = fake_provider(script)
    run = build_orchestrator(provider, max_steps=2).run("q")
    assert run.status == "max_steps" and run.answer.startswith("Summary")
    assert "tools" not in client.requests[-1]                                           # final turn has no tools


def test_provider_failure_returns_error_run():
    class Boom:
        chat = NS(completions=NS(create=lambda **_: (_ for _ in ()).throw(ConnectionError("endpoint down"))))
    provider = OpenAICompatibleProvider("vllm", "http://x/v1", "m", "k", client=Boom())
    run = build_orchestrator(provider).run("q")
    assert run.status == "error" and "endpoint down" in run.answer


# --------------------------------------------------------------------------- parsing & helpers
@pytest.mark.parametrize("text,name,args", [
    ('<tool_call>{"name": "compare_allowance", "arguments": {"previous_date": "2026-07-31"}}</tool_call>',
     "compare_allowance", {"previous_date": "2026-07-31"}),
    ("<tool_call>\n<function=identify_top_contributors>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>",
     "identify_top_contributors", {"n": 5}),
    ('Let me check.\n```json\n{"name": "compare_allowance", "arguments": "{\\"current_date\\": \\"2026-08-31\\"}"}\n```',
     "compare_allowance", {"current_date": "2026-08-31"}),
    ('{"name": "compare_allowance", "parameters": {}}', "compare_allowance", {}),
])
def test_parse_text_tool_calls(text, name, args):
    calls, rest = parse_text_tool_calls(text, {"compare_allowance", "identify_top_contributors"})
    assert len(calls) == 1 and calls[0].name == name and calls[0].arguments == args
    assert "tool_call" not in rest and "{" not in rest


def test_text_parser_ignores_unknown_json():
    calls, rest = parse_text_tool_calls('```json\n{"name": "not_a_tool", "arguments": {}}\n```', {"compare_allowance"})
    assert calls == [] and "not_a_tool" in rest


def test_think_tags_are_split_into_reasoning():
    provider, _ = fake_provider([_msg(content="<think>plan the calls</think>Final answer.")])
    r = provider.chat([{"role": "user", "content": "hi"}])
    assert r.content == "Final answer." and r.reasoning == "plan the calls"


def test_nvidia_sends_reasoning_budget_vllm_does_not():
    nv = OpenAICompatibleProvider("nvidia", "u", "m", "k", client=FakeClient([]), enable_thinking=False, reasoning_budget=512)
    vl = OpenAICompatibleProvider("vllm", "u", "m", "k", client=FakeClient([]), enable_thinking=False, reasoning_budget=512)
    assert nv._extra_body() == {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_budget": 512}
    assert vl._extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_compact_respects_budget():
    big = {"rows": [{"id": i, "text": "x" * 50} for i in range(500)], "total": 1.23456789}
    small = compact(big, 2000)
    assert len(json.dumps(small)) <= 2000 and small["total"] == 1.234568
    assert small["rows"][-1]["_omitted_rows"] > 0


def test_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("RISKLENS_LLM_PROVIDER", "vllm")
    monkeypatch.setenv("RISKLENS_LLM_BASE_URL", "http://brev-host:8000/v1")
    s = llm_settings()
    assert s["provider"] == "vllm" and s["base_url"] == "http://brev-host:8000/v1"
    assert s["model"] == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"          # vLLM model id, not the API one
    assert get_provider().describe()["base_url"] == "http://brev-host:8000/v1"           # vLLM needs no key
    monkeypatch.setenv("RISKLENS_LLM_PROVIDER", "nvidia")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        get_provider()
    assert llm_settings()["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_grounding_rules():
    src = [{"change": 19238021.81, "pct": 19.42, "pd": 0.0521, "bps": 52.8, "note": "DSCR_WORSE(1.33->1.16)"}]
    ok = check_grounding("Up $19.24M (19.4%), PD 5.2%, +53 bps, DSCR 1.33 to 1.16, top 5, on 2026-08-31, CRE_PD_v4.", src)
    assert ok.status == "PASS" and ok.checked == 6
    bad = check_grounding("Up $19.24M; macro $12.50M; 33.3%.", src)
    assert bad.status == "FAIL" and [u["text"] for u in bad.ungrounded] == ["$12.50M", "33.3%"]
    assert check_grounding("No numbers here.", src).status == "NO_NUMBERS"
    assert [m.value for m in extract_numbers("-$2.66M and $1.2 billion")] == [-2_660_000.0, 1_200_000_000.0]


def test_tool_log_records_agent_calls(tmp_path):
    log = ToolCallLog(tmp_path / "calls.jsonl")
    build_orchestrator(MockProvider(), log=log).run("Does everything reconcile?")
    rows = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows and rows[0]["caller"] == "orchestrator" and rows[0]["tool_name"] == "reconcile_allowance_change"
