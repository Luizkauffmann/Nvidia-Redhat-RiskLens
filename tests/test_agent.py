import json

from risklens.agents import grounding
from risklens.agents.agent import InvestigatorAgent, parse_text_tool_calls
from risklens.llm.mock import MockProvider
from risklens.llm.provider import ChatResponse, LLMProvider, ToolCall
from scripts.evaluate import cases


def test_mock_agent_answers_demo_questions(gt):
    agent = InvestigatorAgent(MockProvider())
    for q, tools, must in cases(gt):
        run = agent.run(q)
        assert set(run.tools_used()) & tools, q
        for s in must:
            assert s.lower() in run.answer.lower(), (q, s, run.answer[:400])
        assert run.grounding["ok"], (q, run.grounding)


def test_parse_text_tool_calls():
    a = parse_text_tool_calls('<tool_call>{"name": "compare_allowance", "arguments": {"previous_date": "2026-07-31"}}</tool_call>')
    assert a[0].name == "compare_allowance" and json.loads(a[0].arguments)["previous_date"] == "2026-07-31"
    b = parse_text_tool_calls("<tool_call><function=calculate_pd_migration><parameter=portfolio>CRE</parameter></function></tool_call>")
    assert b[0].name == "calculate_pd_migration" and json.loads(b[0].arguments) == {"portfolio": "CRE"}
    c = parse_text_tool_calls('```json\n{"name": "reconcile_allowance_change", "arguments": {}}\n```')
    assert c[0].name == "reconcile_allowance_change"


def test_grounding_flags_invented_numbers():
    res = [{"change": 19539279.74, "change_pct": 19.234961}]
    assert grounding.check("Allowance rose $19.54M (19.2%).", res).ok
    rep = grounding.check("Allowance rose $25.00M, mostly from a $3.1M model change.", res)
    assert not rep.ok and len(rep.unsupported) == 2


class ScriptedProvider(LLMProvider):
    """Simulates a small model: a bad call, a text-format call, then an ungrounded answer it must fix."""
    name, model = "scripted", "scripted"

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.turn += 1
        if self.turn == 1:
            return ChatResponse(tool_calls=[ToolCall("a", "compare_allowance", '{"previous_date": "2025-01-31"}')])
        if self.turn == 2:   # recovers using the hint, and writes the call as text
            return ChatResponse(content='<tool_call>{"name": "compare_allowance", "arguments": {"previous_date": "July 2026", "current_date": "August 2026"}}</tool_call>')
        if self.turn == 3:
            return ChatResponse(content="FACT: Allowance rose $19.54M, of which $7.77M from a model change.")
        return ChatResponse(content="FACT: Allowance rose $19.54M (19.23%).")


def test_recovery_text_calls_and_grounding_revision():
    run = InvestigatorAgent(ScriptedProvider()).run("Why did allowance increase?")
    m = run.metrics()
    assert m["tool_errors"] == 1 and m["recovered_errors"] == 1
    assert m["text_parsed_calls"] == 1 and m["repaired_calls"] >= 1
    assert m["grounding_revisions"] == 1 and run.grounding["ok"]
    assert "19.54" in run.answer and "7.77" not in run.answer


class SilentProvider(LLMProvider):
    name, model = "silent", "silent"

    def chat(self, messages, tools=None, tool_choice="auto"):
        return ChatResponse(content="")


def test_fallback_when_model_does_not_answer():
    run = InvestigatorAgent(SilentProvider()).run("Why did allowance increase?")
    assert run.metrics()["fallback_used"] == 1
    assert "deterministic summary" in run.answer and run.grounding["ok"]
