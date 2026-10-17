"""Small-model failure modes seen with a local Nemotron 3 Nano 4B (2026-10-17): a malformed tool call returned as
the final answer, and team questions routed to the wrong specialist."""
from __future__ import annotations

from typing import Any

from src.agents.base import ToolAgent
from src.agents.team import ORCHESTRATOR_TEAM_PROMPT
from src.llm.provider import LLMProvider, LLMResponse

MALFORMED = "<tool_call>\n<function=get_reporting_context>"


class Scripted(LLMProvider):
    """Returns the scripted contents in order, then repeats the last one."""

    def __init__(self, *contents: str):
        self.contents, self.calls = list(contents), 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> LLMResponse:
        text = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return LLMResponse(content=text, finish_reason="stop")

    def describe(self) -> dict[str, Any]:
        return {"provider": "scripted", "model": "scripted"}


def _agent(provider: LLMProvider) -> ToolAgent:
    return ToolAgent("orchestrator", "You are a test agent.", provider, tools=[], check_numbers=False)


def test_malformed_tool_call_gets_one_retry_then_a_real_answer():
    run = _agent(Scripted(MALFORMED, "The CECL allowance is reported in the allowance summary.")).run("What is the allowance?")
    assert run.status == "answered" and run.format_retries == 1
    assert "<tool_call>" not in run.answer and run.answer.startswith("The CECL allowance")
    assert any(e["type"] == "format_retry" for e in run.events)
    assert run.metrics()["format_retries"] == 1


def test_persistent_malformed_tool_call_is_never_shown_as_the_answer():
    run = _agent(Scripted(MALFORMED)).run("What is the allowance?")
    assert "<tool_call>" not in run.answer and "<function=" not in run.answer
    assert "malformed tool call" in run.answer


def test_text_before_a_malformed_call_is_kept():
    run = _agent(Scripted("Allowance details follow. " + MALFORMED)).run("What is the allowance?")
    assert run.answer == "Allowance details follow."


def test_plain_answers_are_untouched():
    text = "The allowance rose; see [compare_allowance]. Function of macro and credit drivers."
    run = _agent(Scripted(text)).run("Why?")
    assert run.answer == text and run.format_retries == 0


def test_team_prompt_routes_model_parameters_to_model_risk():
    routing = ORCHESTRATOR_TEAM_PROMPT.split("Who owns what", 1)[1]
    line = next(l for l in routing.splitlines() if "model parameters" in l)
    assert line.strip().endswith("model_risk_agent")
    for agent in ("credit_agent", "portfolio_agent", "macro_agent", "reconciliation_agent", "regulatory_agent"):
        assert agent in routing
