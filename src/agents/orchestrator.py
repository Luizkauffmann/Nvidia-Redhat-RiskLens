"""Phase 3: a single orchestrating agent with access to all tools.

Phase 4 splits the work across specialists (Portfolio, Credit, Macro, Model
Risk, Reconciliation) that reuse ``ToolAgent`` with narrower tool lists.
"""
from __future__ import annotations

from typing import Any

from src.agents.base import AgentRun, ToolAgent, reporting_context
from src.llm.provider import LLMProvider, get_provider, llm_settings
from src.tools import ToolCallLog

ORCHESTRATOR_PROMPT = """You are RiskLens, a credit-risk analyst who explains why the allowance for credit losses (CECL / IFRS 9) moved. You investigate with deterministic tools.

Context
- Reporting dates: {dates}. Unless the user says otherwise compare previous_date={previous} with current_date={current}, framework CECL.
- Portfolios: {portfolios}.

Rules
1. Every number you write must come from a tool result in this conversation. Never calculate, add up, estimate or invent figures; the tools already provide changes, shares and percentages. If you need a number, call a tool.
2. Use few, targeted calls. Good first calls: calculate_portfolio_attribution (why did it change / drivers), calculate_segment_attribution (which portfolio), identify_top_contributors (which borrowers), calculate_variable_attribution (which macro variable), compare_model_versions (model changes), compare_scenario_weights (weights), reconcile_allowance_change (does it reconcile), compare_borrower_periods (one borrower in depth).
3. If a tool returns an error, read its hint, fix the arguments and call again.
4. If the tools cannot answer, say so. You explain and prioritise; you never make lending decisions.

Answer format, concise, for a risk manager:
<one-sentence direct answer>
**Facts**
- <fact with numbers> [tool_name]
**Interpretation**
- <your reading of the facts, no new numbers>
**Next steps**
- <what to investigate next> (optional)
Write money as $X.XXM and rates as X.X%."""


def build_orchestrator(provider: LLMProvider | None = None, log: ToolCallLog | None = None,
                       **kwargs: Any) -> ToolAgent:
    prompt = ORCHESTRATOR_PROMPT.format(**reporting_context())
    llm = llm_settings()
    opts = {k: llm[k] for k in ("max_steps", "max_tool_calls", "result_chars") if k in llm}
    opts.update(kwargs)
    return ToolAgent("orchestrator", prompt, provider or get_provider(), log=log, **opts)


def ask(question: str, provider: LLMProvider | None = None, **kwargs: Any) -> AgentRun:
    """One-shot investigation with the orchestrator."""
    return build_orchestrator(provider, **kwargs).run(question)
