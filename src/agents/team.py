"""Phase 4: the RiskLens team. An orchestrator delegates to specialist agents.

Every agent is the same small model (one shared Nemotron base keeps the system within the 33B
limit) with its own role prompt and tool allow-list from ``config/agents.yaml``. The orchestrator
sees ``ask_<specialist>`` tools plus a few shared tools; a specialist answers its sub-question with
its own tools and returns a structured finding. Specialist tool results travel with the finding,
so the orchestrator's grounding check can trace every number in the final answer.

    from src.agents.team import build_team
    run = build_team(provider).run("Why did the allowance increase in August?")
"""
from __future__ import annotations

from typing import Any

from src.agents.base import AgentRun, ExtraTool, ToolAgent, reporting_context
from src.llm.provider import LLMProvider, get_provider, llm_settings
from src.tools import ToolCallLog
from src.utils import config

COMMON_RULES = """Rules
1. Every number you write must come from a tool result in this conversation. Never calculate, add up, estimate or invent figures. If you need a number, call a tool.
2. If a tool returns an error, read its hint, fix the arguments and call again.
3. If your tools cannot answer, say so plainly; another specialist may own that question.
4. You explain and prioritise; you never make lending decisions.
Reporting dates: {dates}. Unless told otherwise compare previous_date={previous} with current_date={current}, framework CECL. Portfolios: {portfolios}.
Answer format: one-sentence answer, then **Facts** (bullets with numbers and [tool_name]), then **Interpretation** (no new numbers). Money as $X.XXM, rates as X.X%."""

SPECIALISTS: dict[str, dict[str, str]] = {
    "portfolio_agent": {
        "title": "Portfolio Analyst",
        "role": "You analyse exposure: balances, new originations, paydowns and runoff, EAD changes and which segments drove the allowance change.",
    },
    "credit_agent": {
        "title": "Credit Analyst",
        "role": "You analyse borrower deterioration: PD, LGD and rating migration, the borrowers that added most ECL, and why (DSCR, LTV, ratings, delinquency).",
    },
    "macro_agent": {
        "title": "Macro Economist",
        "role": ("You analyse the economic scenarios: how the forecasts changed, which macro variable drove the allowance, scenario weights and sensitivities by portfolio. "
                 "For what-if questions about a named factor scenario (rates +200bp, unemployment 7%, CRE values -30%, Fed severely adverse) "
                 "call apply_scenario_factors and say it is a first-order sensitivity with illustrative factors, not an engine re-run."),
    },
    "model_risk_agent": {
        "title": "Model Risk Officer",
        "role": "You check governance: model versions and parameters, models awaiting validation, scenario weights and management overlays (qualitative adjustments).",
    },
    "reconciliation_agent": {
        "title": "Reconciliation Controller",
        "role": "You verify the numbers: reported vs explained change, residual, PASS/WARNING/FAIL status and whether the main driver is robust to the attribution order.",
    },
    "quant_agent": {
        "title": "Quant Analyst",
        "role": ("You answer ad-hoc questions the standard tools don't cover by writing pandas code. Call describe_datasets first, "
                 "then run_python_analysis with short, vectorised code that assigns `result`. If the code fails, read the "
                 "traceback and fix it. Quote only numbers that appear in the result."),
    },
    "regulatory_agent": {
        "title": "Regulatory Analyst",
        "role": ("You say what the supervisory guidance requires: call search_regulation, then answer from the passages and cite each one "
                 "as [citation], e.g. [IPS 2023 p.12]. Use get_regulatory_passage for the full text and find_guidance_citing for an ASC reference. "
                 "Say when guidance is EU / IFRS 9 rather than US CECL. Never compute or estimate allowance figures; if the question needs "
                 "numbers, say which specialist owns them."),
    },
    "reporting_agent": {
        "title": "Reporting Analyst",
        "role": ("You produce the Excel attribution pack with build_attribution_workbook and tell the user where it is, which "
                 "sheets it has and its headline figures. The workbook's totals are live formulas; mention that."),
    },
}

ORCHESTRATOR_TEAM_PROMPT = """You are the RiskLens lead investigator. You explain why the allowance for credit losses moved by delegating to specialist analysts and combining their findings.

Your team (call ask_<name> with a focused question):
{team}

Plan
1. Start with calculate_portfolio_attribution when the question is about why the allowance moved.
2. Delegate each part of the question to the specialist who owns it; ask 1-3 specialists, not all of them.
3. For "does it reconcile" questions, or before a final executive summary, ask the reconciliation_agent.
4. For custom slices of the data ask the quant_agent; for an Excel pack ask the reporting_agent.
5. For what the rules or guidance say (policy, audit, examiner or model-risk expectations) ask the regulatory_agent; for a what-if factor scenario ask the macro_agent.
Who owns what (route by topic; each specialist only has its own tools):
- model versions, model parameters or coefficients, overlays, models in validation, engine re-run -> model_risk_agent
- PD / LGD / rating migration, individual borrowers or loans -> credit_agent
- balances, originations, paydowns, EAD, segment attribution -> portfolio_agent
- macro forecasts, scenario weights, macro variables, factor what-ifs -> macro_agent
- reconciliation, residual, PASS / WARNING / FAIL -> reconciliation_agent
- guidance, regulation, audit or examiner expectations -> regulatory_agent

""" + COMMON_RULES + """
In **Facts**, tag each bullet with the specialist or tool it came from. End with **Next steps** when useful."""


def specialist_prompt(name: str) -> str:
    s = SPECIALISTS[name]
    return f"You are the RiskLens {s['title']}. {s['role']}\n\n" + COMMON_RULES.format(**reporting_context())


def _llm_opts(kwargs: dict[str, Any]) -> dict[str, Any]:
    llm = llm_settings()
    opts = {k: llm[k] for k in ("max_steps", "max_tool_calls", "result_chars") if k in llm}
    opts.update(kwargs)
    return opts


def build_specialist(name: str, provider: LLMProvider | None = None, log: ToolCallLog | None = None,
                     **kwargs: Any) -> ToolAgent:
    if name not in SPECIALISTS:
        raise ValueError(f"Unknown specialist {name!r}; choose from {sorted(SPECIALISTS)}")
    return ToolAgent(name, specialist_prompt(name), provider or get_provider(), log=log, **_llm_opts(kwargs))


class Team:
    """Orchestrator + specialists sharing one provider (one base model) and one tool-call log."""

    def __init__(self, provider: LLMProvider | None = None, log: ToolCallLog | None = None,
                 specialists: list[str] | None = None, direct_tools: list[str] | None = None, **kwargs: Any):
        self.provider = provider or get_provider()
        self.log = log if log is not None else ToolCallLog()
        self.kwargs = kwargs
        self.names = specialists or list(SPECIALISTS)
        self.sub_runs: list[AgentRun] = []
        cfg = config.load_yaml("agents").get("team", {})
        direct = direct_tools or cfg.get("orchestrator_direct_tools",
                                         ["get_reporting_context", "compare_allowance", "calculate_portfolio_attribution"])
        extra = {f"ask_{n}": ExtraTool(self._schema(n), self._delegate_fn(n)) for n in self.names}
        team = "\n".join(f"- ask_{n}: {SPECIALISTS[n]['title']}. {SPECIALISTS[n]['role'].split('.')[0]}." for n in self.names)
        prompt = ORCHESTRATOR_TEAM_PROMPT.format(team=team, **reporting_context())
        self.orchestrator = ToolAgent("orchestrator", prompt, self.provider, tools=direct, log=self.log,
                                      extra_tools=extra, **_llm_opts(kwargs))

    @staticmethod
    def _schema(name: str) -> dict[str, Any]:
        s = SPECIALISTS[name]
        return {"type": "function", "function": {
            "name": f"ask_{name}",
            "description": f"Delegate to the {s['title']}: {s['role'].split('.')[0]}. Returns its answer, facts and the tool results behind them.",
            "parameters": {"type": "object", "properties": {
                "question": {"type": "string", "description": "A focused question for this specialist, with any dates or portfolio named."}},
                "required": ["question"], "additionalProperties": False}}}

    def _delegate_fn(self, name: str):
        def delegate(question: str) -> dict[str, Any]:
            agent = build_specialist(name, self.provider, log=self.log, **self.kwargs)
            run = agent.run(question)
            self.sub_runs.append(run)
            return {"agent": name, "question": question, "status": run.status, "answer": run.answer,
                    "grounding": (run.grounding or {}).get("status"),
                    "tool_calls": [r.provenance() | {"status": r.status} for r in run.tool_records],
                    "evidence": {r.tool_name: r.result for r in run.tool_records if r.status == "ok"}}
        return delegate

    def run(self, question: str, history: list[dict[str, Any]] | None = None) -> AgentRun:
        self.sub_runs = []
        run = self.orchestrator.run(question, history)
        run.events.append({"type": "team", "specialists": [r.agent for r in self.sub_runs],
                           "sub_runs": [r.to_dict() for r in self.sub_runs]})
        return run


def build_team(provider: LLMProvider | None = None, log: ToolCallLog | None = None, **kwargs: Any) -> Team:
    return Team(provider, log, **kwargs)
