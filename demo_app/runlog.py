"""Run log for model comparisons: what ran, how long it took, and how good the answer was.

Every Ask (from the app or `python -m demo_app.bench`) is appended to
`<outputs>/demo_app_runs.jsonl` with the model, timings, token counts, tool calls, grounding and,
when the question is in the evaluation bank (`src/evaluation/tasks.py`), the Gym verifier's reward.

Evaluation only: scoring reads ground truth AFTER the agent has answered and never feeds anything
back to the agent, as CLAUDE.md requires.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.orchestrator import build_orchestrator
from src.agents.team import build_team
from src.evaluation.tasks import Task, _mention, _money, build_tasks
from src.evaluation.verifier import from_agent_run, score
from src.llm.provider import get_provider
from src.tools import ToolCallLog
from src.utils import config

BENCH_TASKS = ["q02_current_allowance", "q05_drivers", "q07_top_borrowers", "q13_model_change", "q15_reconcile"]
# The same questions as text, so the app and CI don't need ground truth just to list them.
# tests/test_demo_app.py checks they still match the bank.
BENCH_QUESTIONS = [
    "What was the CECL allowance at the end of August 2026?",
    "Why did our allowance increase in August? What were the major drivers?",
    "Which borrowers contributed most to the increase? Who should I investigate first?",
    "Could this increase just be because we changed models?",
    "Does the attribution reconcile to the total allowance change?",
]


log = logging.getLogger(__name__)
_TASKS: dict[str, Any] | None = None   # evaluation bank by normalised question, loaded once ground truth exists
_KB_TASKS: dict[str, Any] | None = None  # knowledge-base suite, cached once the KB answered

# Knowledge-base suite (regulatory agent + scenario factors). Kept separate from the five core questions so
# the committed core baseline stays comparable. Needs the local KB (python -m scripts.load_kb).
KB_QUESTIONS = {
    "kb_reg_reversion": "What does the guidance say about reverting to historical loss information after the forecast period?",
    "kb_reg_model_risk": "Which supervisory guidance now governs model risk management, and does it replace SR 11-7?",
    "kb_factor_unemployment": "What would the allowance be if unemployment hits 7%?",
}


def runs_path() -> Path:
    return config.path("outputs") / "demo_app_runs.jsonl"


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def _kb_tasks() -> list[Task]:
    """Checks need no ground truth: the citation must name the current document, and the what-if figure is
    the deterministic tool's own result (computed here, at scoring time only)."""
    from src.tools import call_tool

    q = KB_QUESTIONS
    tasks = [
        Task("kb_reg_reversion", q["kb_reg_reversion"], "regulatory",
             [_mention("cites the 2023 interagency policy statement", r"IPS 2023 p\.\s?\d+"),
              _mention("historical loss information beyond the forecast period", r"historical loss"),
              _mention("no specific reversion technique or period is required",
                       r"(not|no)\b[^.]{0,60}(specific|particular)[^.]{0,40}(technique|method|period)")],
             [["search_regulation"]], 3),
        Task("kb_reg_model_risk", q["kb_reg_model_risk"], "regulatory",
             [_mention("names SR 26-2", r"SR\s?26-2"), _mention("says it supersedes SR 11-7", r"supersed|replac")],
             [["search_regulation"]], 3),
    ]
    rec = call_tool("apply_scenario_factors", {"factor_scenario": "unemployment_7pct", "as_of_date": "2026-08-31"},
                    caller="evaluation", log=None)
    if rec.status == "ok":
        tasks.append(Task("kb_factor_unemployment", q["kb_factor_unemployment"], "scenario_factors",
                          [_money(rec.result["scenario_allowance"], "what-if allowance under the 7% unemployment factors"),
                           _mention("says the factors are a sensitivity / illustrative",
                                    r"sensitivit|illustrative|first[- ]order|not an engine")],
                          [["apply_scenario_factors"]], 3))
    return tasks


def _tasks() -> dict[str, Any]:
    """Evaluation bank keyed by question: core questions (need ground truth) + knowledge-base questions (need the
    local KB). A missing source is not cached: those runs are logged unscored and the next call retries."""
    global _TASKS, _KB_TASKS
    if _TASKS is None:
        try:
            _TASKS = {_norm(t.question): t for t in build_tasks()}
        except Exception as exc:  # e.g. data not generated yet: don't cache the miss
            log.warning("No evaluation ground truth (%s); runs are logged without a quality score. "
                        "Run python -m scripts.generate_demo_data to enable scoring.", exc)
    if _KB_TASKS is None:
        try:
            kb = _kb_tasks()
            if len(kb) == len(KB_QUESTIONS):          # the factor question needs the KB; retry until it is built
                _KB_TASKS = {_norm(t.question): t for t in kb}
        except Exception as exc:
            log.warning("Knowledge-base questions unscored (%s). Run python -m scripts.load_kb.", exc)
    return {**(_TASKS or {}), **(_KB_TASKS or {})}


def bench_questions(suite: str = "core") -> list[str]:
    """core = the five allowance questions (committed baseline); kb = regulatory + factor questions; all = both."""
    return {"core": list(BENCH_QUESTIONS), "kb": list(KB_QUESTIONS.values()),
            "all": list(BENCH_QUESTIONS) + list(KB_QUESTIONS.values())}[suite]


def quality(question: str, run: Any) -> dict[str, Any] | None:
    task = _tasks().get(_norm(question))
    if task is None:
        return None
    answer, calls = from_agent_run(run)
    v = score(answer, calls, task.to_dict())
    return {"task_id": task.task_id, "reward": round(v.reward, 3),
            "components": {k: round(x, 3) for k, x in v.components.items()},
            "checks_passed": sum(c["passed"] for c in v.checks), "checks_total": len(v.checks),
            "missing_tool_groups": v.missing_tool_groups}


def ask(question: str, mode: str = "team", overrides: dict[str, Any] | None = None, label: str | None = None,
        source: str = "app", record: bool = True) -> tuple[Any, dict[str, Any]]:
    """Run one question through the agents and (unless record=False) append a run record. Returns (AgentRun, record)."""
    provider = get_provider({k: v for k, v in (overrides or {}).items() if v is not None})
    log = ToolCallLog(config.path("outputs") / "demo_app_tool_calls.jsonl")
    build = build_team if mode == "team" else build_orchestrator
    run = build(provider, log=log).run(question)
    d = run.to_dict()
    llm_s = sum(e.get("latency_ms", 0) for e in run.events if e["type"] == "llm") / 1000
    tool_s = sum(c.get("duration_ms") or 0 for c in d["tool_calls"]) / 1000
    usage = run.usage or {}
    desc = provider.describe()
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
        "label": label or desc.get("model"), "provider": desc.get("provider"), "model": desc.get("model"),
        "base_url": desc.get("base_url"), "thinking": desc.get("enable_thinking"), "mode": mode,
        "question": question, "status": run.status, "duration_s": round(run.duration_s, 2),
        "llm_s": round(llm_s, 2), "tool_s": round(tool_s, 2), "llm_turns": d["metrics"]["llm_turns"],
        "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_per_s": round(usage.get("completion_tokens", 0) / llm_s, 1) if llm_s else None,
        "tool_calls": len(d["tool_calls"]), "tool_errors": d["metrics"]["tool_errors"],
        "tools_used": run.tools_used, "grounding": (run.grounding or {}).get("status"),
        "quality": quality(question, run), "answer": run.answer,
    }
    if record:
        path = runs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    return run, rec


def load_runs() -> list[dict[str, Any]]:
    path = runs_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per label: speed and quality side by side."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault(r["label"], []).append(r)
    rows = []
    for label, rs in groups.items():
        scored = [r["quality"]["reward"] for r in rs if r.get("quality")]
        tps = [r["tokens_per_s"] for r in rs if r.get("tokens_per_s")]
        rows.append({
            "label": label, "model": rs[-1]["model"], "provider": rs[-1]["provider"], "runs": len(rs),
            "median_duration_s": round(statistics.median(r["duration_s"] for r in rs), 1),
            "median_llm_s": round(statistics.median(r["llm_s"] for r in rs), 1),
            "mean_llm_turns": round(statistics.mean(r["llm_turns"] for r in rs), 1),
            "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in rs), 1),
            "median_tokens_per_s": round(statistics.median(tps), 1) if tps else None,
            # NO_NUMBERS (nothing to check) is not a failure; only FAIL means an unsupported figure.
            "grounded_pct": round(100 * sum(r["grounding"] != "FAIL" for r in rs) / len(rs)),
            "scored_runs": len(scored), "mean_reward": round(statistics.mean(scored), 3) if scored else None,
        })
    return rows
