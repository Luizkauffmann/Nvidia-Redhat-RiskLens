"""Reward function for RiskLens investigations (used by the NeMo Gym resources server and local eval).

Input is framework-neutral: the final answer text, the tool calls the policy
made (name, arguments, output) and the task metadata from ``tasks.py``. The
NeMo Gym server converts a Responses API trajectory into this shape; the local
evaluator converts an ``AgentRun``. Both therefore get identical scores.

Components (each in [0, 1], weights in ``config/gym.yaml``):

* correctness    share of the task's answer checks passed (facts vs ground truth)
* grounding      share of numbers in the answer that trace to this episode's tool outputs
* tool_selection share of required any-of tool groups that were called
* call_validity  share of tool calls that executed without error
* efficiency     1 within the call budget, budget / calls beyond it

An answer with no successful tool call is multiplied by ``no_evidence_factor`` and an empty
answer scores 0.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.grounding import check_grounding, extract_numbers
from src.utils import config


@dataclass
class ToolCallTrace:
    name: str
    arguments: Any = None
    output: Any = None              # parsed JSON payload returned by the tool endpoint (or raw text)

    @property
    def ok(self) -> bool:
        out = self.output
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except (json.JSONDecodeError, ValueError):
                return False
        if not isinstance(out, dict):
            return False
        if "status" in out:
            return out["status"] == "ok"
        return "error" not in out

    @property
    def result(self) -> Any:
        out = self.output
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except (json.JSONDecodeError, ValueError):
                return None
        return out.get("result", out) if isinstance(out, dict) else None


@dataclass
class VerifyResult:
    reward: float
    components: dict[str, float]
    checks: list[dict[str, Any]] = field(default_factory=list)
    grounding: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    tool_errors: int = 0
    tools_used: list[str] = field(default_factory=list)
    missing_tool_groups: list[list[str]] = field(default_factory=list)
    has_answer: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gym_settings() -> dict[str, Any]:
    try:
        return config.load_yaml("gym")
    except FileNotFoundError:
        return {}


def _check(check: dict[str, Any], answer: str, rel_tol: float) -> tuple[bool, str]:
    kind = check["type"]
    if kind == "mentions":
        pats = check["any_of"]
        hit = next((p for p in pats if re.search(p, answer, re.I)), None)
        return hit is not None, (f"matched /{hit}/" if hit else "no pattern matched")
    if kind == "not_mentions":
        hit = next((p for p in check["none_of"] if re.search(p, answer, re.I)), None)
        return hit is None, (f"forbidden /{hit}/ found" if hit else "ok")
    if kind == "number":
        target = float(check["value"])
        want = check.get("kind", "money")
        for m in extract_numbers(answer):
            if m.kind != want:
                continue
            scale = 1e6 if re.search(r"\d\s?(?:M|million|mn)\b", m.text) else 1e9 if re.search(r"\d\s?(?:B|billion|bn)\b", m.text) else 1.0
            rounding = 0.5 * 10 ** (-m.decimals) * (scale if want == "money" else 1.0)
            tol = float(check.get("rel_tol", rel_tol))
            if abs(abs(m.value) - abs(target)) <= max(tol * abs(target), rounding):
                return True, f"found {m.text}"
        return False, f"no {want} figure near {target:,.2f}"
    raise ValueError(f"Unknown check type {kind!r}")


def score(answer: str, calls: list[ToolCallTrace], task: dict[str, Any],
          settings: dict[str, Any] | None = None) -> VerifyResult:
    """Score one episode. ``task`` holds ``checks``, ``required_tools``, ``call_budget`` (and the question)."""
    s = settings or gym_settings()
    weights = s.get("reward_weights", {"correctness": 0.55, "grounding": 0.2, "tool_selection": 0.1,
                                       "call_validity": 0.1, "efficiency": 0.05})
    rel_tol = float(s.get("number_rel_tol", 0.015))
    answer = (answer or "").strip()

    checks = []
    for c in task.get("checks", []):
        ok, why = _check(c, answer, rel_tol) if answer else (False, "no answer")
        checks.append({"desc": c.get("desc", c["type"]), "type": c["type"], "passed": ok, "detail": why})
    correctness = sum(c["passed"] for c in checks) / len(checks) if checks else 0.0

    sources = [c.result for c in calls if c.ok and c.result is not None] + [task.get("question", "")]
    rep = check_grounding(answer, sources) if answer else None
    if rep is None:
        grounding = 0.0
    elif rep.checked == 0:
        grounding = 1.0
    else:
        grounding = rep.grounded / rep.checked

    used = [c.name for c in calls]
    groups = task.get("required_tools") or []
    missing = [g for g in groups if not any(t in used for t in g)]
    tool_selection = 1.0 - len(missing) / len(groups) if groups else 1.0

    n = len(calls)
    errors = sum(not c.ok for c in calls)
    call_validity = (n - errors) / n if n else (1.0 if not groups else 0.0)
    budget = int(task.get("call_budget") or s.get("default_call_budget", 6))
    efficiency = 1.0 if n <= budget else budget / n

    comps = {"correctness": correctness, "grounding": grounding, "tool_selection": tool_selection,
             "call_validity": call_validity, "efficiency": efficiency}
    reward = sum(weights.get(k, 0.0) * v for k, v in comps.items())
    if groups and not any(c.ok for c in calls):
        # Every episode replays the same seeded story, so a policy could memorise the answers.
        # Without one successful tool call there is no evidence, whatever the text says.
        reward *= float(s.get("no_evidence_factor", 0.25))
    if not answer and s.get("require_final_answer", True):
        reward = 0.0
    return VerifyResult(reward=round(reward, 6), components={k: round(v, 6) for k, v in comps.items()},
                        checks=checks, grounding=(rep.to_dict() if rep else {}), tool_calls=n, tool_errors=errors,
                        tools_used=used, missing_tool_groups=missing, has_answer=bool(answer))


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

def from_responses_output(output: list[Any]) -> tuple[str, list[ToolCallTrace]]:
    """Extract (final answer, tool calls) from an OpenAI Responses API ``output`` list (NeMo Gym rollouts).

    Items may be pydantic objects or dicts. Tool outputs are matched to calls by ``call_id``.
    The final answer is the text of the last ``message`` item.
    """
    def g(item: Any, key: str, default: Any = None) -> Any:
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

    calls: dict[str, ToolCallTrace] = {}
    order: list[str] = []
    answer = ""
    for item in output or []:
        typ = g(item, "type")
        if typ == "function_call":
            cid = g(item, "call_id") or g(item, "id") or f"_{len(order)}"
            calls[cid] = ToolCallTrace(name=g(item, "name"), arguments=g(item, "arguments"))
            order.append(cid)
        elif typ == "function_call_output":
            cid = g(item, "call_id")
            if cid in calls:
                calls[cid].output = g(item, "output")
        elif typ == "message" and g(item, "role", "assistant") == "assistant":
            parts = g(item, "content") or []
            if isinstance(parts, str):
                text = parts
            else:
                text = "".join(g(p, "text", "") or "" for p in parts if g(p, "type") in ("output_text", "text", None))
            if text.strip():
                answer = text
    return answer, [calls[c] for c in order]


def from_agent_run(run: Any) -> tuple[str, list[ToolCallTrace]]:
    """Extract (final answer, tool calls) from a RiskLens ``AgentRun`` (in-process evaluation)."""
    calls = [ToolCallTrace(name=r.tool_name, arguments=r.arguments,
                           output={"status": r.status, "result": r.result, "error": r.error})
             for r in run.tool_records]
    return run.answer, calls
