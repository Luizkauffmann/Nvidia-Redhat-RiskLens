"""Tool-using agent loop shared by the orchestrator (Phase 3) and specialists (Phase 4).

    question -> [model turn -> tool calls -> call_tool -> results]* -> answer
             -> grounding check -> (one revision if a number is not in any tool result)

Reliability features for small models:
* only the agent's allow-listed tool schemas are sent
* tool errors come back as structured hints so the model can fix its call
* identical repeated calls are served from this run's records (no loops)
* tool results are compacted to a character budget
* a final answer with an unsupported number triggers one revision turn
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.grounding import check_grounding

# A "final answer" that is really a tool call the provider could not parse (e.g. a truncated
# <tool_call><function=...> block from a small model). Never shown to the user as an answer.
UNPARSED_TOOL_CALL = re.compile(r"<tool_call>|</?function[=>]|<parameter=|^\s*\{\s*\"name\"\s*:", re.M)
MAX_FORMAT_RETRIES = 1
from src.llm.provider import LLMProvider, LLMResponse
from src.tools import REGISTRY, ToolCallLog, ToolCallRecord, call_tool, tool_schemas
from src.tools.base import get_context


# --------------------------------------------------------------------------- compaction
def compact(obj: Any, max_chars: int = 6000) -> Any:
    """Shrink a JSON-able result to ~max_chars by trimming long lists (keeps the first rows)."""
    def size(o):
        return len(json.dumps(o, separators=(",", ":"), default=str))

    def trim(o, n):
        if isinstance(o, dict):
            return {k: trim(v, n) for k, v in o.items()}
        if isinstance(o, list):
            if len(o) > n:
                return [trim(v, n) for v in o[:n]] + [{"_omitted_rows": len(o) - n}]
            return [trim(v, n) for v in o]
        if isinstance(o, float):
            return round(o, 6)
        return o

    if size(obj) <= max_chars:
        return obj
    for n in (20, 10, 6, 4, 2, 1):
        t = trim(obj, n)
        if size(t) <= max_chars:
            return t
    s = json.dumps(trim(obj, 1), default=str)
    return {"_truncated_json": s[:max_chars]}


# --------------------------------------------------------------------------- run record
@dataclass
class AgentRun:
    run_id: str
    agent: str
    question: str
    answer: str = ""
    status: str = "running"                      # answered | max_steps | error
    provider: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_records: list[ToolCallRecord] = field(default_factory=list)
    grounding: dict[str, Any] | None = None
    revisions: int = 0
    format_retries: int = 0                      # malformed tool calls returned as answers (see UNPARSED_TOOL_CALL)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    duration_s: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def tools_used(self) -> list[str]:
        return [r.tool_name for r in self.tool_records]

    def metrics(self) -> dict[str, Any]:
        recs = self.tool_records
        llm = [e for e in self.events if e["type"] == "llm"]
        return {
            "tool_calls": len(recs), "tool_errors": sum(r.status == "error" for r in recs),
            "repaired_arguments": sum(r.normalized for r in recs),
            "recovered_after_error": _recoveries(recs),
            "text_fallback_tool_calls": sum(e.get("tool_format") == "text_fallback" for e in llm),
            "duplicate_calls_served": sum(e.get("duplicate", False) for e in self.events if e["type"] == "tool"),
            "llm_turns": len(llm), "revisions": self.revisions, "format_retries": self.format_retries,
            "grounding": (self.grounding or {}).get("status"), **self.usage, "duration_s": self.duration_s,
        }

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("tool_records", "messages")}
        d["tool_calls"] = [{**r.provenance(), "status": r.status, "duration_ms": r.duration_ms,
                            "arguments_requested": r.arguments, "error": r.error} for r in self.tool_records]
        d["metrics"] = self.metrics()
        return d


def _recoveries(recs: list[ToolCallRecord]) -> int:
    """Errors followed later by a successful call to the same tool."""
    n = 0
    for i, r in enumerate(recs):
        if r.status == "error" and any(x.tool_name == r.tool_name and x.status == "ok" for x in recs[i + 1:]):
            n += 1
    return n


# --------------------------------------------------------------------------- agent
class ToolAgent:
    def __init__(self, name: str, system_prompt: str, provider: LLMProvider, tools: list[str] | None = None,
                 max_steps: int = 8, max_tool_calls: int = 12, result_chars: int = 6000,
                 log: ToolCallLog | None = None, check_numbers: bool = True, max_revisions: int = 1,
                 extra_tools: dict[str, "ExtraTool"] | None = None):
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider
        allowed = tools if tools is not None else None
        schemas = tool_schemas(name if allowed is None else None)
        self.tools = [s for s in schemas if allowed is None or s["function"]["name"] in allowed]
        self.extra_tools = dict(extra_tools or {})          # e.g. delegation to specialist agents
        self.tools += [t.schema for t in self.extra_tools.values()]
        self.tool_names = {s["function"]["name"] for s in self.tools}
        self.max_steps, self.max_tool_calls, self.result_chars = max_steps, max_tool_calls, result_chars
        self.log = log if log is not None else ToolCallLog()
        self.check_numbers, self.max_revisions = check_numbers, max_revisions

    # ------------------------------------------------------------------
    def run(self, question: str, history: list[dict[str, Any]] | None = None) -> AgentRun:
        t0 = time.perf_counter()
        run = AgentRun(run_id=uuid.uuid4().hex[:10], agent=self.name, question=question,
                       provider=self.provider.describe())
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}, *(history or []),
                                      {"role": "user", "content": question}]
        seen: dict[str, ToolCallRecord] = {}
        try:
            for step in range(self.max_steps):
                budget_left = len(run.tool_records) < self.max_tool_calls
                resp = self._turn(run, msgs, tools=self.tools if budget_left else None)
                if resp.tool_calls and budget_left:
                    msgs.append(_assistant_tool_msg(resp))
                    for tc in resp.tool_calls:
                        msgs.append(self._execute(run, tc, seen))
                    continue
                answer = (resp.content or "").strip()
                if not answer:
                    msgs.append({"role": "user", "content": "Please give your final answer now, using only the tool results above."})
                    continue
                if UNPARSED_TOOL_CALL.search(answer):
                    if run.format_retries < MAX_FORMAT_RETRIES:
                        run.format_retries += 1
                        run.events.append({"type": "format_retry", "content_preview": answer[:200]})
                        msgs.append({"role": "assistant", "content": answer})
                        msgs.append({"role": "user", "content": (
                            "Your last message looks like a tool call written as text, and it was not executed. "
                            "Either call the tool using the tool-calling interface, or give your final answer in plain "
                            "sentences using only the tool results above.")})
                        continue
                    answer = UNPARSED_TOOL_CALL.split(answer)[0].strip() or (
                        "[RiskLens could not complete the investigation: the model kept returning a malformed tool call.]")
                if self.check_numbers:
                    rep = check_grounding(answer, [r.result for r in run.tool_records if r.result] + [question])
                    run.grounding = rep.to_dict()
                    run.events.append({"type": "grounding", "status": rep.status, "ungrounded": [u["text"] for u in rep.ungrounded]})
                    if rep.status == "FAIL" and run.revisions < self.max_revisions:
                        run.revisions += 1
                        msgs.append({"role": "assistant", "content": answer})
                        msgs.append({"role": "user", "content": (
                            "Grounding check failed: these numbers do not appear in any tool result: "
                            f"{', '.join(u['text'] for u in rep.ungrounded)}. Do not calculate figures yourself. "
                            "Call a tool that provides them, or rewrite the answer without them.")})
                        continue
                run.answer, run.status = answer, "answered"
                break
            else:
                resp = self._turn(run, msgs + [{"role": "user", "content": "Step limit reached. Answer now with the facts you have."}], tools=None)
                run.answer, run.status = (resp.content or "").strip(), "max_steps"
                if self.check_numbers and run.answer:
                    run.grounding = check_grounding(run.answer, [r.result for r in run.tool_records if r.result] + [question]).to_dict()
        except Exception as exc:                          # provider/network failure: return what we have
            run.status = "error"
            run.events.append({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            run.answer = run.answer or f"[RiskLens could not complete the investigation: {type(exc).__name__}: {exc}]"
        run.messages = msgs
        run.duration_s = round(time.perf_counter() - t0, 2)
        return run

    # ------------------------------------------------------------------
    def _turn(self, run: AgentRun, msgs: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        resp = self.provider.chat(msgs, tools=tools)
        for k, v in resp.usage.items():
            run.usage[k] = run.usage.get(k, 0) + int(v or 0)
        run.events.append({"type": "llm", "tool_calls": [tc.name for tc in resp.tool_calls],
                           "tool_format": resp.raw_tool_format, "content_preview": (resp.content or "")[:200],
                           "reasoning_chars": len(resp.reasoning or ""), "latency_ms": resp.latency_ms})
        return resp

    def _execute(self, run: AgentRun, tc, seen: dict[str, ToolCallRecord]) -> dict[str, Any]:
        if tc.name not in self.tool_names and tc.name in REGISTRY:
            payload = {"error": {"error_type": "tool_not_allowed",
                                 "message": f"{tc.name} is not available to {self.name}",
                                 "hint": {"available_tools": sorted(self.tool_names)}}}
            run.events.append({"type": "tool", "name": tc.name, "status": "not_allowed"})
            return {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(payload)}
        key = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}"
        dup = key in seen and seen[key].status == "ok"
        if dup:
            rec = seen[key]
        elif tc.name in self.extra_tools:
            rec = self.extra_tools[tc.name].call(tc.arguments, caller=self.name, log=self.log)
        else:
            rec = call_tool(tc.name, tc.arguments, caller=self.name, log=self.log)
        if not dup:
            run.tool_records.append(rec)
            seen[key] = rec
            seen[f"{rec.tool_name}:{json.dumps(rec.parameters, sort_keys=True, default=str)}"] = rec
        run.events.append({"type": "tool", "name": tc.name, "call_id": rec.call_id, "status": rec.status,
                           "parameters": rec.parameters, "duration_ms": rec.duration_ms, "duplicate": dup,
                           "error": rec.error})
        if rec.status == "ok":
            payload = {"call_id": rec.call_id, "result": compact(rec.result, self.result_chars)}
            if dup:
                payload["note"] = "Same call already made in this investigation; result repeated."
        else:
            payload = {"call_id": rec.call_id, "error": rec.error,
                       "instruction": "Fix the arguments using the hint and call the tool again, or choose another tool."}
        return {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(payload, default=str)}


@dataclass
class ExtraTool:
    """A non-registry tool an agent may call (delegation to another agent). ``fn(arguments) -> result dict``."""
    schema: dict[str, Any]
    fn: Any

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]

    def call(self, arguments: Any, caller: str | None = None, log: ToolCallLog | None = None) -> ToolCallRecord:
        import hashlib
        from datetime import datetime, timezone

        started = time.perf_counter()
        rec = ToolCallRecord(call_id=uuid.uuid4().hex[:12], tool_name=self.name, arguments={}, parameters={},
                             timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                             duration_ms=0.0, status="error", caller=caller)
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            rec.arguments = rec.parameters = dict(arguments or {})
            required = self.schema["function"]["parameters"].get("required", [])
            missing = [k for k in required if not rec.parameters.get(k)]
            if missing:
                raise ValueError(f"missing argument(s): {', '.join(missing)}")
            rec.result = self.fn(**rec.parameters)
            rec.result_sha256 = hashlib.sha256(json.dumps(rec.result, sort_keys=True, default=str).encode()).hexdigest()[:16]
            rec.status = "ok"
        except Exception as exc:                       # report to the calling model, don't crash its loop
            rec.error = {"error_type": "invalid_arguments" if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError))
                         else "execution_error", "message": f"{type(exc).__name__}: {exc}",
                         "hint": {"expected": self.schema["function"]["parameters"]}}
        rec.duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if log is not None:
            log.add(rec)
        return rec


def _assistant_tool_msg(resp: LLMResponse) -> dict[str, Any]:
    return {"role": "assistant", "content": resp.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.name,
                                         "arguments": tc.arguments if isinstance(tc.arguments, str)
                                         else json.dumps(tc.arguments)}}
                           for tc in resp.tool_calls]}


def reporting_context() -> dict[str, Any]:
    ctx = get_context()
    return {"dates": ctx.dates, "previous": ctx.dates[-2] if len(ctx.dates) > 1 else ctx.dates[0],
            "current": ctx.dates[-1], "portfolios": ctx.portfolios}
