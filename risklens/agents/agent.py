"""Tool-calling CECL investigator agent.

One small model plans the investigation and calls deterministic tools. The
loop adds the safeguards a small model needs to be reliable:

* tool calls written as text are parsed (<tool_call> JSON, Qwen-style XML, fenced JSON)
* arguments are validated and repaired by the registry; errors come back with a hint
* duplicate calls are answered from cache instead of re-run
* large results are compacted before going back to the model
* the final answer is grounding-checked; unsupported figures trigger one revision
* if the model fails to finish, a deterministic summary of the tool results is returned
* every step is recorded with provenance and metrics
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

from risklens import config
from risklens.agents import grounding
from risklens.llm.provider import LLMProvider, ToolCall, dumps, get_provider
from risklens.tools import call_tool, tool_schemas
from risklens.tools.registry import TOOLS, _load_tools

SYSTEM_PROMPT = """You are RiskLens, a CECL allowance investigator for a bank's Credit Risk, Finance and Model Risk teams.
You explain why the credit-loss allowance changed between two reporting dates.

Rules:
1. Never calculate or invent financial numbers. Every figure you state must come from a tool result in this conversation.
   Copy figures from tool results; you may round them (e.g. 12631388.95 -> $12.63M) but do not add, subtract or divide them yourself.
2. Call tools to get facts. Prefer one or two well-chosen tools over many. If a tool returns an error, read its hint and retry with corrected arguments.
3. If the tools cannot answer, say what is missing. Do not guess.
4. Structure the answer with lines starting "FACT:" (from tools) and "INTERPRETATION:" (your reading of the facts).
5. You investigate and explain; you do not make lending or provisioning decisions. Humans decide.
6. Be concise: short sentences, money in $M with two decimals, percentages with one or two decimals.

Reporting dates available: {dates}. Unless the user says otherwise, compare previous_date={prev} with current_date={cur}.
Useful starting points: calculate_allowance_attribution (drivers), reconcile_allowance_change (does it add up),
identify_top_contributors (borrowers), calculate_variable_attribution (macro variables), compare_model_versions (model changes)."""


@dataclass
class Step:
    kind: str                      # model | tool | grounding | fallback
    detail: dict
    t: float = field(default_factory=time.time)


@dataclass
class AgentRun:
    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: "run_" + uuid.uuid4().hex[:8])
    model: str = ""
    elapsed_s: float = 0.0
    counters: dict = field(default_factory=lambda: {
        "model_turns": 0, "tool_calls": 0, "tool_ok": 0, "tool_errors": 0, "repaired_calls": 0,
        "text_parsed_calls": 0, "malformed_calls": 0, "duplicate_calls": 0, "recovered_errors": 0,
        "grounding_revisions": 0, "fallback_used": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def metrics(self) -> dict:
        c = dict(self.counters)
        n = c["tool_calls"] or 1
        c["tool_success_rate"] = c["tool_ok"] / n
        c["argument_validity_rate"] = 1 - (c["malformed_calls"] / n)
        c["recovery_rate"] = (c["recovered_errors"] / c["tool_errors"]) if c["tool_errors"] else None
        c["grounded"] = self.grounding.get("ok")
        c["elapsed_s"] = round(self.elapsed_s, 2)
        return c

    def tools_used(self) -> list[str]:
        return [p["tool_name"] for p in self.provenance if p["ok"]]


# ---------------------------------------------------------------------------
# Parsing tool calls that a model wrote as text instead of structured calls.

_TC_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_TC_XML = re.compile(r"<function=([\w\-]+)>(.*?)</function>", re.S)
_TC_PARAM = re.compile(r"<parameter=([\w\-]+)>\s*(.*?)\s*</parameter>", re.S)
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_text_tool_calls(text: str) -> list[ToolCall]:
    if not text:
        return []
    calls = []
    for raw in _TC_JSON.findall(text):
        try:
            obj = json.loads(raw)
            calls.append(ToolCall(f"text_{len(calls)}", obj.get("name", ""), json.dumps(obj.get("arguments", obj.get("parameters", {})))))
        except json.JSONDecodeError:
            calls.append(ToolCall(f"text_{len(calls)}", "__malformed__", raw))
    for name, body in _TC_XML.findall(text):
        args = {k: v for k, v in _TC_PARAM.findall(body)}
        calls.append(ToolCall(f"text_{len(calls)}", name, json.dumps(args)))
    if not calls:
        _load_tools()
        for raw in _FENCED.findall(text):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name") in TOOLS:
                calls.append(ToolCall(f"text_{len(calls)}", obj["name"], json.dumps(obj.get("arguments", {}))))
    return calls


def compact(obj, budget: int) -> str:
    """Shrink a tool result for the model: trim long lists, then drop the bulkiest keys."""
    s = dumps(obj)
    if len(s) <= budget:
        return s

    def trim(o, k):
        if isinstance(o, dict):
            return {key: trim(v, k) for key, v in o.items()}
        if isinstance(o, list):
            return [trim(v, k) for v in o[:k]] + ([f"... {len(o) - k} more"] if len(o) > k else [])
        return o
    for k in (10, 5, 3):
        s = dumps(trim(obj, k))
        if len(s) <= budget:
            return s
    if isinstance(obj, dict) and isinstance(obj.get("result"), dict):
        res = dict(obj["result"])
        while len(dumps({**obj, "result": res})) > budget and res:
            big = max(res, key=lambda key: len(dumps(res[key])))
            res[big] = "[omitted for length]"
            if all(v == "[omitted for length]" or len(dumps(v)) < 50 for v in res.values()):
                break
        return dumps({**obj, "result": trim(res, 3)})[:budget]
    return s[:budget]


class InvestigatorAgent:
    def __init__(self, provider: LLMProvider | None = None, tools: list[str] | None = None):
        self.provider = provider or get_provider()
        self.cfg = config.load("agents")["agent"]
        _load_tools()
        allow = config.load("agents")["agents"]["orchestrator"]
        self.tool_names = tools or (list(TOOLS) if allow == "all" else allow)
        self.schemas = tool_schemas(self.tool_names)

    def system_prompt(self) -> str:
        from risklens.tools.registry import available_dates
        d = available_dates()
        return SYSTEM_PROMPT.format(dates=", ".join(d), prev=d[0], cur=d[-1])

    def run(self, question: str, history: list[dict] | None = None) -> AgentRun:
        t0 = time.perf_counter()
        run = AgentRun(question=question, model=getattr(self.provider, "model", ""))
        messages = [{"role": "system", "content": self.system_prompt()}] + list(history or []) + [{"role": "user", "content": question}]
        seen: dict[str, dict] = {}
        last_error_tool: set[str] = set()
        revisions = 0
        final = None
        for _ in range(self.cfg["max_steps"]):
            resp = self.provider.chat(messages, tools=self.schemas)
            run.counters["model_turns"] += 1
            for k_src, k_dst in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens")):
                run.counters[k_dst] += int(resp.usage.get(k_src, 0) or 0)
            calls = list(resp.tool_calls)
            if not calls:
                calls = parse_text_tool_calls(resp.content)
                run.counters["text_parsed_calls"] += len(calls)
            run.steps.append(Step("model", {"content": resp.content[:2000], "tool_calls": [c.__dict__ for c in calls],
                                            "latency_s": round(resp.latency_s, 2)}))
            if calls and run.counters["tool_calls"] < self.cfg["max_tool_calls"]:
                calls = calls[: self.cfg["max_parallel_calls"]]
                messages.append({"role": "assistant", "content": resp.content or "",
                                 "tool_calls": [{"id": c.id, "type": "function",
                                                 "function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
                for c in calls:
                    msg = self._execute(c, run, seen, last_error_tool)
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": msg})
                continue
            final = resp.content.strip()
            if not final:
                break
            rep = grounding.check(final, run.tool_results)
            run.steps.append(Step("grounding", rep.to_dict()))
            if rep.ok or revisions >= self.cfg["grounding_revisions"] or not run.tool_results:
                run.grounding = rep.to_dict()
                break
            revisions += 1
            run.counters["grounding_revisions"] += 1
            messages.append({"role": "assistant", "content": final})
            messages.append({"role": "user", "content":
                             "These figures do not appear in any tool result: " + ", ".join(rep.unsupported[:10]) +
                             ". Rewrite the answer using only figures copied from tool results (call a tool if you need a figure)."})
            final = None
        if not final:
            final = self._fallback(run)
        run.answer = final
        if not run.grounding:
            run.grounding = grounding.check(final, run.tool_results).to_dict()
        run.elapsed_s = time.perf_counter() - t0
        return run

    def _execute(self, c: ToolCall, run: AgentRun, seen: dict, last_error_tool: set) -> str:
        key = f"{c.name}:{c.arguments}"
        if key in seen:
            run.counters["duplicate_calls"] += 1
            run.steps.append(Step("tool", {"name": c.name, "duplicate": True}))
            return compact({**seen[key], "note": "duplicate call; same result as before"}, self.cfg["result_chars"])
        if c.name == "__malformed__":
            run.counters["malformed_calls"] += 1
            run.counters["tool_calls"] += 1
            run.counters["tool_errors"] += 1
            return dumps({"ok": False, "error": "Could not parse your tool call as JSON.",
                          "hint": 'Use {"name": "<tool>", "arguments": {...}} with valid JSON.'})
        try:
            json.loads(c.arguments or "{}")
        except json.JSONDecodeError:
            run.counters["malformed_calls"] += 1
        r = call_tool(c.name, c.arguments)
        run.counters["tool_calls"] += 1
        if r.repairs:
            run.counters["repaired_calls"] += 1
        if r.ok:
            run.counters["tool_ok"] += 1
            if c.name in last_error_tool:
                run.counters["recovered_errors"] += 1
                last_error_tool.discard(c.name)
            run.tool_results.append(r.result)
            seen[key] = r.to_message()
        else:
            run.counters["tool_errors"] += 1
            last_error_tool.add(c.name)
        run.provenance.append(r.provenance())
        run.steps.append(Step("tool", {"name": c.name, "arguments": r.arguments, "ok": r.ok, "error": r.error,
                                       "repairs": r.repairs, "duration_ms": r.duration_ms, "call_id": r.call_id}))
        return compact(r.to_message(), self.cfg["result_chars"])

    def _fallback(self, run: AgentRun) -> str:
        """Model did not finish: return the tool facts deterministically rather than nothing."""
        run.counters["fallback_used"] += 1
        from risklens.llm.mock import _render
        by_tool: dict[str, list] = {}
        for p, res in zip([p for p in run.provenance if p["ok"]], run.tool_results):
            by_tool.setdefault(p["tool_name"], []).append(res)
        if not by_tool:
            for name in ("compare_allowance", "calculate_allowance_attribution", "reconcile_allowance_change"):
                r = call_tool(name, {})
                if r.ok:
                    by_tool.setdefault(name, []).append(r.result)
                    run.tool_results.append(r.result)
                    run.provenance.append(r.provenance())
        run.steps.append(Step("fallback", {"tools": list(by_tool)}))
        return "(Model did not complete; deterministic summary of tool results.)\n" + _render(by_tool)
