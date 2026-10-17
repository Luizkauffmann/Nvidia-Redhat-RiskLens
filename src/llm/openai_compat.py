"""OpenAI-compatible backend: NVIDIA API catalog, vLLM (Red Hat AI), NIM.

Small models sometimes emit a tool call as text instead of a structured
``tool_calls`` entry (e.g. when the server runs without a tool parser). Rather
than failing, we parse the common formats out of the text and flag the
response as ``raw_tool_format="text_fallback"`` so the evaluation can count it.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from src.llm.provider import LLMProvider, LLMResponse, ToolCall

_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_TOOL_TAG = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_XML_FN = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.\-]+)>(.*?)</parameter>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _coerce_param(text: str) -> Any:
    t = text.strip()
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return t


def _from_json_blob(blob: str) -> ToolCall | None:
    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    args = obj.get("arguments", obj.get("parameters", (obj.get("function") or {}).get("arguments", {})))
    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            pass
    return ToolCall(id=f"txt_{uuid.uuid4().hex[:8]}", name=str(name), arguments=args if args is not None else {})


def parse_text_tool_calls(text: str, known_tools: set[str] | None = None) -> tuple[list[ToolCall], str]:
    """Extract tool calls written as text. Returns (calls, remaining_text)."""
    if not text:
        return [], text
    calls: list[ToolCall] = []
    remaining = text
    for m in _XML_FN.finditer(text):                                   # qwen3_coder XML
        args = {p.group(1): _coerce_param(p.group(2)) for p in _XML_PARAM.finditer(m.group(2))}
        calls.append(ToolCall(id=f"txt_{uuid.uuid4().hex[:8]}", name=m.group(1), arguments=args))
    if not calls:
        for m in _TOOL_TAG.finditer(text):                             # <tool_call>{json}</tool_call>
            tc = _from_json_blob(m.group(1).strip())
            if tc:
                calls.append(tc)
    if not calls:
        for m in _FENCE.finditer(text):                                # ```json {...} ```
            tc = _from_json_blob(m.group(1))
            if tc and (known_tools is None or tc.name in known_tools):
                calls.append(tc)
    if not calls and text.strip().startswith("{"):                     # bare JSON
        tc = _from_json_blob(text.strip())
        if tc and (known_tools is None or tc.name in known_tools):
            calls.append(tc)
    if calls:
        remaining = _TOOL_TAG.sub("", _XML_FN.sub("", _FENCE.sub("", text))).strip()
        if remaining.startswith("{") and remaining.endswith("}"):
            remaining = ""
    return calls, remaining


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, name: str, base_url: str, model: str, api_key: str, temperature: float | None = None,
                 top_p: float | None = None, max_tokens: int = 4096, enable_thinking: bool | None = None,
                 reasoning_budget: int | None = None, timeout: float = 120, max_retries: int = 2,
                 client: Any = None):
        self.name, self.base_url, self.model = name, base_url, model
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.enable_thinking, self.reasoning_budget = enable_thinking, reasoning_budget
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.client = client

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "base_url": self.base_url,
                "enable_thinking": self.enable_thinking}

    def _extra_body(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        if self.reasoning_budget and self.name == "nvidia":
            extra["reasoning_budget"] = int(self.reasoning_budget)
        return extra

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             **kwargs: Any) -> LLMResponse:
        params: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.pop("tool_choice", "auto")
        for k in ("temperature", "top_p"):
            v = kwargs.pop(k, getattr(self, k))
            if v is not None:
                params[k] = v
        extra = {**self._extra_body(), **kwargs.pop("extra_body", {})}
        if extra:
            params["extra_body"] = extra
        params.update(kwargs)
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**params)
        latency = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        think = _THINK.findall(content)                       # server without a reasoning parser
        if think:
            reasoning = (reasoning or "") + "\n".join(think)
            content = _THINK.sub("", content).strip()
        calls: list[ToolCall] = []
        fmt = "none"
        for tc in (msg.tool_calls or []):
            raw = tc.function.arguments
            try:
                args: dict[str, Any] | str = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, ValueError):
                args = raw                                      # pass through; call_tool reports malformed JSON
            calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}", name=tc.function.name, arguments=args))
        if calls:
            fmt = "native"
        elif tools:
            known = {t["function"]["name"] for t in tools}
            calls, content = parse_text_tool_calls(content, known)
            fmt = "text_fallback" if calls else "none"
        usage = {}
        if getattr(resp, "usage", None):
            usage = {k: getattr(resp.usage, k, 0) or 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return LLMResponse(content=content, tool_calls=calls, reasoning=reasoning, finish_reason=choice.finish_reason,
                           usage=usage, raw_tool_format=fmt, latency_ms=round(latency, 1))
