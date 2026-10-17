"""LLM backends behind one small interface.

`LLMProvider.chat(messages, tools)` returns a ChatResponse with text and/or
tool calls. The OpenAI-compatible implementation covers the NVIDIA API
catalog, self-hosted vLLM (Red Hat AI Inference Server) on Brev, NIM, and any
other OpenAI-style endpoint, so switching models is configuration only.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from risklens import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON string as produced by the model


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    model: str = ""


class LLMProvider:
    name = "base"
    model = ""

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> ChatResponse:
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}], tools=None).content


def llm_settings() -> dict:
    cfg = dict(config.load("agents")["llm"])
    provider = os.environ.get("RISKLENS_LLM_PROVIDER", cfg["provider"])
    cfg["provider"] = provider
    cfg["model"] = os.environ.get("RISKLENS_LLM_MODEL", cfg["models"].get(provider, ""))
    cfg["base_url"] = os.environ.get("RISKLENS_LLM_BASE_URL", cfg.get("base_urls", {}).get(provider))
    key = os.environ.get("RISKLENS_LLM_API_KEY")
    if not key and provider == "nvidia":
        key = os.environ.get("NVIDIA_API_KEY")
    if not key and provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
    cfg["api_key"] = key or ("EMPTY" if provider == "vllm" else None)
    return cfg


class OpenAICompatProvider(LLMProvider):
    """NVIDIA API catalog / vLLM / NIM / OpenAI-compatible chat completions with tool calling."""

    def __init__(self, cfg: dict):
        from openai import OpenAI
        if not cfg.get("api_key"):
            raise RuntimeError(f"No API key for provider '{cfg['provider']}'. Set NVIDIA_API_KEY or RISKLENS_LLM_API_KEY.")
        self.cfg = cfg
        self.name = cfg["provider"]
        self.model = cfg["model"]
        self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=cfg.get("timeout_s", 120))

    def chat(self, messages, tools=None, tool_choice="auto") -> ChatResponse:
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=self.cfg["temperature"],
                                      top_p=self.cfg["top_p"], max_tokens=self.cfg["max_tokens"])
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self.name in ("nvidia", "vllm"):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(self.cfg.get("enable_thinking"))}}
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some servers reject chat_template_kwargs; retry once without it.
            if "extra_body" in kwargs and ("chat_template_kwargs" in str(e) or "extra" in str(e).lower()):
                kwargs.pop("extra_body")
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id or f"call_{i}", tc.function.name, tc.function.arguments or "{}")
                 for i, tc in enumerate(msg.tool_calls or [])]
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return ChatResponse(content=msg.content or "", tool_calls=calls, usage=usage,
                            latency_s=time.perf_counter() - t0, model=self.model)


def get_provider(name: str | None = None) -> LLMProvider:
    if name:
        os.environ["RISKLENS_LLM_PROVIDER"] = name
    cfg = llm_settings()
    if cfg["provider"] == "mock":
        from risklens.llm.mock import MockProvider
        return MockProvider()
    if cfg["provider"] in ("nvidia", "vllm", "openai"):
        return OpenAICompatProvider(cfg)
    raise ValueError(f"Unknown LLM provider '{cfg['provider']}' (use mock, nvidia, vllm or openai)")


def dumps(obj) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
