"""Model-agnostic chat + tool-calling interface.

Every backend returns the same ``LLMResponse`` (text, tool calls, optional
reasoning), so agents never depend on a vendor SDK. Backends:

* ``nvidia``  NVIDIA API catalog (build.nvidia.com), Nemotron 3.5 Lightning
* ``vllm``    self-hosted vLLM / Red Hat AI on Brev (OpenAI-compatible)
* ``openai``  any other OpenAI-compatible endpoint (NIM, LM Studio, ...)
* ``mock``    deterministic scripted planner for tests and offline demos

Configuration: ``config/agents.yaml -> llm`` with environment overrides
(``RISKLENS_LLM_PROVIDER``, ``RISKLENS_LLM_BASE_URL``, ``RISKLENS_LLM_MODEL``,
``RISKLENS_LLM_API_KEY`` / ``NVIDIA_API_KEY``).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.utils import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | str          # str when the model produced invalid JSON


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw_tool_format: str = "native"           # native | text_fallback | none
    latency_ms: float = 0.0


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             **kwargs: Any) -> LLMResponse:
        """One model turn. ``messages`` use the OpenAI chat format."""

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model}


DEFAULTS: dict[str, dict[str, Any]] = {
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1",
               "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "api_key_env": "NVIDIA_API_KEY"},
    "vllm": {"base_url": "http://localhost:8000/v1",
             "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "api_key_env": "VLLM_API_KEY"},
    "openai": {"base_url": "http://localhost:8000/v1", "model": "", "api_key_env": "OPENAI_API_KEY"},
    "mock": {"model": "mock-planner"},
}


def llm_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge config/agents.yaml -> llm, provider defaults, env vars and explicit overrides."""
    cfg = dict(config.load_yaml("agents").get("llm", {}))
    provider = (overrides or {}).get("provider") or os.environ.get("RISKLENS_LLM_PROVIDER") or cfg.get("provider", "mock")
    merged = {**DEFAULTS.get(provider, {}), **{k: v for k, v in cfg.items() if v is not None and k != "provider"}}
    if cfg.get("provider") != provider:                    # config values were for another backend
        for k in ("base_url", "model"):
            merged[k] = DEFAULTS.get(provider, {}).get(k, merged.get(k))
    env = {"base_url": os.environ.get("RISKLENS_LLM_BASE_URL"), "model": os.environ.get("RISKLENS_LLM_MODEL")}
    merged.update({k: v for k, v in env.items() if v})
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    merged["provider"] = provider
    return merged


def get_provider(overrides: dict[str, Any] | None = None) -> LLMProvider:
    s = llm_settings(overrides)
    if s["provider"] == "mock":
        from src.llm.mock_provider import MockProvider
        return MockProvider()
    if s["provider"] in ("nvidia", "vllm", "openai"):
        from src.llm.openai_compat import OpenAICompatibleProvider
        key = (s.get("api_key") or os.environ.get("RISKLENS_LLM_API_KEY")
               or os.environ.get(s.get("api_key_env", ""), "") or ("EMPTY" if s["provider"] == "vllm" else ""))
        if not key:
            raise RuntimeError(f"No API key for provider '{s['provider']}'. Set {s.get('api_key_env')} or RISKLENS_LLM_API_KEY.")
        return OpenAICompatibleProvider(name=s["provider"], base_url=s["base_url"], model=s["model"], api_key=key,
                                        temperature=s.get("temperature"), top_p=s.get("top_p"),
                                        max_tokens=s.get("max_tokens", 4096),
                                        enable_thinking=s.get("enable_thinking"),
                                        reasoning_budget=s.get("reasoning_budget"),
                                        timeout=s.get("timeout_s", 120))
    raise ValueError(f"Unknown LLM provider '{s['provider']}' (use nvidia | vllm | openai | mock)")
