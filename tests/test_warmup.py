"""Warm-up / keep-warm: sends exactly the agent's prompt prefix, never while a question runs, and is a no-op for mock."""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from demo_app import warmup
from src.agents.orchestrator import build_orchestrator
from src.llm.mock_provider import MockProvider
from src.llm.provider import LLMProvider, LLMResponse


class Recorder(LLMProvider):
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return LLMResponse(content="OK", finish_reason="stop")

    def describe(self) -> dict[str, Any]:
        return {"provider": "vllm", "model": "recorder"}


@pytest.fixture(scope="module")
def ctx_for_agents(generated, attribution, tmp_path_factory):
    from src.tools import ToolContext, set_context
    from tests.conftest import CURR, PREV

    c = ToolContext(data_dir=generated[0], cache_dir=tmp_path_factory.mktemp("cache"), disk_cache=False)
    c._mem[f"attr_default_CECL_{PREV}_{CURR}"] = attribution
    set_context(c)
    yield c
    set_context(None)


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(warmup, "get_provider", lambda overrides=None: rec)
    return rec


def _wait(w: warmup.Warmer, state: str, timeout: float = 30) -> None:
    t0 = time.time()
    while w.state != state and time.time() - t0 < timeout:
        time.sleep(0.05)
    assert w.state == state, w.status()


def test_warm_up_sends_exactly_the_agent_prefix(recorder, ctx_for_agents):
    w = warmup.Warmer(keep_warm_every_s=3600)
    w.start({"provider": "vllm", "model": "m", "base_url": "http://localhost:11434/v1"}, "orchestrator")
    _wait(w, "ready")
    agent = build_orchestrator(MockProvider())
    call = recorder.calls[0]
    assert call["messages"][0] == {"role": "system", "content": agent.system_prompt}
    assert call["tools"] == agent.tools and call["max_tokens"] == 1
    assert w.status()["warm_s"] is not None


def test_mock_needs_no_warm_up(recorder):
    w = warmup.Warmer(keep_warm_every_s=3600)
    assert w.start({"provider": "mock"}, "team")["state"] == "not_needed"
    assert recorder.calls == []


def test_keep_warm_pings_only_when_idle_and_not_busy(recorder, ctx_for_agents):
    lock = threading.Lock()
    w = warmup.Warmer(lock, keep_warm_every_s=3600)
    w.start({"provider": "vllm", "model": "m"}, "orchestrator")
    _wait(w, "ready")
    n = len(recorder.calls)
    assert not w.maybe_ping(now=time.time())                                  # just warmed: no ping
    later = time.time() + warmup.IDLE_BEFORE_PING_S + 1
    with lock:                                                                # a question is running
        assert not w.maybe_ping(now=later)
    assert w.maybe_ping(now=later) and len(recorder.calls) == n + 1 and w.pings == 1
    w.touch()                                                                 # a real question counts as activity
    assert not w.maybe_ping(now=time.time())


def test_server_reports_warm_up_state(ctx_for_agents):
    from fastapi.testclient import TestClient

    from demo_app.server import app

    client = TestClient(app)
    assert client.post("/api/warmup", json={"provider": "mock", "mode": "team"}).json()["state"] == "not_needed"
    assert client.get("/api/warmup").json()["state"] == "not_needed"
