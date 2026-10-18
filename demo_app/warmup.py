"""Warm-up and keep-warm for a local model server (Ollama / vLLM) behind the demo app.

On an M4 the agent's fixed prompt start (system prompt + tool schemas, ~12.7k tokens for the single agent) takes
60-90 s to read cold, but ~11 s per new question once the server has it cached, because Ollama reuses a shared
prompt prefix. Two things throw that cache away in a demo: the first question after startup, and Ollama unloading
an idle model after 5 minutes. The Warmer sends that exact prompt start once (max_tokens=1, through the same
provider the agents use, so the tokens match), then pings it every few minutes while the app is idle.

Ollama keeps one cache per loaded model here, so in team mode only the orchestrator's prefix stays warm; the
specialists' prompts (1-5k tokens) are smaller and cheaper to read cold.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from src.agents.orchestrator import build_orchestrator
from src.agents.team import build_team
from src.llm.provider import get_provider

KEEP_WARM_EVERY_S = 180      # ping interval; Ollama's default keep_alive is 5 minutes
IDLE_BEFORE_PING_S = 150     # a real question in the last 150 s already kept the model loaded
WARM_QUESTION = "Warm-up request: reply OK."


def agent_prefix(mode: str, provider) -> tuple[str, list[dict[str, Any]]]:
    """The system prompt and tool schemas the agent for `mode` sends on every turn."""
    agent = build_team(provider).orchestrator if mode == "team" else build_orchestrator(provider)
    return agent.system_prompt, agent.tools


def warm_now(overrides: dict[str, Any], mode: str) -> float:
    """Send the prefix once and wait; returns seconds taken (used by `demo_app.bench --warmup`)."""
    t0 = time.time()
    provider = get_provider({k: v for k, v in overrides.items() if v is not None})
    system, tools = agent_prefix(mode, provider)
    provider.chat([{"role": "system", "content": system}, {"role": "user", "content": WARM_QUESTION}],
                  tools=tools, max_tokens=1)
    return round(time.time() - t0, 1)


class Warmer:
    def __init__(self, lock: threading.Lock | None = None, keep_warm_every_s: float = KEEP_WARM_EVERY_S):
        self.lock = lock or threading.Lock()   # the app's agent lock: never ping while a question is running
        self.every = keep_warm_every_s
        self.target: dict[str, Any] | None = None
        self.state = "idle"                     # idle | not_needed | warming | ready | error
        self.error: str | None = None
        self.warm_s: float | None = None        # duration of the last full warm-up
        self.last_activity = 0.0                # last warm-up, ping or real question
        self.pings = 0
        self._loop: threading.Thread | None = None
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ public
    def start(self, overrides: dict[str, Any], mode: str) -> dict[str, Any]:
        """Warm the prefix for this model + mode in the background (no-op for the mock planner)."""
        overrides = {k: v for k, v in overrides.items() if v is not None}
        target = {"overrides": overrides, "mode": mode}
        with self._guard:
            if target == self.target and self.state in ("warming", "ready"):
                return self.status()
            self.target, self.error, self.warm_s = target, None, None
            if overrides.get("provider", get_provider(overrides).describe()["provider"]) == "mock":
                self.state = "not_needed"
                return self.status()
            self.state = "warming"
        threading.Thread(target=self._warm, args=(target,), daemon=True).start()
        self._ensure_loop()
        return self.status()

    def touch(self) -> None:
        """A real question just ran: the model is loaded and its cache holds the prefix."""
        self.last_activity = time.time()

    def status(self) -> dict[str, Any]:
        t = self.target or {}
        return {"state": self.state, "mode": t.get("mode"), "model": (t.get("overrides") or {}).get("model"),
                "base_url": (t.get("overrides") or {}).get("base_url"), "warm_s": self.warm_s, "error": self.error,
                "pings": self.pings, "seconds_since_activity": round(time.time() - self.last_activity) if self.last_activity else None,
                "keep_warm_every_s": self.every}

    def maybe_ping(self, now: float | None = None) -> bool:
        """Re-send the prefix if the model has been idle long enough that it might unload. Returns True if pinged."""
        now = now or time.time()
        target = self.target
        if self.state != "ready" or target is None or now - self.last_activity < IDLE_BEFORE_PING_S:
            return False
        if not self.lock.acquire(blocking=False):      # a question is running: it keeps the model warm itself
            return False
        try:
            self._send(target)
            self.pings += 1
            self.last_activity = time.time()
            return True
        except Exception as exc:                        # server gone: show it, retry on the next tick
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            return False
        finally:
            self.lock.release()

    # ------------------------------------------------------------------ internals
    def _send(self, target: dict[str, Any]) -> None:
        warm_now(target["overrides"], target["mode"])

    def _warm(self, target: dict[str, Any]) -> None:
        t0 = time.time()
        try:
            with self.lock:                             # queue behind a running question rather than compete with it
                self._send(target)
            if target == self.target:
                self.state, self.warm_s, self.last_activity = "ready", round(time.time() - t0, 1), time.time()
        except Exception as exc:
            if target == self.target:
                self.state, self.error = "error", f"{type(exc).__name__}: {exc}"

    def _ensure_loop(self) -> None:
        if self._loop and self._loop.is_alive():
            return

        def loop() -> None:
            while True:
                time.sleep(self.every)
                if self.state == "error" and self.target:   # e.g. Ollama restarted: try a fresh warm-up
                    self.state = "warming"
                    self._warm(self.target)
                else:
                    self.maybe_ping()

        self._loop = threading.Thread(target=loop, daemon=True)
        self._loop.start()
