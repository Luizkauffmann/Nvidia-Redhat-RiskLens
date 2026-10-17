# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym resources server for RiskLens CECL investigations.

* Tools: every RiskLens analytics tool is exposed as ``POST /<tool_name>``. The body is the
  tool's JSON arguments; the response is ``{"status": "ok", "call_id", "result"}`` or
  ``{"status": "error", "error": {"error_type", "message", "hint"}}``. Errors are normal tool
  outputs (HTTP 200) so the policy can read the hint and retry, which is what we want it to learn.
* Verify: scores the final answer against the task's ``verifier_metadata`` (derived from the
  seeded ground truth when the dataset was built) with ``src.evaluation.verifier.score``.
  ``reward`` is the weighted sum of the components in ``reward_components``.

The server never reads ``data/ground_truth``; expected facts arrive with each dataset row.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # make the RiskLens package importable from the server's own venv
    sys.path.insert(0, str(REPO_ROOT))

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    SimpleResourcesServer,
)

from src.agents.base import compact  # noqa: E402
from src.evaluation import verifier  # noqa: E402
from src.tools import REGISTRY, ToolCallLog, call_tool, get_context  # noqa: E402


class RiskLensResourcesServerConfig(BaseResourcesServerConfig):
    result_chars: int = 6000            # tool results are compacted to this size for small context windows
    tool_log_path: str | None = None    # optional JSONL provenance log of every tool call
    warm_cache: bool = True             # compute attribution/Shapley once at startup (slow first call otherwise)


class RiskLensVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskLensVerifyResponse(BaseMultiRewardVerifyResponse):
    verifier_metadata: Dict[str, Any] = Field(default_factory=dict)
    correctness: float = 0.0
    grounding: float = 0.0
    tool_selection: float = 0.0
    call_validity: float = 0.0
    efficiency: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    checks: list[Dict[str, Any]] = Field(default_factory=list)
    final_answer: str = ""


def tool_payload(rec: Any, result_chars: int) -> dict[str, Any]:
    if rec.status == "ok":
        return {"status": "ok", "call_id": rec.call_id, "tool": rec.tool_name, "result": compact(rec.result, result_chars)}
    return {"status": "error", "call_id": rec.call_id, "tool": rec.tool_name, "error": rec.error}


class RiskLensResourcesServer(SimpleResourcesServer):
    config: RiskLensResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._lock = asyncio.Lock()          # the tool context caches are not thread-safe
        log_path = Path(self.config.tool_log_path) if self.config.tool_log_path else None
        if log_path is not None and not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        self._log = ToolCallLog(log_path)
        if self.config.warm_cache:
            ctx = get_context()
            prev, curr = ctx.dates[-2], ctx.dates[-1]
            ctx.attribution(prev, curr)
            ctx.macro_shapley(prev, curr)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in REGISTRY:
            app.post(f"/{name}")(self._make_tool_endpoint(name))
        return app

    def _make_tool_endpoint(self, name: str):
        async def endpoint(request: Request) -> dict[str, Any]:
            raw = await request.body()
            try:
                args: Any = json.loads(raw) if raw.strip() else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                args = raw.decode(errors="replace")      # call_tool turns this into a malformed_arguments error
            async with self._lock:
                rec = await run_in_threadpool(call_tool, name, args, caller="nemo_gym", log=self._log)
            return tool_payload(rec, self.config.result_chars)

        endpoint.__name__ = f"tool_{name}"
        return endpoint

    async def verify(self, body: RiskLensVerifyRequest) -> RiskLensVerifyResponse:
        answer, calls = verifier.from_responses_output(body.response.output)
        meta = dict(body.verifier_metadata or {})
        res = verifier.score(answer, calls, meta)
        return RiskLensVerifyResponse(
            **body.model_dump(),
            reward=res.reward,
            reward_components=res.components,
            **res.components,
            tool_calls=res.tool_calls,
            tool_errors=res.tool_errors,
            checks=res.checks,
            final_answer=answer,
        )


if __name__ == "__main__":
    RiskLensResourcesServer.run_webserver()
