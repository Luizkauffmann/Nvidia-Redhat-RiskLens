"""Serve a RiskLens LLM provider as an OpenAI-compatible endpoint (for NeMo Gym or any client).

Main use: run the full NeMo Gym pipeline offline with the deterministic mock planner,
so the environment, verifier and rollout collection can be tested without a GPU.

    python -m scripts.serve_policy --provider mock --port 8011
    gym env start --resources-server risklens_cecl --model-type vllm_model \
        --model mock-planner --model-url http://127.0.0.1:8011/v1 --model-api-key none

Endpoints: GET /v1/models, POST /v1/chat/completions (non-streaming, tool calls supported).
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request

from src.llm.provider import LLMProvider, get_provider


def _norm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten content-part lists to strings; the RiskLens providers expect plain text content."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        elif c is None:
            m["content"] = ""
        out.append(m)
    return out


def build_app(provider: LLMProvider) -> FastAPI:
    app = FastAPI(title="RiskLens policy endpoint")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": provider.model, "object": "model", "owned_by": "risklens"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict[str, Any]:
        body = await request.json()
        resp = provider.chat(_norm_messages(body.get("messages", [])), tools=body.get("tools"))
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content or None}
        if resp.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.name, "arguments": tc.arguments if isinstance(tc.arguments, str)
                                               else json.dumps(tc.arguments)}} for tc in resp.tool_calls]
        usage = resp.usage or {}
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", provider.model),
                "choices": [{"index": 0, "message": msg, "logprobs": None,
                             "finish_reason": "tool_calls" if resp.tool_calls else "stop"}],
                "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                          "completion_tokens": usage.get("completion_tokens", 0),
                          "total_tokens": usage.get("total_tokens", 0)}}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    uvicorn.run(build_app(get_provider({"provider": args.provider})), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
