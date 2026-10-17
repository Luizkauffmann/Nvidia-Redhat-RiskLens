"""Smoke-test an LLM endpoint before the demo: connectivity, plain chat, one tool call.

    NVIDIA_API_KEY=nvapi-... python -m scripts.check_llm --provider nvidia
    python -m scripts.check_llm --provider vllm --base-url http://<brev-host>:8000/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.provider import get_provider  # noqa: E402
from src.tools import tool_schemas  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="nvidia", choices=["nvidia", "vllm", "openai", "mock"])
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    a = ap.parse_args()
    p = get_provider({"provider": a.provider, "model": a.model, "base_url": a.base_url})
    print("endpoint:", json.dumps(p.describe()))
    ok = True
    try:
        r = p.chat([{"role": "user", "content": "Reply with exactly: RiskLens ready"}])
        print(f"[1] chat           OK  {r.latency_ms:.0f} ms  -> {r.content.strip()[:80]!r}  (reasoning {len(r.reasoning or '')} chars)")
    except Exception as exc:
        print(f"[1] chat           FAIL {type(exc).__name__}: {exc}")
        return 1
    tools = [s for s in tool_schemas() if s["function"]["name"] == "compare_allowance"]
    r = p.chat([{"role": "system", "content": "Use the tools. Dates: 2026-07-31 and 2026-08-31."},
                {"role": "user", "content": "How much did the CECL allowance change from July to August 2026?"}], tools=tools)
    if r.tool_calls:
        tc = r.tool_calls[0]
        print(f"[2] tool calling   OK  {r.latency_ms:.0f} ms  -> {tc.name}({json.dumps(tc.arguments)})  format={r.raw_tool_format}")
        if r.raw_tool_format != "native":
            print("    note: the server returned the call as text. For vLLM add --enable-auto-tool-choice --tool-call-parser qwen3_coder")
    else:
        ok = False
        print(f"[2] tool calling   FAIL no tool call. Content: {r.content[:200]!r}")
    print("usage:", r.usage)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
