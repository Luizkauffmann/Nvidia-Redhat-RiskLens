"""Smoke-test an LLM endpoint: one chat turn and one tool call.

    NVIDIA_API_KEY=nvapi-... python -m scripts.check_llm --provider nvidia
    RISKLENS_LLM_BASE_URL=http://<brev-host>:8000/v1 python -m scripts.check_llm --provider vllm
"""
import argparse

from risklens.llm.provider import get_provider
from risklens.tools import tool_schemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="nvidia", choices=["nvidia", "vllm", "openai", "mock"])
    p = get_provider(ap.parse_args().provider)
    print("model:", p.model)
    r = p.chat([{"role": "user", "content": "Reply with the single word: ready"}])
    print("chat:", r.content.strip()[:80], f"({r.latency_s:.1f}s)")
    r = p.chat([{"role": "user", "content": "What was the allowance change between 2026-07-31 and 2026-08-31? Use a tool."}],
               tools=tool_schemas(["compare_allowance", "get_reporting_context"]))
    print("tool call:", [(c.name, c.arguments) for c in r.tool_calls] or r.content[:200])


if __name__ == "__main__":
    main()
