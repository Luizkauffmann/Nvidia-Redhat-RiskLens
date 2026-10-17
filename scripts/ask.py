"""Ask RiskLens a question from the command line.

    python -m scripts.ask "Why did the allowance increase in August?"
    python -m scripts.ask "Which borrowers should I investigate?" --provider nvidia     # needs NVIDIA_API_KEY
    RISKLENS_LLM_BASE_URL=http://<brev-host>:8000/v1 python -m scripts.ask "..." --provider vllm
    python -m scripts.ask "..." --trace            # show every tool call with provenance
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.orchestrator import build_orchestrator  # noqa: E402
from src.llm.provider import get_provider  # noqa: E402
from src.tools import ToolCallLog  # noqa: E402
from src.utils import config  # noqa: E402


def print_trace(run) -> None:
    print("\n--- trace ---")
    for e in run.events:
        if e["type"] == "llm":
            what = ", ".join(e["tool_calls"]) or "answer"
            print(f"  model  -> {what}  ({e['latency_ms']:.0f} ms, tool format {e['tool_format']}, reasoning {e['reasoning_chars']} chars)")
        elif e["type"] == "tool":
            tag = "dup" if e.get("duplicate") else e["status"]
            print(f"  tool   {e['name']}({json.dumps(e.get('parameters') or {})}) [{tag}] call_id={e.get('call_id')}")
            if e.get("error"):
                print(f"         error: {e['error']['message']}  hint: {e['error'].get('hint')}")
        elif e["type"] == "grounding":
            print(f"  ground {e['status']} {e['ungrounded'] or ''}")
        else:
            print(f"  {e}")
    print("--- metrics ---")
    print(" ", json.dumps(run.metrics()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--provider", choices=["mock", "nvidia", "vllm", "openai"])
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full run record as JSON")
    a = ap.parse_args()
    overrides = {"provider": a.provider, "model": a.model, "base_url": a.base_url}
    if a.no_thinking:
        overrides["enable_thinking"] = False
    provider = get_provider(overrides)
    log = ToolCallLog(config.path("outputs") / "agent_tool_calls.jsonl")
    run = build_orchestrator(provider, log=log).run(a.question)
    if a.json:
        print(json.dumps(run.to_dict(), indent=2, default=str))
        return
    print(f"[{provider.describe()['provider']}:{provider.describe()['model']}] status={run.status} "
          f"grounding={(run.grounding or {}).get('status')} tools={run.tools_used} {run.duration_s}s\n")
    print(run.answer)
    if a.trace:
        print_trace(run)


if __name__ == "__main__":
    main()
