"""Ask the investigator agent a question.

    python -m scripts.ask "Why did the allowance increase?"                     # mock planner, offline
    python -m scripts.ask "Which borrowers should I investigate?" --provider nvidia --trace
    python -m scripts.ask "Does everything reconcile?" --provider vllm          # RISKLENS_LLM_BASE_URL=http://<brev-host>:8000/v1
"""
import argparse
import json

from risklens.agents.agent import InvestigatorAgent
from risklens.llm.provider import get_provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", default=["Why did our allowance increase in August?"])
    ap.add_argument("--provider", choices=["mock", "nvidia", "vllm", "openai"])
    ap.add_argument("--trace", action="store_true", help="print every model turn and tool call")
    args = ap.parse_args()
    agent = InvestigatorAgent(get_provider(args.provider))
    run = agent.run(" ".join(args.question))
    if args.trace:
        for s in run.steps:
            print(f"[{s.kind}] {json.dumps(s.detail, default=str)[:400]}")
        print()
    print(run.answer)
    print("\n-- tools:", ", ".join(run.tools_used()))
    print("-- grounding:", run.grounding)
    print("-- metrics:", {k: v for k, v in run.metrics().items() if v not in (0, None)})


if __name__ == "__main__":
    main()
