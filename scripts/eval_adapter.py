"""Before/after: base Nemotron vs the LoRA adapter on the same Gym questions, same settings.

Serve both from one vLLM (deploy/brev_serve_nemotron.sh with MODEL=<BF16 base> LORA=<adapter dir>), then:

    python -m scripts.eval_adapter                               # held-out property_shock, 4 attempts each
    python -m scripts.eval_adapter --variant credit_led --repeats 2
    python -m scripts.eval_adapter --thinking                    # base with reasoning on (adapter trained off)

Both runs use thinking off by default, the format the adapter is trained on. Writes rollouts to
data/outputs/gym/eval_<label>_<variant>_<think|nothink>_example_rollouts.jsonl and
evidence/lora_eval/<variant>_<think|nothink>.{json,md}. Output tokens per episode come from Gym's usage.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.evaluation.gym_runner import POLICIES, ROOT, load_rollouts, run_gym_eval

COMPONENTS = ["correctness", "grounding", "tool_selection", "call_validity", "efficiency"]


def rollout_file(label: str, variant: str, mode: str) -> Path:
    return ROOT / "data" / "outputs" / "gym" / f"eval_{label}_{variant}_{mode}_example_rollouts.jsonl"


def summarize(variant: str, repeats: int, thinking: bool, runs: dict[str, str]) -> str:
    """Compare the two rollout files; write evidence/lora_eval/<variant>_<mode>.{json,md}, return the markdown."""
    mode = "think" if thinking else "nothink"
    frames, tokens = {}, {}
    for label in runs:
        out = rollout_file(label, variant, mode)
        frames[label] = load_rollouts(out)
        usage = [(json.loads(line).get("response") or {}).get("usage") or {} for line in out.open()]
        tokens[label] = round(sum(u.get("output_tokens") or 0 for u in usage) / max(len(usage), 1), 1)
    per_task = pd.DataFrame({k: df.groupby("task_id")["reward"].mean() for k, df in frames.items()})
    per_task["delta"] = per_task["lora"] - per_task["base"]
    summary = {"variant": variant, "repeats": repeats, "thinking": thinking, "models": runs,
               "mean_reward": {k: round(float(df["reward"].mean()), 4) for k, df in frames.items()},
               "components": {k: {c: round(float(df[c].mean()), 4) for c in COMPONENTS} for k, df in frames.items()},
               "tool_errors": {k: int(df["tool_errors"].sum()) for k, df in frames.items()},
               "tool_calls": {k: int(df["tool_calls"].sum()) for k, df in frames.items()},
               "no_tool_answers": {k: int((df["tool_calls"] == 0).sum()) for k, df in frames.items()},
               "output_tokens_per_episode": tokens,
               "per_task": per_task.round(4).to_dict("index")}
    out_dir = ROOT / "evidence" / "lora_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant}_{mode}.json").write_text(json.dumps(summary, indent=2))
    md = [f"# Base vs LoRA on `{variant}` ({repeats} attempts per question, thinking {'on' if thinking else 'off'})", "",
          "| | base | LoRA |", "|---|---:|---:|",
          f"| mean reward | {summary['mean_reward']['base']:.3f} | {summary['mean_reward']['lora']:.3f} |",
          *[f"| {c} | {summary['components']['base'][c]:.3f} | {summary['components']['lora'][c]:.3f} |" for c in COMPONENTS],
          f"| tool calls / errors | {summary['tool_calls']['base']} / {summary['tool_errors']['base']} | "
          f"{summary['tool_calls']['lora']} / {summary['tool_errors']['lora']} |",
          f"| answers with no tool call | {summary['no_tool_answers']['base']} | {summary['no_tool_answers']['lora']} |",
          f"| output tokens / episode | {tokens['base']} | {tokens['lora']} |", "",
          "| Question | base | LoRA | Δ |", "|---|---:|---:|---:|",
          *[f"| {t} | {r['base']:.3f} | {r['lora']:.3f} | {r['delta']:+.3f} |" for t, r in summary["per_task"].items()]]
    (out_dir / f"{variant}_{mode}.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="property_shock")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--adapter", default="risklens-lora", help="served LoRA model name")
    ap.add_argument("--thinking", action="store_true", help="reasoning on (default off, as trained)")
    ap.add_argument("--policy", default="vllm", help="endpoint from gym_runner.POLICIES (mock: dry run)")
    ap.add_argument("--summarize-only", action="store_true", help="re-read existing rollout files, no Gym run")
    args = ap.parse_args()
    runs = {"base": os.environ.get("RISKLENS_VLLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),
            "lora": args.adapter}
    mode = "think" if args.thinking else "nothink"
    if not args.summarize_only:
        for label, model in runs.items():
            run_gym_eval(replace(POLICIES[args.policy], name=label, model=model), split="example",
                         variant=args.variant, num_repeats=args.repeats, thinking=args.thinking,
                         output=rollout_file(label, args.variant, mode))
    print(summarize(args.variant, args.repeats, args.thinking, runs))

if __name__ == "__main__":
    main()
