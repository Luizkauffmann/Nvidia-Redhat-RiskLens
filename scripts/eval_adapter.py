"""Before/after: base Nemotron vs the LoRA adapter on the same Gym questions, same settings.

Serve both from one vLLM (deploy/brev_serve_nemotron.sh with MODEL=<BF16 base> LORA=<adapter dir>), then:

    python -m scripts.eval_adapter                               # held-out property_shock, 4 attempts each
    python -m scripts.eval_adapter --variant credit_led --repeats 2
    python -m scripts.eval_adapter --thinking                    # base with reasoning on (adapter trained off)

Both runs use thinking off by default, the format the adapter is trained on. Writes rollouts to
data/outputs/gym/eval_<label>_<variant>_example_rollouts.jsonl and evidence/lora_eval/<variant>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import pandas as pd

from src.evaluation.gym_runner import POLICIES, ROOT, load_rollouts, run_gym_eval

COMPONENTS = ["correctness", "grounding", "tool_selection", "call_validity", "efficiency"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="property_shock")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--adapter", default="risklens-lora", help="served LoRA model name")
    ap.add_argument("--thinking", action="store_true", help="reasoning on (default off, as trained)")
    ap.add_argument("--policy", default="vllm", help="endpoint from gym_runner.POLICIES (mock: dry run)")
    args = ap.parse_args()
    base_model = os.environ.get("RISKLENS_VLLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    runs = {"base": base_model, "lora": args.adapter}
    frames = {}
    for label, model in runs.items():
        policy = replace(POLICIES[args.policy], name=label, model=model)
        out = ROOT / "data" / "outputs" / "gym" / f"eval_{label}_{args.variant}_example_rollouts.jsonl"
        run_gym_eval(policy, split="example", variant=args.variant, num_repeats=args.repeats,
                     thinking=args.thinking, output=out)
        frames[label] = load_rollouts(out)

    per_task = pd.DataFrame({k: df.groupby("task_id")["reward"].mean() for k, df in frames.items()})
    per_task["delta"] = per_task["lora"] - per_task["base"]
    summary = {"variant": args.variant, "repeats": args.repeats, "thinking": args.thinking, "models": runs,
               "mean_reward": {k: round(float(df["reward"].mean()), 4) for k, df in frames.items()},
               "components": {k: {c: round(float(df[c].mean()), 4) for c in COMPONENTS} for k, df in frames.items()},
               "tool_errors": {k: int(df["tool_errors"].sum()) for k, df in frames.items()},
               "tool_calls": {k: int(df["tool_calls"].sum()) for k, df in frames.items()},
               "per_task": per_task.round(4).to_dict("index")}
    out_dir = ROOT / "evidence" / "lora_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.variant}.json").write_text(json.dumps(summary, indent=2))
    md = [f"# Base vs LoRA on `{args.variant}` ({args.repeats} attempts per question, thinking "
          f"{'on' if args.thinking else 'off'})", "",
          "| | base | LoRA |", "|---|---:|---:|",
          f"| mean reward | {summary['mean_reward']['base']:.3f} | {summary['mean_reward']['lora']:.3f} |",
          *[f"| {c} | {summary['components']['base'][c]:.3f} | {summary['components']['lora'][c]:.3f} |" for c in COMPONENTS],
          f"| tool calls / errors | {summary['tool_calls']['base']} / {summary['tool_errors']['base']} | "
          f"{summary['tool_calls']['lora']} / {summary['tool_errors']['lora']} |", "",
          "| Question | base | LoRA | Δ |", "|---|---:|---:|---:|",
          *[f"| {t} | {r['base']:.3f} | {r['lora']:.3f} | {r['delta']:+.3f} |" for t, r in summary["per_task"].items()]]
    (out_dir / f"{args.variant}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
