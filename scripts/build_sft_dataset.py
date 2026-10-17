"""Assemble the LoRA SFT set from collected Gym rollouts.

    python -m scripts.build_sft_dataset                          # data/outputs/gym/vllm_*_{train,hint}_rollouts.jsonl
    python -m scripts.build_sft_dataset --min-reward 0.9 --policy vllm

Keeps train and hint rollouts only (never the example split or a held-out variant), re-scores
them with the current verifier, keeps reward >= --min-reward, and writes
data/outputs/sft/<policy>_sft.jsonl plus a per-task count summary next to it.
Then: python -m scripts.finetune_lora --data data/outputs/sft/<policy>_sft.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.evaluation.story_variants import HELD_OUT
from src.evaluation.trajectories import rollouts_to_sft
from src.utils import config

GYM_OUT = config.ROOT / "data" / "outputs" / "gym"


def sources(policy: str) -> list[Path]:
    files = sorted(p for split in ("train", "hint") for p in GYM_OUT.glob(f"{policy}_*{split}_rollouts.jsonl"))
    return [p for p in files if not any(f"_{v}_" in p.name for v in HELD_OUT) and p.stat().st_size > 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="vllm")
    ap.add_argument("--min-reward", type=float, default=0.95)
    args = ap.parse_args()
    files = sources(args.policy)
    if not files:
        raise SystemExit(f"no {args.policy}_*_train/hint rollouts in {GYM_OUT}")
    out = config.ROOT / "data" / "outputs" / "sft" / f"{args.policy}_sft.jsonl"
    rows = rollouts_to_sft(files, min_reward=args.min_reward, out=out, rescore=True)
    total = sum(1 for p in files for _ in p.open())
    by_task = collections.Counter(r["metadata"]["task_id"] or "?" for r in rows)
    summary = {"min_reward": args.min_reward, "rollouts": total, "kept": len(rows),
               "sources": [p.name for p in files], "by_task": dict(sorted(by_task.items()))}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    for p in files:
        print(f"  {p.name}")
    print(f"kept {len(rows)} of {total} rollouts (reward >= {args.min_reward}) -> {out}")
    print(json.dumps(summary["by_task"], indent=1))


if __name__ == "__main__":
    main()
