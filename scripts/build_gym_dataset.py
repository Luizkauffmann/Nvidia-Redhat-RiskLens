"""Write the NeMo Gym datasets for the RiskLens environment.

    python -m scripts.build_gym_dataset                 # orchestrator (all tools)
    python -m scripts.build_gym_dataset --agent credit_agent  # a specialist's tool allow-list

Outputs resources_servers/risklens_cecl/data/{example,train,validation}.jsonl.
Rebuild after regenerating data or changing the task bank (src/evaluation/tasks.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.gym_dataset import build_rows, write_jsonl

OUT = Path(__file__).resolve().parents[1] / "resources_servers" / "risklens_cecl" / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None, help="agent whose tool allow-list the policy gets")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    splits = build_rows(agent=args.agent)
    for name, rows in splits.items():
        path = write_jsonl(rows, Path(args.out) / f"{name}.jsonl")
        print(f"{name:<11} {len(rows):>3} rows -> {path}")


if __name__ == "__main__":
    main()
