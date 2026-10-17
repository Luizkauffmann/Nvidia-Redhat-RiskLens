"""Write hint.jsonl next to every train.jsonl (base story and each training variant; held-out variants skipped).

    python -m scripts.build_hint_rows

Rows are the hinted questions from src/evaluation/hint_distill.py. Collect, then export:
    run_gym_eval("vllm", split="hint", num_repeats=8)                      # base story
    run_gym_eval("vllm", split="hint", variant="credit_led", num_repeats=8)
    rollouts_to_sft([...], rescore=True)      # hinted rollouts are saved with the original prompt
"""
from __future__ import annotations

from src.evaluation.gym_runner import DATA_DIR
from src.evaluation.hint_distill import write_hint_file
from src.evaluation.story_variants import HELD_OUT


def main() -> None:
    for train in sorted([DATA_DIR / "train.jsonl", *DATA_DIR.glob("variants/*/train.jsonl")]):
        if train.exists() and train.parent.name not in HELD_OUT:
            out = write_hint_file(train)
            print(f"{sum(1 for _ in out.open()):>3} rows -> {out}")


if __name__ == "__main__":
    main()
