"""Hint distillation: teach a behaviour the policy never shows on its own.

Base Nemotron answers "what happened in CRE?" at portfolio level and never drills down to
borrowers (0 of 16 rollouts named a deteriorated CRE borrower), so its own passing rollouts
contain nothing to learn from. We re-run those questions with one extra sentence in the system
prompt, keep the rollouts the verifier passes, and export them with the original prompt
(``verifier_metadata.distill_system``): the fine-tuned model learns the behaviour unprompted.

Rows are built from an existing ``train.jsonl`` (base story or a variant) and written next to it
as ``hint.jsonl``; collect with ``run_gym_eval(policy, split="hint", variant=...)``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

HINTS: dict[str, str] = {
    "q14_cre": ("For a question about one portfolio, also call identify_top_contributors with that "
                "portfolio (for example portfolio=\"CRE\") and name the borrowers, by counterparty ID, "
                "whose deterioration drives its change."),
}
EXTRA_QUESTIONS: dict[str, list[str]] = {
    "q14_cre": ["Walk me through the Commercial Real Estate movement this month.",
                "Explain the change in the CRE allowance between July and August."],
}


def _base_task(row: dict[str, Any]) -> str:
    meta = row["verifier_metadata"]
    return meta.get("paraphrase_of") or meta["task_id"]


def hinted(row: dict[str, Any], hint: str, question: str | None = None, suffix: str = "hint") -> dict[str, Any]:
    out = copy.deepcopy(row)
    msgs = out["responses_create_params"]["input"]
    system = next(m for m in msgs if m["role"] == "system")
    out["verifier_metadata"]["distill_system"] = system["content"]
    system["content"] = f"{system['content']}\n\n{hint}"
    if question:
        next(m for m in msgs if m["role"] == "user")["content"] = question
        out["verifier_metadata"]["question"] = question
    out["id"] = f"{row['id']}__{suffix}"
    return out


def build_hint_rows(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in train_rows:
        task = _base_task(row)
        if task not in HINTS or row["verifier_metadata"].get("paraphrase_of"):
            continue
        rows.append(hinted(row, HINTS[task]))
        rows += [hinted(row, HINTS[task], q, f"hint_p{i}") for i, q in enumerate(EXTRA_QUESTIONS.get(task, []), 1)]
    return rows


def write_hint_file(train_path: Path | str) -> Path:
    train_path = Path(train_path)
    rows = [json.loads(line) for line in train_path.open() if line.strip()]
    out = train_path.with_name("hint.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in build_hint_rows(rows)))
    return out
