"""Compare any model's run log against the frontier reference, question by question.

    python -m scripts.compare_to_frontier                                  # your local run log
    python -m scripts.compare_to_frontier --runs demo_app/baselines/2026-10-17_stock-4b/runs.jsonl
    python -m scripts.compare_to_frontier --label "trained 4B" --markdown out.md

Runs are matched to the reference by the verifier's ``task_id`` (q01..q15, paraphrases map to their base
question), so any run logged by demo_app (bench or the app) that hit an evaluation-bank question counts.
Repeated runs of one question are averaged. Means are reported on the overlapping questions only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import config  # noqa: E402

DEFAULT_REF = Path("demo_app/baselines/2026-10-17_frontier-claude/frontier_reference.json")
COMPONENTS = ["correctness", "grounding", "tool_selection", "call_validity", "efficiency"]


def base_id(task_id: str) -> str:
    return re.sub(r"_p\d+$", "", task_id)


def load_runs(path: Path, label: str | None) -> dict[str, dict[str, list[dict]]]:
    """{label: {task_id: [runs]}} for scored runs."""
    out: dict[str, dict[str, list[dict]]] = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("quality") or r.get("source") == "frontier":
            continue
        if label and r["label"] != label:
            continue
        out.setdefault(r["label"], {}).setdefault(base_id(r["quality"]["task_id"]), []).append(r)
    return out


def compare(ref: dict, tasks: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    rows = []
    for tid, runs in sorted(tasks.items()):
        if tid not in ref["tasks"]:
            continue
        f = ref["tasks"][tid]
        rew = statistics.mean(r["quality"]["reward"] for r in runs)
        comps = {c: statistics.mean(r["quality"]["components"].get(c, 0.0) for r in runs) for c in COMPONENTS}
        rows.append({"task_id": tid, "question": f["question"], "runs": len(runs), "reward": round(rew, 3),
                     "frontier": f["reward"], "gap": round(rew - f["reward"], 3),
                     "components": {k: round(v, 3) for k, v in comps.items()},
                     "tools": sorted({t for r in runs for t in r["tools_used"]}), "frontier_tools": f["tools_used"]})
    tot = {"questions": len(rows)}
    if rows:
        tot["mean_reward"] = round(statistics.mean(r["reward"] for r in rows), 3)
        tot["frontier_mean_same_questions"] = round(statistics.mean(r["frontier"] for r in rows), 3)
        tot["recovery_pct"] = round(100 * tot["mean_reward"] / tot["frontier_mean_same_questions"], 1)
        tot["components"] = {c: round(statistics.mean(r["components"][c] for r in rows), 3) for c in COMPONENTS}
    return rows, tot


def render(label: str, rows: list[dict], tot: dict, ref: dict) -> str:
    if not rows:
        return f"## {label}\n\nNo scored evaluation-bank runs.\n"
    lines = [f"## {label}", "",
             f"{tot['questions']} questions in common. Mean reward **{tot['mean_reward']}** vs frontier "
             f"**{tot['frontier_mean_same_questions']}** on the same questions "
             f"(**{tot['recovery_pct']}%** of frontier).", "",
             "Mean components: " + ", ".join(f"{k} {v}" for k, v in tot["components"].items()), "",
             "| Task | Question | Runs | Reward | Frontier | Gap | Tools (model) | Tools (frontier) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task_id']} | {r['question']} | {r['runs']} | {r['reward']} | {r['frontier']} | "
                     f"{r['gap']:+.3f} | {', '.join(r['tools']) or '-'} | {', '.join(r['frontier_tools'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=DEFAULT_REF)
    ap.add_argument("--runs", type=Path, help="run log (default: data/outputs/demo_app_runs.jsonl)")
    ap.add_argument("--label", help="only this model label")
    ap.add_argument("--markdown", type=Path, help="also write the report to this file")
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    a = ap.parse_args()
    ref = json.loads(a.reference.read_text())
    runs_path = a.runs or config.path("outputs") / "demo_app_runs.jsonl"
    by_label = load_runs(runs_path, a.label)
    results = {lab: compare(ref, t) for lab, t in by_label.items()}
    if a.json:
        print(json.dumps({lab: {"summary": s, "tasks": r} for lab, (r, s) in results.items()}, indent=1))
        return
    head = (f"# Comparison with {ref['name']}\n\nReference: {a.reference} (mean {ref['mean_reward_all']} on all "
            f"{len(ref['tasks'])} questions, {ref['mean_reward_bench5']} on the 5 bench questions). Runs: {runs_path}\n")
    body = "\n".join(render(lab, r, s, ref) for lab, (r, s) in results.items()) or "No scored runs found.\n"
    print(head + "\n" + body)
    if a.markdown:
        a.markdown.write_text(head + "\n" + body)


if __name__ == "__main__":
    main()
