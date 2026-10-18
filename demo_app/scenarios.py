"""Scenario scoreboard for the demo app: every model configuration we evaluated, side by side.

Built only from committed result files, so the static GitHub Pages snapshot and the live app show the same table:

* base story, 15 evaluation-bank questions:
  - frontier reference (demo_app/baselines/*_frontier-claude/frontier_reference.json)
  - local 4B runs (demo_app/baselines/*_bank15/runs.jsonl, team runs already credited with the specialists' tools)
  - Lightning 30B base through NeMo Gym (evidence/nemotron_baseline/vllm_example_rollouts.jsonl)
* held-out story (property_shock, never in training), thinking on/off:
  - base vs LoRA from scripts/eval_adapter.py (evidence/lora_eval/<variant>_<think|nothink>.json)
  - local 4B through the same Gym harness (demo_app/baselines/*_bank15/local4b_<variant>_nothink_rollouts.jsonl)

No scoring happens here: every reward is the verifier's, as recorded in those files (demo_app/report.py builds the
markdown version of the same comparison). The two stories have different correct answers, so their tables are
kept apart.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from demo_app.report import _tid, heldout_row, load_gym
from demo_app.runlog import effective_quality, failed

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "demo_app" / "baselines"
EVIDENCE = ROOT / "evidence"
HF_ADAPTER = "https://huggingface.co/gcav66/nemotron-3.5-lightning-risklens-lora"


def _latest(pattern: str) -> Path | None:
    found = sorted(BASELINES.glob(pattern))
    return found[-1] if found else None


def _norm(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def _short(label: str, agents: str) -> str:
    """Column header for the per-question table."""
    if label.startswith("Frontier"):
        return "Frontier"
    if "Lightning" in label:
        return "Lightning 30B"
    if "4B" in label:
        runtime = next((rt for rt in ("MLX", "Ollama", "vLLM") if rt in label), "")
        return f"4B {runtime} {'team' if agents == 'team' else 'single'}".replace("  ", " ")
    return label[:24]


def _row(label: str, model: str, agents: str, hardware: str, source: str, per_q: dict[str, dict[str, Any]],
         order: list[str]) -> dict[str, Any]:
    q = [per_q[t]["quality"] for t in order if t in per_q]
    secs = [per_q[t]["seconds"] for t in order if per_q.get(t, {}).get("seconds") is not None]
    return {"label": label, "short": _short(label, agents), "model": model, "agents": agents, "hardware": hardware,
            "source": source,
            "answered": len(q), "mean_quality": round(statistics.mean(q), 3) if q else None,
            "fully_right": sum(x >= 0.999 for x in q),
            "median_s": round(statistics.median(secs), 1) if secs else None,
            "failed": sum(1 for t in order if per_q.get(t, {}).get("failed")),
            "per_question": {t: per_q[t] for t in order if t in per_q}}


def base_story() -> dict[str, Any]:
    """All configurations on the base story's 15 questions (q01-q15), same verifier."""
    frontier_dir = _latest("*_frontier-claude")
    bank_dir = _latest("*_bank15")
    if frontier_dir is None:
        return {"questions": [], "rows": []}
    ftasks = json.loads((frontier_dir / "frontier_reference.json").read_text())["tasks"]
    order = sorted(ftasks)
    questions = [{"task_id": t, "question": ftasks[t]["question"]} for t in order]
    by_question = {_norm(ftasks[t]["question"]): t for t in order}
    rows = [_row("Frontier reference (Claude, hand-driven)", "claude-opus-5-5", "single", "n/a", "frontier",
                 {t: {"quality": round(float(ftasks[t]["reward"]), 3), "seconds": None, "failed": False} for t in order},
                 order)]

    def collect(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
        by_label: dict[str, dict[str, dict[str, Any]]] = {}
        for r in runs:
            t = _tid(r) or by_question.get(_norm(r["question"]))
            if t in ftasks:                                   # last run per question wins, as in report.py
                by_label.setdefault(r["label"], {})[t] = r
        return by_label

    if bank_dir and (bank_dir / "runs.jsonl").exists():
        logged = [json.loads(line) for line in (bank_dir / "runs.jsonl").read_text().splitlines() if line.strip()]
        for label, per in collect(logged).items():
            r0 = next(iter(per.values()))
            rows.append(_row(label, r0["model"], "team" if r0.get("mode") == "team" else "single",
                             "M4 16 GB laptop" if "M4" in label else "", "app run log",
                             {t: {"quality": round(effective_quality(r)["reward"], 3), "seconds": r.get("duration_s"),
                                  "failed": failed(r.get("answer"))} for t, r in per.items()}, order))
    gym = EVIDENCE / "nemotron_baseline" / "vllm_example_rollouts.jsonl"
    if gym.exists():
        label = "Nemotron 3.5 Lightning 30B base | vLLM A100 | single-agent | NeMo Gym"
        per = collect(load_gym(gym, label)).get(label, {})
        if per:
            rows.append(_row(label, next(iter(per.values()))["model"], "single", "A100 80 GB", "NeMo Gym",
                             {t: {"quality": round(float(r["quality"]["reward"]), 3), "seconds": None, "failed": False}
                              for t, r in per.items()}, order))
    return {"story": "base", "questions": questions, "rows": rows}


def heldout_story() -> dict[str, Any]:
    """Held-out story (never in training): base vs LoRA with thinking on/off, plus the local 4B, same verifier."""
    evals = sorted((json.loads(p.read_text()) for p in (EVIDENCE / "lora_eval").glob("*think.json")),
                   key=lambda d: not d.get("thinking"))
    if not evals:
        return {"rows": [], "questions": []}
    variant, repeats = evals[0]["variant"], evals[0]["repeats"]
    order = sorted({t for d in evals for t in d["per_task"]})
    rows = []
    for d in evals:
        mode = "on" if d.get("thinking") else "off"
        for k, name in (("base", "Lightning 30B base"), ("lora", "Lightning 30B + LoRA")):
            rows.append({"label": f"{name}, thinking {mode}", "short": f"{'LoRA' if k == 'lora' else 'Base'}, think {mode}",
                         "model": d["models"][k], "thinking": mode,
                         "attempts_per_question": repeats, "mean_quality": round(d["mean_reward"][k], 3),
                         "no_tool_answers": d["no_tool_answers"][k], "answers": 15 * repeats,
                         "output_tokens": round(d["output_tokens_per_episode"][k]),
                         "per_question": {t: round(d["per_task"][t][k], 3) for t in order if t in d["per_task"]}})
    bank_dir = _latest("*_bank15")
    local = bank_dir and next(iter(sorted(bank_dir.glob(f"local4b_{variant}_nothink_rollouts.jsonl"))), None)
    if local:
        e = heldout_row("Local 4B stock (Ollama, M4 16 GB), thinking off", local)
        per: dict[str, list[float]] = {}
        for line in local.read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                tid = re.sub(r"_p\d+$", "", (g.get("verifier_metadata") or {}).get("task_id") or "")
                if tid:
                    per.setdefault(tid, []).append(float(g["reward"]))
        rows.append({"label": e["label"], "short": "Local 4B, think off", "model": "nemotron-3-nano-4b-32k", "thinking": "off",
                     "attempts_per_question": 1,
                     "mean_quality": round(e["mean"], 3), "no_tool_answers": e["no_tool"], "answers": e["n"],
                     "output_tokens": round(e["tokens"]),
                     "per_question": {t: round(statistics.mean(v), 3) for t, v in per.items()}})
    return {"story": variant, "questions": [{"task_id": t} for t in order], "rows": rows, "adapter": HF_ADAPTER}


def scoreboard() -> dict[str, Any]:
    return {"base": base_story(), "heldout": heldout_story(),
            "note": "Quality is the NeMo Gym verifier's reward (0-1): correct facts, grounding, tool choice, valid calls, "
                    "efficiency. The held-out story has different correct answers, so compare its rows with each other."}
