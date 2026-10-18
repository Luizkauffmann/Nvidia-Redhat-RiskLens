"""Model comparison report for the 15 evaluation-bank questions: summary, charts and answers for review.

    python -m demo_app.report --out demo_app/baselines/2026-10-17_bank15 \\
        --label "nemotron-3-nano 4B 32k | Ollama | single-agent | M4 16GB" --label "..."

Writes to --out:
  runs.jsonl    the selected runs from data/outputs/demo_app_runs.jsonl
  README.md     per-configuration summary, two Mermaid bar charts (GitHub renders them), per-question table
  RESPONSES.md  each question with the frontier reference answer and every configuration's answer, for SME review

The frontier reference (demo_app/baselines/2026-10-17_frontier-claude) is included as the upper bound; it was
hand-driven, so it has quality but no comparable time. Failed runs score 0 (runlog.effective_quality).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_app.runlog import context_truncated, effective_quality, failed, load_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "demo_app" / "baselines" / "2026-10-17_frontier-claude" / "frontier_reference.json"
BAR_COLOR = "#2a78d6"   # categorical slot 1, validated on GitHub light and dark surfaces


def _tid(r: dict) -> str | None:
    q = r.get("quality") or {}
    return re.sub(r"_p\d+$", "", q["task_id"]) if q.get("task_id") else None


def _short(label: str) -> str:
    """Chart axis label: the part before the first '|' detail, kept short."""
    parts = [p.strip() for p in label.split("|")]
    return " · ".join(parts[1:3]) if len(parts) >= 3 else label[:28]


def _cell(text: str) -> str:
    """Markdown table cell: labels use '|' as a separator, which would split the column."""
    return str(text).replace("|", "\\|")


def _bar_chart(title: str, y_title: str, labels: list[str], values: list[float], y_max: float) -> str:
    cats = ", ".join(json.dumps(x) for x in labels)
    vals = ", ".join(f"{v:.3g}" for v in values)
    return "\n".join([
        "```mermaid",
        "%%{init: {'themeVariables': {'xyChart': {'plotColorPalette': '" + BAR_COLOR + "'}}}}%%",
        "xychart-beta",
        f'    title "{title}"',
        f"    x-axis [{cats}]",
        f'    y-axis "{y_title}" 0 --> {y_max:.3g}',
        f"    bar [{vals}]",
        "```",
    ])


def load_gym(path: Path, label: str) -> list[dict]:
    """NeMo Gym rollouts (e.g. evidence/nemotron_baseline) as run-log-shaped records: same verifier, so the same
    reward; Gym does not record per-question wall time, so time is None."""
    out = []
    for line in open(path):
        if not line.strip():
            continue
        g = json.loads(line)
        q = g["responses_create_params"]["input"][-1]["content"]
        q = q if isinstance(q, str) else "".join(p.get("text", "") for p in q)
        items = (g.get("response") or {}).get("output") or []
        tools = [i.get("name") for i in items if i.get("type") == "function_call"]
        model = (g.get("response") or {}).get("model") or "?"
        out.append({"label": label, "source": "gym", "model": model, "mode": "orchestrator", "question": q,
                    "duration_s": None, "llm_s": None, "tools_used": tools, "answer": g.get("final_answer") or "",
                    "quality": {"reward": float(g["reward"]), "task_id": None}, "llm_turns": 0, "prompt_tokens": 0})
    return out


HF_ADAPTER = "https://huggingface.co/gcav66/nemotron-3.5-lightning-risklens-lora"


def heldout_row(label: str, path: Path) -> dict:
    """Summary of extra Gym rollouts on the same held-out story (e.g. the local 4B), for the four-way table."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    tokens = [((r.get("response") or {}).get("usage") or {}).get("output_tokens", 0) for r in rows]
    return {"label": label, "n": len(rows), "mean": statistics.mean(float(r["reward"]) for r in rows),
            "no_tool": sum(1 for r in rows if not r.get("tool_calls")), "tokens": statistics.mean(tokens) if tokens else 0}


def lora_section(paths: list[Path], source: str, hardware: str, extra: list[dict] | None = None) -> list[str]:
    """Base vs fine-tuned from scripts/eval_adapter.py outputs (evidence/lora_eval/<variant>_<think|nothink>.json):
    one four-way table across thinking on/off, a chart, then per-question detail per mode. Kept separate from the
    cross-model table: it is scored on a held-out story with different correct answers."""
    evals = sorted((json.loads(p.read_text()) for p in paths), key=lambda d: not d.get("thinking"))
    variant, repeats = evals[0]["variant"], evals[0]["repeats"]
    n = 15 * repeats
    rows = [(f"{'base' if k == 'base' else 'LoRA'}, thinking {'on' if d.get('thinking') else 'off'}", d, k)
            for d in evals for k in ("base", "lora")]
    md = ["", f"## Fine-tuning result: base vs LoRA (held-out story `{variant}`)", "",
          f"Gus's evaluation ([{source}]({source})): base Lightning 30B and the LoRA adapter (`risklens-lora`) served "
          f"from one vLLM on {hardware}, same NeMo Gym environment and verifier, {repeats} attempts per question. "
          f"The `{variant}` story was never in training and has different correct answers from the base story above, "
          "so compare these rows with each other, not with the table above.", "",
          f"The adapter is public on Hugging Face: [{HF_ADAPTER.split('co/')[1]}]({HF_ADAPTER}) "
          "(PEFT LoRA, rank 16, 45.5 MB; runs on top of `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`).", "",
          "| On the held-out story | Mean quality | Answers with no tool call | Output tokens / answer |",
          "|---|---:|---:|---:|"]
    md += [f"| {name} | **{d['mean_reward'][k]:.3f}** | {d['no_tool_answers'][k]} / {n} | "
           f"{d['output_tokens_per_episode'][k]:,.0f} |" for name, d, k in rows]
    md += [f"| {_cell(e['label'])} | **{e['mean']:.3f}** | {e['no_tool']} / {e['n']} | {e['tokens']:,.0f} |" for e in extra or []]
    labels = [name for name, _, _ in rows] + [_short(e["label"]) if "|" in e["label"] else e["label"] for e in extra or []]
    values = [d["mean_reward"][k] for _, d, k in rows] + [e["mean"] for e in extra or []]
    md += ["", _bar_chart(f"Mean quality on the held-out story ({variant})", "quality", labels, values, 1.0), ""]
    if extra:
        md += ["Extra rows ran the same story, Gym harness, verifier and thinking-off setting (Ollama needs its own "
               "`reasoning_effort: none`; it ignores `chat_template_kwargs`), with 1 attempt per question.", ""]
    if len(evals) == 2:
        on, off = evals
        md += [f"**Reading:** with thinking on, the adapter changes nothing ({on['mean_reward']['lora']:.3f} vs base "
               f"{on['mean_reward']['base']:.3f}). With thinking off it lifts quality from {off['mean_reward']['base']:.3f} "
               f"to {off['mean_reward']['lora']:.3f} and stops answers without a tool call "
               f"({off['no_tool_answers']['base']} → {off['no_tool_answers']['lora']} of {n}), close to base with thinking "
               f"on while writing about {100 * (1 - off['output_tokens_per_episode']['lora'] / on['output_tokens_per_episode']['base']):.0f}% "
               f"fewer output tokens ({off['output_tokens_per_episode']['lora']:,.0f} vs {on['output_tokens_per_episode']['base']:,.0f}).", ""]
    for d in evals:
        mode = "thinking on" if d.get("thinking") else "thinking off"
        c = d["components"]
        md += [f"### {mode.capitalize()}", "", "| | base | LoRA |", "|---|---:|---:|"]
        md += [f"| {k.replace('_', ' ')} | {c['base'][k]:.3f} | {c['lora'][k]:.3f} |" for k in c["base"]]
        md += [f"| tool calls / errors | {d['tool_calls']['base']} / {d['tool_errors']['base']} | "
               f"{d['tool_calls']['lora']} / {d['tool_errors']['lora']} |", "",
               "| Question | base | LoRA | change |", "|---|---:|---:|---:|"]
        md += [f"| {t} | {v['base']:.3f} | {v['lora']:.3f} | {v['delta']:+.3f} |" for t, v in d["per_task"].items()]
        md.append("")
    return md


WEIGHTS = {"correctness": 0.55, "grounding": 0.2, "tool_selection": 0.1, "call_validity": 0.1, "efficiency": 0.05}


def credit_specialists(runs: list[dict], call_log: Path) -> None:
    """Team runs logged before the scorer counted specialists' tools: credit a missing required tool group when the
    tool-call log shows a specialist called one of its tools during that run. Keeps the logged score as
    quality_as_logged. config/gym.yaml weights."""
    if not call_log.exists():
        return
    calls = [json.loads(line) for line in open(call_log) if line.strip()]
    t = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    for r in runs:
        q = r.get("quality")
        if r.get("mode") != "team" or r.get("scored_with_specialists") or not q or not q.get("missing_tool_groups"):
            continue
        end = t(r["ts"]); start = end - timedelta(seconds=r["duration_s"] + 2)
        used = {c["tool_name"] for c in calls if c.get("status") == "ok" and start <= t(c["timestamp"]) <= end + timedelta(seconds=2)}
        if all(any(tool in used for tool in group) for group in q["missing_tool_groups"]):
            comp = {**q["components"], "tool_selection": 1.0}
            r["quality_as_logged"] = q
            r["quality"] = {**q, "components": comp, "missing_tool_groups": [],
                            "reward": round(sum(WEIGHTS[k] * comp.get(k, 0) for k in WEIGHTS), 3),
                            "checks_passed": q.get("checks_passed"),
                            "note": "specialists' tool calls credited from the tool-call log"}


def build(out: Path, labels: list[str], gym: list[tuple[str, Path]] | None = None,
          lora: tuple[list[Path], str, str] | None = None, extra: list[dict] | None = None) -> None:
    logged = [r for r in load_runs() if r["label"] in labels]
    from demo_app.runlog import runs_path
    credit_specialists(logged, runs_path().parent / "demo_app_tool_calls.jsonl")
    runs = list(logged)
    for glabel, gpath in gym or []:
        runs += load_gym(gpath, glabel)
        labels = labels + [glabel]
    frontier = json.loads(FRONTIER.read_text())
    ftasks = frontier["tasks"]
    order = sorted(ftasks)                                   # q01..q15
    by_question = {re.sub(r"\W+", " ", ftasks[t]["question"].lower()).strip(): t for t in order}
    by_label: dict[str, dict[str, dict]] = {lab: {} for lab in labels}
    for r in runs:
        t = _tid(r) or by_question.get(re.sub(r"\W+", " ", r["question"].lower()).strip())
        if t in ftasks:
            by_label[r["label"]][t] = r                      # last run per question wins
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, default=str) + "\n" for r in logged)

    rows = []
    for lab in labels:
        rs = list(by_label[lab].values())
        if not rs:
            continue
        q = [effective_quality(r)["reward"] for r in rs]
        timed = [r for r in rs if r.get("duration_s") is not None]
        # team runs logged before runlog counted the specialists' turns: their model time is incomplete
        model_timed = [r for r in timed if r.get("mode") != "team" or r.get("scored_with_specialists")]
        rows.append({"label": lab, "model": rs[-1]["model"], "mode": rs[-1]["mode"], "answered": len(rs),
                     "mean_quality": round(statistics.mean(q), 3), "fully_right": sum(x >= 0.999 for x in q),
                     "median_s": round(statistics.median(r["duration_s"] for r in timed), 1) if timed else None,
                     "median_model_s": round(statistics.median(r["llm_s"] for r in model_timed), 1) if model_timed else None,
                     "credited": sum(1 for r in rs if r.get("quality_as_logged")),
                     "as_logged": round(statistics.mean((r.get("quality_as_logged") or r["quality"])["reward"] for r in rs), 3),
                     "total_min": round(sum(r["duration_s"] for r in timed) / 60, 1) if timed else None,
                     "failed": sum(failed(r.get("answer")) for r in rs), "truncated": sum(context_truncated(r) for r in rs)})
    fmean = round(statistics.mean(float(ftasks[t]["reward"]) for t in order), 3)

    md = ["# Model comparison: 15 evaluation-bank questions", "",
          "Same questions (q01–q15), tools, prompts and verifier (`src/evaluation/verifier.py`) for every configuration. "
          "Runs were driven through the demo app's Ask panel and logged by `demo_app/runlog.py`. Quality is the verifier "
          "reward (0–1): correct facts, grounding, tool choice, valid calls, efficiency. A run that gave up scores 0. "
          "Answers for review: [RESPONSES.md](RESPONSES.md).", "",
          "| Configuration | Model | Agents | Answered | Mean quality | Fully right | Median time / question | Median model time | Total | Failed | Truncated prompts |",
          "|---|---|---|---|---|---|---|---|---|---|---|",
          f"| **Frontier reference** (Claude Opus 5.5, hand-driven) | claude-opus-5-5 | single | 15 | **{fmean}** | "
          f"{sum(float(ftasks[t]['reward']) >= 0.999 for t in order)} / 15 | n/a | n/a | n/a | 0 | 0 |"]
    na = lambda v, unit: "n/a" if v is None else f"{v} {unit}"
    md += [f"| {_cell(r['label'])} | `{r['model']}` | {'team' if r['mode'] == 'team' else 'single'} | {r['answered']} | **{r['mean_quality']}** | "
           f"{r['fully_right']} / {r['answered']} | {na(r['median_s'], 's')} | {na(r['median_model_s'], 's')} | "
           f"{na(r['total_min'], 'min')} | {r['failed']} | {r['truncated']} |" for r in rows]
    for r in rows:
        if r.get("credited"):
            md += ["", f"*{_cell(r['label'])}*: quality credits the specialists' tool calls, verified from the tool-call "
                   f"log for {r['credited']} delegated answers (the scorer only saw the orchestrator's `ask_<specialist>`); "
                   f"as first scored: {r['as_logged']}. Median model time is n/a because those runs only timed the "
                   "orchestrator's turns."]
    if rows:
        md += ["", "## Charts", "",
               _bar_chart("Mean answer quality (verifier reward, 0-1)", "quality",
                          ["Frontier ref."] + [_short(r["label"]) for r in rows], [fmean] + [r["mean_quality"] for r in rows], 1.0),
               "", _bar_chart("Median seconds per question (lower is faster)", "seconds",
                              [_short(r["label"]) for r in rows if r["median_s"] is not None],
                              [r["median_s"] for r in rows if r["median_s"] is not None],
                              max([r["median_s"] for r in rows if r["median_s"] is not None] or [1]) * 1.15),
               "", "Configurations without a time bar have no comparable per-question time: the frontier reference was "
                   "driven by hand, and NeMo Gym rollouts do not record wall time per question."]
    md += ["", "## Quality by question", "",
           "| Question | Frontier | " + " | ".join(_cell(_short(r["label"])) for r in rows) + " |",
           "|---|---|" + "---|" * len(rows)]
    for t in order:
        cells = []
        for r in rows:
            run = by_label[r["label"]].get(t)
            if run is None:
                cells.append("–")
            else:
                secs = "" if run.get("duration_s") is None else f" ({run['duration_s']:.0f} s)"
                cells.append(f"{effective_quality(run)['reward']:.2f}{secs}")
        md.append(f"| {t}: {ftasks[t]['question']} | {float(ftasks[t]['reward']):.2f} | " + " | ".join(cells) + " |")
    md += ["", "Cells: quality (seconds for that question)."]
    if lora:
        md += lora_section(*lora, extra=extra)
    (out / "README.md").write_text("\n".join(md) + "\n")

    resp = ["# Answers for review", "",
            "For each question: the frontier reference answer, then each configuration's answer with its verifier score, "
            "time and tools. Scores come from the automated verifier; please flag answers you would judge differently.", ""]
    for t in order:
        f = ftasks[t]
        resp += [f"## {t}: {f['question']}", "",
                 f"**Frontier reference** (quality {float(f['reward']):.2f}; tools: {', '.join(f['tools_used'])})", "",
                 "> " + f["answer"].strip().replace("\n", "\n> "), ""]
        for r in rows:
            run = by_label[r["label"]].get(t)
            if run is None:
                resp += [f"**{r['label']}**: not run", ""]
                continue
            eq = effective_quality(run)
            note = " (gave up)" if failed(run.get("answer")) else ""
            secs = "time n/a" if run.get("duration_s") is None else f"{run['duration_s']:.0f} s"
            resp += [f"<details><summary><b>{r['label']}</b>: quality {eq['reward']:.2f}{note}, {secs}, "
                     f"tools: {', '.join(run['tools_used']) or 'none'}</summary>", "",
                     "> " + (run["answer"] or "").strip().replace("\n", "\n> "), "", "</details>", ""]
    (out / "RESPONSES.md").write_text("\n".join(resp) + "\n")
    print(f"wrote {out}/README.md, RESPONSES.md, runs.jsonl ({len(runs)} runs, {len(rows)} configurations)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", action="append", required=True, help="run label to include (repeatable, in order)")
    ap.add_argument("--gym", action="append", default=[], metavar="LABEL=PATH",
                    help="NeMo Gym rollouts to include, e.g. 'Lightning 30B | vLLM A100=evidence/nemotron_baseline/vllm_example_rollouts.jsonl'")
    ap.add_argument("--lora-eval", type=Path, action="append", default=[],
                    help="scripts/eval_adapter.py summary json (evidence/lora_eval/*.json); repeat for thinking on and off")
    ap.add_argument("--lora-source", default="../../../evidence/lora_eval/README.md", help="link shown for the LoRA section")
    ap.add_argument("--lora-hardware", default="one H100", help="GPU the LoRA evaluation ran on")
    ap.add_argument("--heldout-extra", action="append", default=[], metavar="LABEL=PATH",
                    help="extra Gym rollouts on the same held-out story, added to the base-vs-LoRA table")
    a = ap.parse_args()
    extra = [heldout_row(e.split("=", 1)[0], Path(e.split("=", 1)[1])) for e in a.heldout_extra]
    build(a.out, a.label, [(g.split("=", 1)[0], Path(g.split("=", 1)[1])) for g in a.gym],
          (a.lora_eval, a.lora_source, a.lora_hardware) if a.lora_eval else None, extra)


if __name__ == "__main__":
    main()
