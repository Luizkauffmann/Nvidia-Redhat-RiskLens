"""Frontier validation: a frontier model (Claude, driven by hand in a chat session) answers the evaluation
questions with exactly the same inputs as the small models, and is scored by the same Gym verifier.

Same inputs means:
  * the same system prompt (``ORCHESTRATOR_PROMPT``, printed by ``prompt``),
  * the same 33 deterministic tools, called through ``call_tool`` (validation, argument repair, provenance),
  * no access to ``data/ground_truth`` while answering (the scorer reads it only after the answer is submitted),
  * the same verifier (``src/evaluation/verifier.py``) and run log (``data/outputs/demo_app_runs.jsonl``),
    so ``python -m demo_app.bench --summary`` puts the frontier run next to the 4B and mock baselines.

    python -m scripts.frontier_validate prompt
    python -m scripts.frontier_validate start "Why did our allowance increase in August?"
    python -m scripts.frontier_validate call calculate_portfolio_attribution '{"previous_date":"2026-07-31","current_date":"2026-08-31"}'
    python -m scripts.frontier_validate answer answer.md        # scores + logs the episode
    python -m scripts.frontier_validate questions               # the evaluation bank questions (text only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import reporting_context  # noqa: E402
from src.agents.grounding import check_grounding  # noqa: E402
from src.agents.orchestrator import ORCHESTRATOR_PROMPT  # noqa: E402
from src.tools import ToolCallLog, call_tool  # noqa: E402
from src.utils import config  # noqa: E402

LABEL = "frontier: Claude Opus 5.5 (hand-driven, same tools)"
MODEL = "claude-opus-5-5"


def session_path() -> Path:
    return config.path("outputs") / "frontier" / "current_session.json"


def load_session() -> dict:
    p = session_path()
    if not p.exists():
        sys.exit("no open episode: run `start \"<question>\"` first")
    return json.loads(p.read_text())


def save_session(s: dict) -> None:
    p = session_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1, default=str))


def cmd_prompt(_: argparse.Namespace) -> None:
    print(ORCHESTRATOR_PROMPT.format(**reporting_context()))


def cmd_questions(_: argparse.Namespace) -> None:
    # Question text only; the checks (which carry expected values) stay with the scorer.
    from src.evaluation.tasks import build_tasks
    for t in build_tasks(include_paraphrases=False):
        print(f"{t.task_id:24s} {t.question}")


def cmd_start(a: argparse.Namespace) -> None:
    save_session({"question": a.question, "started": time.time(), "calls": []})
    print(f"episode started: {a.question}")


def cmd_call(a: argparse.Namespace) -> None:
    s = load_session()
    log = ToolCallLog(config.path("outputs") / "frontier" / "tool_calls.jsonl")
    rec = call_tool(a.tool, a.arguments or "{}", caller="frontier", log=log)
    s["calls"].append({"name": rec.tool_name, "arguments": rec.arguments, "status": rec.status,
                       "result": rec.result, "error": rec.error, "duration_ms": rec.duration_ms,
                       "call_id": rec.call_id})
    save_session(s)
    out = rec.result if rec.status == "ok" else {"error": rec.error}
    text = json.dumps(out, indent=1, default=str)
    if a.max_chars and len(text) > a.max_chars:
        text = text[: a.max_chars] + f"\n... [truncated, {len(text)} chars]"
    print(f"[{rec.status}] {rec.tool_name} call_id={rec.call_id} {rec.duration_ms:.0f} ms")
    print(text)


def cmd_answer(a: argparse.Namespace) -> None:
    from demo_app.runlog import quality, runs_path
    s = load_session()
    answer = Path(a.file).read_text() if a.file != "-" else sys.stdin.read()
    run = SimpleNamespace(answer=answer, tool_records=[
        SimpleNamespace(tool_name=c["name"], arguments=c["arguments"], status=c["status"],
                        result=c["result"], error=c["error"]) for c in s["calls"]])
    sources = [c["result"] for c in s["calls"] if c["status"] == "ok"] + [s["question"]]
    g = check_grounding(answer, sources)
    q = quality(s["question"], run)            # reads ground truth only now, after the answer exists
    tool_s = sum(c["duration_ms"] for c in s["calls"]) / 1000
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "frontier",
        "label": LABEL, "provider": "anthropic (chat session)", "model": MODEL, "base_url": None,
        "thinking": None, "mode": "orchestrator", "question": s["question"], "status": "answered",
        "duration_s": round(time.time() - s["started"], 2), "llm_s": None, "tool_s": round(tool_s, 2),
        "llm_turns": len(s["calls"]) + 1, "prompt_tokens": 0, "completion_tokens": 0, "tokens_per_s": None,
        "tool_calls": len(s["calls"]), "tool_errors": sum(c["status"] != "ok" for c in s["calls"]),
        "tools_used": [c["name"] for c in s["calls"]], "grounding": g.to_dict().get("status"),
        "grounding_detail": g.to_dict(), "quality": q, "answer": answer,
    }
    for path in (runs_path(), config.path("outputs") / "frontier" / "frontier_runs.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    session_path().unlink()
    print(json.dumps({"question": s["question"], "tools_used": rec["tools_used"], "grounding": rec["grounding"],
                      "ungrounded": g.to_dict().get("ungrounded"), "quality": q}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompt").set_defaults(fn=cmd_prompt)
    sub.add_parser("questions").set_defaults(fn=cmd_questions)
    p = sub.add_parser("start"); p.add_argument("question"); p.set_defaults(fn=cmd_start)
    p = sub.add_parser("call"); p.add_argument("tool"); p.add_argument("arguments", nargs="?")
    p.add_argument("--max-chars", type=int, default=6000); p.set_defaults(fn=cmd_call)
    p = sub.add_parser("answer"); p.add_argument("file"); p.set_defaults(fn=cmd_answer)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
