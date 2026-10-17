"""Run the NeMo Gym task bank in-process and score it with the same verifier.

This is the no-servers path: useful on a laptop without Gym running, in CI and
inside the notebooks. ``scripts/run_gym_eval.py --mode gym`` runs the same rows
through real NeMo Gym servers; both report the same reward components.

    from src.evaluation.local_eval import evaluate
    from src.llm.provider import get_provider
    report = evaluate(get_provider({"provider": "mock"}))
    report.summary()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.agents.base import ToolAgent
from src.evaluation import verifier
from src.evaluation.gym_dataset import _agent_tools, build_rows
from src.llm.provider import LLMProvider, llm_settings


@dataclass
class EvalReport:
    provider: dict[str, Any]
    harness: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        flat = []
        for r in self.rows:
            flat.append({"task_id": r["task_id"], "category": r["category"], "reward": r["reward"],
                         **{f"c_{k}": v for k, v in r["components"].items()},
                         "tool_calls": r["tool_calls"], "tool_errors": r["tool_errors"],
                         "checks_passed": f"{sum(c['passed'] for c in r['checks'])}/{len(r['checks'])}",
                         "seconds": r["seconds"]})
        return pd.DataFrame(flat)

    def summary(self) -> dict[str, Any]:
        df = self.frame()
        if df.empty:
            return {"episodes": 0}
        comps = {c[2:]: round(float(df[c].mean()), 4) for c in df.columns if c.startswith("c_")}
        return {"provider": self.provider.get("model", self.provider.get("name")), "harness": self.harness,
                "episodes": len(df), "mean_reward": round(float(df["reward"].mean()), 4),
                "pass_rate_reward_ge_0.8": round(float((df["reward"] >= 0.8).mean()), 4),
                "mean_components": comps, "mean_tool_calls": round(float(df["tool_calls"].mean()), 2),
                "tool_error_rate": round(float(df["tool_errors"].sum() / max(df["tool_calls"].sum(), 1)), 4)}

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, default=str) + "\n")
        return path


def run_row(row: dict[str, Any], provider: LLMProvider, harness: str = "risklens") -> dict[str, Any]:
    """Run one Gym dataset row with the RiskLens ToolAgent and score it."""
    meta = row["verifier_metadata"]
    msgs = row["responses_create_params"]["input"]
    system = next(m["content"] for m in msgs if m["role"] == "system")
    tools = [t["name"] for t in row["responses_create_params"]["tools"]]
    llm = llm_settings()
    agent = ToolAgent(row.get("agent", "orchestrator"), system, provider, tools=tools,
                      max_steps=llm.get("max_steps", 8), max_tool_calls=llm.get("max_tool_calls", 12),
                      result_chars=llm.get("result_chars", 6000),
                      check_numbers=(harness == "risklens"))
    t0 = time.perf_counter()
    run = agent.run(meta["question"])
    answer, calls = verifier.from_agent_run(run)
    res = verifier.score(answer, calls, meta)
    return {"task_id": meta["task_id"], "category": meta["category"], "question": meta["question"],
            "answer": answer, "status": run.status, **res.to_dict(),
            "seconds": round(time.perf_counter() - t0, 2), "agent_metrics": run.metrics()}


def evaluate(provider: LLMProvider, rows: Iterable[dict[str, Any]] | None = None, split: str = "example",
             harness: str = "risklens", agent: str | None = None, verbose: bool = False) -> EvalReport:
    """Evaluate ``provider`` on a split (example | train | validation) or explicit rows.

    harness="risklens" keeps our safeguards (grounding revision turn); "plain" turns them off,
    which is closer to NeMo Gym's simple_agent loop.
    """
    rows = list(rows) if rows is not None else build_rows(agent=agent)[split]
    report = EvalReport(provider=provider.describe(), harness=harness)
    for row in rows:
        r = run_row(row, provider, harness)
        report.rows.append(r)
        if verbose:
            print(f"{r['task_id']:<28} reward={r['reward']:.3f} calls={r['tool_calls']} "
                  f"checks={sum(c['passed'] for c in r['checks'])}/{len(r['checks'])}")
    return report


__all__ = ["EvalReport", "evaluate", "run_row", "_agent_tools"]
