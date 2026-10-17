"""Build NeMo Gym JSONL datasets from the task bank.

Each row is one episode in NeMo Gym's native format:

    {"id": ..., "responses_create_params": {"input": [system, user], "tools": [...], "parallel_tool_calls": false},
     "agent": "orchestrator", "verifier_metadata": {task_id, question, checks, required_tools, call_budget}}

``responses_create_params.tools`` are the agent's tool allow-list in Responses API
function format; the resources server exposes each one as ``POST /<tool_name>``.
``verifier_metadata`` is what ``verify()`` scores against; it is derived from the
ground truth at build time, so the server itself never reads ground truth.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.agents.base import reporting_context
from src.agents.orchestrator import ORCHESTRATOR_PROMPT
from src.evaluation.tasks import Task, build_tasks
from src.evaluation.verifier import gym_settings
from src.tools import list_tools, tool_schemas
from src.utils import config


def responses_tools(agent: str = "orchestrator") -> list[dict[str, Any]]:
    """The agent's tools as Responses API function definitions (OpenAI chat schemas flattened)."""
    allowed = set(_agent_tools(agent))
    out = []
    for s in tool_schemas():
        f = s["function"]
        if f["name"] in allowed:
            out.append({"type": "function", "name": f["name"], "description": f["description"],
                        "parameters": f["parameters"], "strict": False})
    return out


def _agent_tools(agent: str) -> list[str]:
    agents = config.load_yaml("agents").get("agents", {})
    if agent in agents and agents[agent].get("tools"):
        return list(agents[agent]["tools"])
    return list_tools(agent if agent != "orchestrator" else None)


def system_prompt() -> str:
    return ORCHESTRATOR_PROMPT.format(**reporting_context())


def task_row(task: Task, agent: str = "orchestrator", tools: list[dict[str, Any]] | None = None,
             prompt: str | None = None) -> dict[str, Any]:
    variant = config.story_variant()
    return {
        "id": f"{variant}__{task.task_id}" if variant else task.task_id,
        "agent": agent,
        "responses_create_params": {
            "input": [{"role": "system", "content": prompt or system_prompt()},
                      {"role": "user", "content": task.question}],
            "tools": tools if tools is not None else responses_tools(agent),
            "parallel_tool_calls": False,
        },
        "verifier_metadata": {
            "task_id": task.task_id, "question": task.question, "category": task.category,
            "checks": task.checks, "required_tools": task.required_tools, "call_budget": task.call_budget,
            "paraphrase_of": task.paraphrase_of, **({"story_variant": variant} if variant else {}),
        },
    }


def build_rows(agent: str | None = None, include_paraphrases: bool | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return {"example": [...], "train": [...], "validation": [...]} rows.

    example    = the 15 canonical demo questions (what the demo is judged on)
    validation = a held-out sample of paraphrases
    train      = the canonical questions + the remaining paraphrases
    """
    ds = gym_settings().get("dataset", {})
    agent = agent or ds.get("agent", "orchestrator")
    inc = ds.get("include_paraphrases", True) if include_paraphrases is None else include_paraphrases
    tasks = build_tasks(include_paraphrases=inc)
    tools, prompt = responses_tools(agent), system_prompt()
    rows = {t.task_id: task_row(t, agent, tools, prompt) for t in tasks}
    canonical = [t.task_id for t in tasks if t.paraphrase_of is None]
    paras = [t.task_id for t in tasks if t.paraphrase_of is not None]
    rng = random.Random(ds.get("seed", 7))
    n_val = round(len(paras) * float(ds.get("validation_fraction", 0.25)))
    val = set(rng.sample(paras, n_val)) if paras else set()
    return {"example": [rows[t] for t in canonical],
            "validation": [rows[t] for t in paras if t in val],
            "train": [rows[t] for t in canonical] + [rows[t] for t in paras if t not in val]}


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path
