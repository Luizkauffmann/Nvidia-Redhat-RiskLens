"""Turn scored NeMo Gym rollouts into fine-tuning data.

Only rollouts that pass the verifier (``reward >= min_reward``) are kept, so the
SFT set contains investigations that reached the right conclusion with grounded
numbers and valid tool calls. Output is OpenAI chat format with tool calls, which
NeMo RL / NeMo Customizer / TRL SFT loaders accept:

    {"messages": [system, user, assistant(tool_calls), tool, ..., assistant(answer)], "tools": [...],
     "metadata": {"task_id", "reward", "policy"}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def rollout_to_messages(rollout: dict[str, Any]) -> dict[str, Any]:
    params = rollout["responses_create_params"]
    messages: list[dict[str, Any]] = [{"role": m["role"], "content": _text(m.get("content"))}
                                      for m in params.get("input", []) if m.get("role") in ("system", "user")]
    pending: list[dict[str, Any]] = []
    for item in (rollout.get("response") or {}).get("output", []):
        typ = item.get("type")
        if typ == "function_call":
            pending.append({"id": item["call_id"], "type": "function",
                            "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}})
        elif typ == "function_call_output":
            if pending:
                messages.append({"role": "assistant", "content": "", "tool_calls": pending})
                pending = []
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item.get("output", "")})
        elif typ == "message" and item.get("role", "assistant") == "assistant":
            if pending:
                messages.append({"role": "assistant", "content": "", "tool_calls": pending})
                pending = []
            text = _text(item.get("content"))
            if text.strip():
                messages.append({"role": "assistant", "content": text})
    tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
             for t in params.get("tools", [])]
    meta = rollout.get("verifier_metadata") or {}
    return {"messages": messages, "tools": tools,
            "metadata": {"task_id": meta.get("task_id"), "reward": rollout.get("reward"),
                         "components": rollout.get("reward_components")}}


def rescored(rollout: dict[str, Any]) -> dict[str, Any]:
    """The rollout with reward and components recomputed by the current verifier (rollouts keep the
    score of the verifier that ran at collection time)."""
    from src.evaluation import verifier

    answer, calls = verifier.from_responses_output((rollout.get("response") or {}).get("output", []))
    res = verifier.score(answer, calls, rollout.get("verifier_metadata") or {})
    return {**rollout, "reward": res.reward, "reward_components": res.components, "checks": res.checks}


def rollouts_to_sft(paths: Iterable[Path | str], min_reward: float = 0.8, out: Path | str | None = None,
                    dedupe: bool = True, rescore: bool = False) -> list[dict[str, Any]]:
    """Filter rollouts by reward and convert them; optionally write JSONL to ``out``.

    ``rescore`` re-runs the current verifier first, so a verifier fix applies to rollouts already collected.
    """
    rows, seen = [], set()
    for path in paths:
        with Path(path).open() as fh:
            for line in fh:
                r = json.loads(line)
                if rescore:
                    r = rescored(r)
                if (r.get("reward") or 0.0) < min_reward:
                    continue
                ex = rollout_to_messages(r)
                if not ex["messages"] or ex["messages"][-1]["role"] != "assistant":
                    continue
                key = json.dumps(ex["messages"], sort_keys=True)
                if dedupe and key in seen:
                    continue
                seen.add(key)
                ex["metadata"]["source"] = Path(path).name
                rows.append(ex)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for ex in rows:
                fh.write(json.dumps(ex) + "\n")
    return rows
