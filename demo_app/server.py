"""Offline RiskLens demo app: the attribution pack in the browser, plus an Ask RiskLens panel.

    python -m demo_app.server                 # http://127.0.0.1:8050, mock planner, no network needed
    RISKLENS_LLM_PROVIDER=vllm RISKLENS_LLM_BASE_URL=http://localhost:11434/v1 \\
      RISKLENS_LLM_MODEL=nemotron-3-nano:4b python -m demo_app.server   # local Ollama

Everything is served from localhost: no CDN, no external fonts. Numbers come only from `call_tool`
(the pack) and from the agents' own tool calls (the panel), so the golden rule in CLAUDE.md holds.
"""
from __future__ import annotations

import argparse
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from demo_app.pack import build_pack  # noqa: E402
from src.agents.orchestrator import build_orchestrator  # noqa: E402
from src.agents.team import build_team  # noqa: E402
from src.llm.provider import get_provider  # noqa: E402
from src.tools import ToolCallLog, call_tool  # noqa: E402
from src.tools.base import ToolError  # noqa: E402
from src.utils import config  # noqa: E402

STATIC = Path(__file__).parent / "static"
DEMO_QUESTIONS = [
    "Why did the allowance increase in August?",
    "Which borrowers should I investigate?",
    "Does the attribution reconcile?",
    "Did any model or scenario weight change?",
]
_agent_lock = threading.Lock()  # one agent run at a time: small local models, shared tool cache

app = FastAPI(title="RiskLens offline demo")


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


class AskRequest(BaseModel):
    question: str
    mode: str = "team"                 # team | orchestrator
    provider: str | None = None        # mock | vllm | nvidia | openai (default: config / env)
    model: str | None = None
    base_url: str | None = None
    no_thinking: bool = True


@lru_cache(maxsize=8)
def _pack(previous_date: str, current_date: str, framework: str, top_n: int) -> dict[str, Any]:
    return build_pack(previous_date, current_date, framework, top_n)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/context")
def context() -> dict[str, Any]:
    rec = call_tool("get_reporting_context", {}, caller="demo_app", log=None)
    provider = get_provider().describe()
    return {**rec.result, "llm": provider, "demo_questions": DEMO_QUESTIONS}


@app.get("/api/pack")
def pack(previous_date: str, current_date: str, framework: str = "CECL", top_n: int = 25) -> dict[str, Any]:
    try:
        return _pack(previous_date, current_date, framework, top_n)
    except ToolError as exc:
        raise HTTPException(400, detail=str(exc)) from exc


@app.post("/api/tool")
def tool(req: ToolRequest) -> dict[str, Any]:
    """Call any registered tool directly: handy for wiring and testing new agents against the app."""
    rec = call_tool(req.name, req.arguments, caller="demo_app", log=None)
    return {**rec.provenance(), "status": rec.status, "result": rec.result, "error": rec.error}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    overrides = {"provider": req.provider, "model": req.model, "base_url": req.base_url}
    if req.no_thinking:
        overrides["enable_thinking"] = False
    provider = get_provider({k: v for k, v in overrides.items() if v is not None})
    log = ToolCallLog(config.path("outputs") / "demo_app_tool_calls.jsonl")
    build = build_team if req.mode == "team" else build_orchestrator
    with _agent_lock:
        run = build(provider, log=log).run(req.question)
    d = run.to_dict()
    team = next((e for e in run.events if e["type"] == "team"), None)
    return {"answer": run.answer, "status": run.status, "grounding": run.grounding, "provider": provider.describe(),
            "duration_s": run.duration_s, "metrics": d["metrics"], "tool_calls": d["tool_calls"],
            "specialists": team["specialists"] if team else []}


@app.get("/api/export.xlsx")
def export(previous_date: str, current_date: str, framework: str = "CECL") -> FileResponse:
    """The original Excel pack, for anyone who does have a spreadsheet app."""
    rec = call_tool("build_attribution_workbook", {"previous_date": previous_date, "current_date": current_date,
                                                   "framework": framework}, caller="demo_app", log=None)
    if rec.status != "ok":
        raise HTTPException(400, detail=rec.error["message"])
    path = Path(rec.result["path"])
    return FileResponse(path, filename=path.name)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
