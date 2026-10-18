"""Offline demo app: the browser pack must match the Excel pack and tie out, and the API must work offline."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo_app.pack import SHEETS, build_pack
from demo_app.server import STATIC, app
from src.tools import ToolContext, call_tool, set_context
from src.utils import config
from tests.conftest import CURR, PREV


@pytest.fixture(scope="module", autouse=True)
def ctx(generated, attribution, tmp_path_factory):
    c = ToolContext(data_dir=generated[0], cache_dir=tmp_path_factory.mktemp("cache"), disk_cache=False)
    c._mem[f"attr_default_CECL_{PREV}_{CURR}"] = attribution
    set_context(c)
    old = os.environ.get("RISKLENS_OUTPUT_DIR")
    os.environ["RISKLENS_OUTPUT_DIR"] = str(tmp_path_factory.mktemp("outputs"))
    config.reload()
    yield c
    set_context(None)
    if old is None:
        os.environ.pop("RISKLENS_OUTPUT_DIR", None)
    else:
        os.environ["RISKLENS_OUTPUT_DIR"] = old
    config.reload()


@pytest.fixture(scope="module")
def pack():
    return build_pack(PREV, CURR, "CECL")


def _cells(pack):
    for sheet in pack["sheets"]:
        for t in sheet["tables"]:
            for row in [*t["rows"], *t.get("totals", []), *t.get("tail", [])]:
                yield sheet["name"], row


def test_pack_has_the_excel_sheets(pack):
    rec = call_tool("build_attribution_workbook", {"previous_date": PREV, "current_date": CURR}, log=None)
    assert rec.status == "ok"
    assert [s["name"] for s in pack["sheets"]] == SHEETS == rec.result["sheets"]


def test_pack_headline_matches_the_excel_pack(pack):
    rec = call_tool("build_attribution_workbook", {"previous_date": PREV, "current_date": CURR}, log=None)
    xl = rec.result["headline"]
    for key in ("previous_allowance", "current_allowance", "change", "explained_change", "primary_driver",
                "reconciliation", "top_borrower"):
        assert pack["headline"][key] == pytest.approx(xl[key]) if isinstance(xl[key], float) else pack["headline"][key] == xl[key]


def test_pack_ties_out(pack):
    h = pack["headline"]
    assert h["change"] == pytest.approx(h["current_allowance"] - h["previous_allowance"])
    assert abs(h["residual"]) < 1.0
    tail = pack["sheets"][1]["tables"][0]["tail"]
    assert abs(tail[1][1]["v"]) < 1.0                       # walk minus reported current


def test_every_tool_output_cites_a_listed_call(pack):
    ids = {c["call_id"] for c in pack["source_calls"]}
    inputs = [c for _, row in _cells(pack) for c in row if c["k"] == "input"]
    assert inputs and all(c["src"] in ids for c in inputs)
    derived = [c for _, row in _cells(pack) for c in row if c["k"] in ("formula", "link")]
    assert derived and all(c.get("x") for c in derived)     # every derived figure shows its formula


def test_api_works_offline():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    ctx = client.get("/api/context").json()
    assert PREV in ctx["reporting_dates"] and ctx["demo_questions"]
    p = client.get("/api/pack", params={"previous_date": PREV, "current_date": CURR}).json()
    assert p["headline"]["reconciliation"] == "PASS"
    t = client.post("/api/tool", json={"name": "compare_allowance",
                                       "arguments": {"previous_date": "July 2026", "current_date": "2026-08"}}).json()
    assert t["status"] == "ok" and t["parameters"]["previous_date"] == PREV
    a = client.post("/api/ask", json={"question": "Why did the allowance increase in August?", "provider": "mock"}).json()
    assert a["status"] == "answered" and a["tool_calls"] and a["grounding"]["status"] == "PASS"
    assert client.get("/api/pack", params={"previous_date": "1999-01-01", "current_date": CURR}).status_code == 400


def test_static_assets_load_nothing_from_the_internet():
    for f in Path(STATIC).iterdir():
        urls = re.findall(r"https?://[^\s\"'`)]+", f.read_text())
        assert all(re.match(r"https?://(localhost|127\.0\.0\.1)", u) for u in urls), (f.name, urls)


def test_static_export_replays_the_server(tmp_path):
    from scripts.export_static_demo import export, pack_name

    m = export(tmp_path / "site", frameworks=["CECL"])
    index = (tmp_path / "site" / "index.html").read_text()
    assert "window.RISKLENS_STATIC = true" in index and 'src="static/app.js"' in index
    assert (tmp_path / "site" / "static" / "app.js").exists()
    data = tmp_path / "site" / "data"
    snap = json.loads((data / pack_name("CECL", PREV, CURR, "json")).read_text())
    live = build_pack(PREV, CURR, "CECL")
    for key in ("previous_allowance", "current_allowance", "change", "residual", "reconciliation", "top_borrower"):
        assert snap["headline"][key] == live["headline"][key]
    assert (data / pack_name("CECL", PREV, CURR, "xlsx")).stat().st_size > 0
    answers = json.loads((data / "answers.json").read_text())
    for mode in ("team", "orchestrator"):
        assert set(answers[mode]) == set(m["demo_questions"])
        assert all(a["tool_calls"] for a in answers[mode].values())
    assert answers["team"][m["demo_questions"][0]]["grounding"]["status"] == "PASS"
    runs = json.loads((data / "runs.json").read_text())             # committed baselines for the Model runs tab
    assert set(runs) == {"summary", "runs"}


def test_runs_are_logged_with_timing_and_quality(ground_truth, monkeypatch):
    from demo_app import runlog
    from src.evaluation.tasks import build_tasks

    # Score against the test fixture's ground truth (CI has no data/ground_truth checked in).
    monkeypatch.setattr(runlog, "_TASKS", {runlog._norm(t.question): t for t in build_tasks(ground_truth)})
    client = TestClient(app)
    questions = client.get("/api/context").json()["demo_questions"]
    assert questions == runlog.bench_questions("all")       # core bank questions + knowledge-base questions
    before = len(runlog.load_runs())
    a = client.post("/api/ask", json={"question": questions[0], "provider": "mock", "label": "mock test"}).json()
    assert a["quality"] and 0.0 <= a["quality"]["reward"] <= 1.0
    assert set(a["timing"]) == {"llm_s", "tool_s", "llm_turns", "tokens_per_s"}
    client.post("/api/ask", json={"question": "Free-form question outside the bank?", "provider": "mock"})
    data = client.get("/api/runs").json()
    assert len(data["runs"]) == before + 2
    newest, scored = data["runs"][0], data["runs"][1]
    assert newest["quality"] is None and scored["label"] == "mock test" and scored["model"] == "mock-planner"
    row = next(r for r in data["summary"] if r["label"] == "mock test")
    assert row["runs"] >= 1 and row["mean_reward"] is not None


def test_demo_questions_are_evaluation_bank_questions(ground_truth):
    from demo_app import runlog
    from src.evaluation.tasks import build_tasks

    bank = {t.task_id: t.question for t in build_tasks(ground_truth)}
    assert [bank[i] for i in runlog.BENCH_TASKS] == runlog.BENCH_QUESTIONS


def test_failed_and_truncated_runs_are_flagged():
    from demo_app import runlog

    base = {"label": "x", "model": "m", "provider": "vllm", "duration_s": 1.0, "llm_s": 1.0, "tool_calls": 0,
            "tokens_per_s": None, "grounding": "NO_NUMBERS"}
    gave_up = {**base, "llm_turns": 2, "prompt_tokens": 4100,          # 2 turns x ~2,050 = Ollama's 4k cap
               "answer": "[RiskLens could not complete the investigation: the model kept returning a malformed tool call.]",
               "quality": {"reward": 0.988, "checks_passed": 2, "checks_total": 2}}   # text checks matched "not ... model"
    fine = {**base, "llm_turns": 2, "prompt_tokens": 25000, "answer": "No model version changed.",
            "quality": {"reward": 1.0, "checks_passed": 2, "checks_total": 2}}
    assert runlog.effective_quality(gave_up)["reward"] == 0.0 and runlog.effective_quality(fine)["reward"] == 1.0
    assert runlog.context_truncated(gave_up) and not runlog.context_truncated(fine)
    row = runlog.summary([gave_up, fine])[0]
    assert row["mean_reward"] == 0.5 and row["failed_runs"] == 1 and row["truncated_runs"] == 1


def test_team_runs_credit_the_specialists_tools(ground_truth, monkeypatch):
    """In team mode the orchestrator only shows ask_<specialist>; the specialist's own calls must count."""
    from demo_app import runlog
    from src.evaluation.tasks import build_tasks

    # Score against the test fixture's ground truth (CI has no data/ground_truth checked in).
    monkeypatch.setattr(runlog, "_TASKS", {runlog._norm(t.question): t for t in build_tasks(ground_truth)})

    q = "Did probability of default increase between July and August?"          # needs calculate_pd_migration
    run, rec = runlog.ask(q, "team", {"provider": "mock"}, "mock team", record=False)
    assert any(t.startswith("ask_") for t in run.tools_used)                      # it really delegated
    assert rec["scored_with_specialists"] and rec["quality"]["components"]["tool_selection"] == 1.0
    assert rec["quality"]["missing_tool_groups"] == []
