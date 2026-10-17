"""Export the offline demo app as a static site (GitHub Pages), with every number precomputed by the tools.

    python -m scripts.generate_demo_data
    python -m scripts.export_static_demo            # writes _site/ ; open _site/index.html via any static server

What the static site contains:

* the attribution pack (same JSON as `GET /api/pack`) for every reporting-date pair and framework,
* the Excel pack for each (same file as `GET /api/export.xlsx`),
* recorded agent runs for the demo questions (same JSON as `POST /api/ask`, mock planner, team and single agent).

The page runs `demo_app/static/app.js` in snapshot mode: it reads these files instead of calling the server.
It cannot answer new questions; that needs the live server (`python -m demo_app.server`).
No numbers are computed in this script: it only calls the server's own functions and writes their output.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from demo_app import server
from demo_app.pack import build_pack
from src.tools import call_tool

ROOT = Path(__file__).resolve().parents[1]
APP_TAG = '<script src="static/app.js"></script>'
SNAPSHOT_TAG = '<script>window.RISKLENS_STATIC = true;</script>\n' + APP_TAG


def pack_name(framework: str, previous_date: str, current_date: str, ext: str) -> str:
    return f"pack_{framework}_{previous_date}_{current_date}.{ext}"


def export(out: Path, frameworks: list[str] | None = None) -> dict:
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    data.mkdir(parents=True)
    shutil.copytree(server.STATIC, out / "static")
    index = (server.STATIC / "index.html").read_text()
    if APP_TAG not in index:
        raise SystemExit(f"index.html no longer contains {APP_TAG!r}; update scripts/export_static_demo.py")
    (out / "index.html").write_text(index.replace(APP_TAG, SNAPSHOT_TAG))
    (out / ".nojekyll").write_text("")

    ctx = call_tool("get_reporting_context", {}, caller="static_export", log=None).result
    dates, frameworks = ctx["reporting_dates"], frameworks or ctx["frameworks"]
    packs = []
    for framework, (prev, cur) in itertools.product(frameworks, itertools.combinations(dates, 2)):
        p = build_pack(prev, cur, framework)
        (data / pack_name(framework, prev, cur, "json")).write_text(json.dumps(p))
        rec = call_tool("build_attribution_workbook", {"previous_date": prev, "current_date": cur, "framework": framework},
                        caller="static_export", log=None)
        if rec.status == "ok":
            shutil.copyfile(rec.result["path"], data / pack_name(framework, prev, cur, "xlsx"))
        packs.append({"framework": framework, "previous_date": prev, "current_date": cur, "xlsx": rec.status == "ok"})
        print(f"pack {framework} {prev} -> {cur}: change {p['headline']['change']:,.2f}, {p['headline']['reconciliation']}")

    answers: dict[str, dict[str, dict]] = {}
    for mode in ("team", "orchestrator"):
        answers[mode] = {}
        for q in server.DEMO_QUESTIONS:
            a = server.ask(server.AskRequest(question=q, mode=mode, provider="mock"))
            answers[mode][q] = a
            print(f"ask [{mode}] {q} -> {a['status']}, grounding {(a['grounding'] or {}).get('status')}")
    (data / "answers.json").write_text(json.dumps(answers))

    manifest = {**ctx, "demo_questions": server.DEMO_QUESTIONS, "packs": packs,
                "llm": {"provider": "snapshot", "model": "mock planner (recorded)"},
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    (data / "context.json").write_text(json.dumps(manifest))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    ap.add_argument("--framework", action="append", help="limit to these frameworks (default: all)")
    a = ap.parse_args()
    m = export(a.out, a.framework)
    print(f"Static demo written to {a.out} ({len(m['packs'])} packs, generated {m['generated_at']})")


if __name__ == "__main__":
    main()
