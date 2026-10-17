"""Quant Analyst tools: ad-hoc pandas analysis of the booked data in a guarded subprocess.

The 30 fixed tools answer the standard questions. For anything else ("loans over $5M in Texas whose
LTV rose more than 5 points") the Quant Analyst writes pandas code. The code runs in a separate
process (``src/sandbox/runner.py``) with time/memory limits, no network, no file writes and no access
to the ground truth. The numbers in the answer come from the execution result, never from the model,
so the golden rule still holds; the code itself is part of the provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.tools.base import PARAM_DOCS, ToolError, get_context, tool

QUANT_AGENTS = ["orchestrator", "quant_agent"]
RUNNER = Path(__file__).resolve().parents[1] / "sandbox" / "runner.py"
TIMEOUT_S = 60
MAX_CODE_CHARS = 8000

PARAM_DOCS.update({
    "code": ("Python (pandas) code. DataFrames available: loans, counterparties, results (loan-level ECL), "
             "summary, macro, weights, overlays, model_metadata, model_assignments, model_parameters. "
             "Helpers: pd, np, at(df, date), PREV, CURR. Assign the answer to `result`."),
    "purpose": "One sentence: what question this code answers (kept in the provenance log).",
})

TABLE_NOTES = {
    "loans": "Loan snapshots, one row per loan per reporting date (balances, EAD, risk drivers, model ids).",
    "counterparties": "Borrower master per date (ratings, financials, FICO, watchlist).",
    "results": "Official loan-level ECL per framework (CECL / IFRS9) and date: ecl, pd_12m, lgd, ead, scenario ECLs.",
    "summary": "Official allowance totals per run (allowance, overlays, scenario ECLs, weighted PD / LGD).",
    "macro": "Macro scenario paths per forecast vintage (as_of_date), scenario and quarter.",
    "weights": "Scenario weights per date.",
    "overlays": "Management overlays (qualitative adjustments) per date.",
    "model_metadata": "Model registry: model_id, type, version, status, deployment / calibration dates.",
    "model_assignments": "Which PD / LGD / EAD model each product used on each date.",
    "model_parameters": "Scorecard parameters by model_id.",
}


@tool("analysis", QUANT_AGENTS)
def describe_datasets() -> dict[str, Any]:
    """List the tables available to run_python_analysis with their columns, types, row counts and dates; call this before writing code."""
    store = get_context().store
    out = []
    for name, note in TABLE_NOTES.items():
        df = getattr(store, name)
        dates = sorted(df["as_of_date"].dt.strftime("%Y-%m-%d").unique().tolist()) if "as_of_date" in df else None
        out.append({"table": name, "description": note, "rows": int(len(df)), "dates": dates,
                    "columns": {c: str(t) for c, t in df.dtypes.items()}})
    return {"tables": out,
            "helpers": {"at(df, date)": "rows of df for one reporting date", "PREV / CURR": "the two reporting dates (Timestamps)"},
            "rules": ["Assign the answer to `result` (DataFrame, Series, dict or number); at most 50 rows come back.",
                      "Read-only: no file writes, network or subprocesses. 60 s limit.",
                      "Join loans to results on [loan_id, as_of_date]; filter results by framework == 'CECL'."]}


@tool("analysis", QUANT_AGENTS, slow=True)
def run_python_analysis(code: str, purpose: str = "") -> dict[str, Any]:
    """Run pandas code against the booked portfolio and ECL results for questions no other tool answers; returns the value assigned to `result`."""
    ctx = get_context()
    code = (code or "").strip()
    if not code:
        raise ToolError("code is empty", hint={"example": "result = at(results[results.framework == 'CECL'], CURR)"
                                                          ".groupby('portfolio').ecl.sum()"})
    if len(code) > MAX_CODE_CHARS:
        raise ToolError(f"code is longer than {MAX_CODE_CHARS} characters; split the analysis")
    prev, curr = ctx.dates[-2], ctx.dates[-1]          # PREV / CURR inside the code
    with tempfile.TemporaryDirectory(prefix="risklens_sbx_") as tmp:
        code_file = Path(tmp) / "analysis.py"
        code_file.write_text(code)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-I", str(RUNNER), str(ctx.store.dir), str(code_file), prev, curr],
                                  capture_output=True, text=True, timeout=TIMEOUT_S, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            raise ToolError(f"analysis exceeded {TIMEOUT_S} s", "timeout",
                            {"tip": "filter to one date / portfolio first and avoid row-by-row Python loops"})
        runtime = round(time.perf_counter() - t0, 2)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        out = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise ToolError("analysis process failed", "execution_error",
                        {"stderr": proc.stderr[-1500:], "returncode": proc.returncode})
    if out["status"] != "ok":
        raise ToolError(f"{out['error_type']}: {out['message']}", "code_error",
                        {"traceback": out.get("traceback"),
                         "tip": "call describe_datasets for table and column names, fix the code and run again"})
    return {"purpose": purpose, "previous_date": prev, "current_date": curr, "result": out["result"],
            "stdout": out.get("stdout", ""), "runtime_s": runtime,
            "code": code, "code_sha256": hashlib.sha256(code.encode()).hexdigest()[:16]}
