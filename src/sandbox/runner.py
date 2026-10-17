"""Child process for ``run_python_analysis``: load the booked data, run the agent's code, write JSON.

    python -I src/sandbox/runner.py <data_dir> <code_file> <previous_date> <current_date>

The JSON result is the last line of the real stdout; anything the code prints is captured separately.

Guardrails, in order (a guardrail, not a security boundary: production would run this in a
container such as an OpenShift sandboxed container with no network):

1. CPU-time and memory limits (``resource``), wall-clock timeout in the parent.
2. The data is loaded first, then an audit hook blocks network, subprocesses, file writes,
   deletes, ``ctypes`` and any read of ``ground_truth``.
3. The code sees DataFrames, ``pd``, ``np`` and a few helpers, and must assign ``result``.
"""
from __future__ import annotations

import io
import json
import math
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_ROWS = 50
MAX_STDOUT = 2000


def _limits(cpu_s: int = 30, mem_gb: float = 4.0) -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        mem = int(mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ImportError, ValueError, OSError):
        pass


BLOCKED_PREFIXES = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.fork", "os.posix_spawn",
                    "os.kill", "ctypes.", "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
                    "shutil.rmtree", "os.mkdir", "os.chmod", "os.symlink", "os.truncate", "urllib.Request",
                    "http.client.connect", "webbrowser.open")


def _guard(event: str, args: tuple) -> None:
    if event.startswith(BLOCKED_PREFIXES):
        raise PermissionError(f"blocked in the analysis sandbox: {event}")
    if event == "open":
        path, mode = str(args[0]), args[1]
        if "ground_truth" in path:
            raise PermissionError("ground truth is evaluation-only and not readable from analysis code")
        if isinstance(mode, str) and any(m in mode for m in "wax+"):
            raise PermissionError("the analysis sandbox is read-only")
        if isinstance(mode, int) and mode & 0o3:      # os.open with O_WRONLY / O_RDWR
            raise PermissionError("the analysis sandbox is read-only")


def _jsonable(x, depth: int = 0):
    import numpy as np
    import pandas as pd

    if isinstance(x, pd.DataFrame):
        frame = x.reset_index() if not isinstance(x.index, pd.RangeIndex) else x
        rows = json.loads(frame.head(MAX_ROWS).to_json(orient="records", date_format="iso"))
        return {"type": "table", "n_rows": int(len(x)), "columns": [str(c) for c in frame.columns],
                "rows": rows, "truncated": len(x) > MAX_ROWS}
    if isinstance(x, pd.Series):
        s = x.head(MAX_ROWS)
        return {"type": "series", "name": str(x.name), "n": int(len(x)),
                "values": {str(k): _jsonable(v, depth + 1) for k, v in s.items()}, "truncated": len(x) > MAX_ROWS}
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(x.items())[:200]}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(x)[:200]]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if x is None or isinstance(x, (int, str)):
        return x
    return str(x)


def main(data_dir: str, code_file: str, previous_date: str, current_date: str) -> None:
    out: dict = {"status": "error"}
    try:
        _limits()
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import pandas as pd

        from src.data.loader import DataStore

        store = DataStore(data_dir)
        tables = {name: getattr(store, name) for name in
                  ("loans", "counterparties", "results", "summary", "macro", "weights", "overlays",
                   "model_metadata", "model_assignments", "model_parameters")}
        prev, curr = pd.Timestamp(previous_date), pd.Timestamp(current_date)

        def at(df: pd.DataFrame, date) -> pd.DataFrame:
            """Rows of ``df`` for one reporting date ('2026-08-31', PREV or CURR)."""
            return df[df["as_of_date"] == pd.Timestamp(date)]

        code = Path(code_file).read_text()
        namespace = {"pd": pd, "np": np, "at": at, "PREV": prev, "CURR": curr, **tables}
        sys.addaudithook(_guard)
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(code, "<analysis>", "exec"), namespace)          # noqa: S102  (the point of the tool)
        if "result" not in namespace:
            raise NameError("Assign your answer to a variable named `result` (DataFrame, Series, dict or number).")
        out = {"status": "ok", "result": _jsonable(namespace["result"]), "stdout": buf.getvalue()[-MAX_STDOUT:]}
    except BaseException as exc:                                          # report everything back to the agent
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        user_tb = [line for line in tb if "<analysis>" in line or not line.startswith("  File")]
        out = {"status": "error", "error_type": type(exc).__name__, "message": str(exc)[:500],
               "traceback": "".join(user_tb)[-1500:]}
    sys.__stdout__.write("\n" + json.dumps(out) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main(*sys.argv[1:5])
