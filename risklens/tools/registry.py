"""Tool registry: schemas, argument validation and repair, provenance logging.

Agents only reach the analytics through `call_tool`. Every call returns a
ToolResult with structured JSON-compatible output (never prose) and a
provenance record (tool name, arguments, timestamp, sha256 of the result).
Mistakes a small model typically makes are repaired where the intent is
unambiguous ("cre" -> "CRE", "Aug 2026" -> "2026-08-31") and otherwise
answered with a structured error plus a recovery hint.
"""
from __future__ import annotations

import difflib
import hashlib
import inspect
import json
import math
import re
import time
import typing
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from risklens import config


class ToolError(Exception):
    def __init__(self, message: str, hint: dict | str | None = None):
        super().__init__(message)
        self.hint = hint


@dataclass
class Param:
    name: str
    type: str                      # string | integer | number | boolean | object
    description: str
    required: bool
    default: Any = None
    kind: str | None = None        # date | portfolio | scenario | None (drives repair)
    enum: list | None = None


@dataclass
class ToolSpec:
    name: str
    func: Callable
    description: str
    category: str
    params: list[Param]

    def schema(self) -> dict:
        props, req = {}, []
        for p in self.params:
            d: dict = {"type": p.type, "description": p.description}
            if p.enum:
                d["enum"] = p.enum
            props[p.name] = d
            if p.required:
                req.append(p.name)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": {"type": "object", "properties": props, "required": req}}}


TOOLS: dict[str, ToolSpec] = {}
_PY_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}
_KIND_BY_NAME = {"as_of_date": "date", "previous_date": "date", "current_date": "date",
                 "portfolio": "portfolio", "scenario": "scenario"}
_KIND_DESC = {
    "date": "Reporting date, YYYY-MM-DD (e.g. 2026-07-31).",
    "portfolio": "Portfolio: CRE, C&I, Mortgage, Auto, Consumer or Credit Card. Omit for the whole book.",
    "scenario": "Macro scenario: baseline, adverse or severe_adverse.",
}


def _parse_doc(doc: str) -> tuple[str, dict[str, str]]:
    lines = inspect.cleandoc(doc or "").splitlines()
    summary = lines[0] if lines else ""
    descs = {}
    for ln in lines[1:]:
        m = re.match(r"\s*(\w+):\s*(.+)", ln)
        if m:
            descs[m.group(1)] = m.group(2).strip()
    return summary, descs


def tool(category: str, enums: dict[str, list] | None = None):
    """Register a function as an agent tool. Docstring line 1 = description; 'arg: text' lines = arg docs."""
    def deco(fn):
        summary, descs = _parse_doc(fn.__doc__)
        params = []
        hints = typing.get_type_hints(fn)
        for name, p in inspect.signature(fn).parameters.items():
            ann = hints.get(name, str)
            base = next((a for a in getattr(ann, "__args__", (ann,)) if a is not type(None)), str)
            kind = _KIND_BY_NAME.get(name)
            desc = descs.get(name) or _KIND_DESC.get(kind, name)
            enum = (enums or {}).get(name)
            if kind == "scenario":
                enum = ["baseline", "adverse", "severe_adverse"]
            params.append(Param(name, _PY_TYPES.get(base, "string"), desc,
                                p.default is inspect._empty, None if p.default is inspect._empty else p.default,
                                kind, enum))
        TOOLS[fn.__name__] = ToolSpec(fn.__name__, fn, summary, category, params)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Argument repair

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_PORTFOLIO_ALIASES = {
    "cre": "CRE", "commercial real estate": "CRE", "real estate": "CRE", "commercialrealestate": "CRE",
    "c&i": "C&I", "ci": "C&I", "c and i": "C&I", "c&amp;i": "C&I", "commercial and industrial": "C&I",
    "commercial & industrial": "C&I", "commercial term loan": "C&I", "commercial": "C&I",
    "mortgage": "Mortgage", "mortgages": "Mortgage", "residential mortgage": "Mortgage", "residential": "Mortgage",
    "auto": "Auto", "autos": "Auto", "auto loan": "Auto", "auto loans": "Auto",
    "consumer": "Consumer", "personal": "Consumer", "personal loan": "Consumer", "personal loans": "Consumer",
    "credit card": "Credit Card", "credit cards": "Credit Card", "card": "Credit Card", "cards": "Credit Card", "cc": "Credit Card",
}
_ALL = {"", "all", "total", "none", "null", "whole book", "portfolio", "bank", "entire portfolio", "*"}


def available_dates() -> list[str]:
    from risklens.data.store import get_store
    return get_store().dates


def normalize_date(value: Any) -> str:
    dates = available_dates()
    s = str(value).strip().lower()
    if s in dates:
        return s
    if s in {"previous", "prior", "last month", "prior month", "previous_date", "prior period", "previous period"}:
        return dates[0]
    if s in {"current", "latest", "this month", "current month", "current_date"}:
        return dates[-1]
    m = re.match(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?$", s)
    month = year = None
    if m:
        year, month = int(m.group(1)), int(m.group(2))
    else:
        for name, num in _MONTHS.items():
            if re.search(rf"\b{name}", s):
                month = num
        y = re.search(r"(20\d{2})", s)
        year = int(y.group(1)) if y else None
    if month:
        hits = [d for d in dates if int(d[5:7]) == month and (year is None or int(d[:4]) == year)]
        if len(hits) == 1:
            return hits[0]
    raise ToolError(f"Unknown reporting date '{value}'.", {"available_dates": dates})


def normalize_portfolio(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _ALL:
        return None
    pfs = config.methodology()["portfolios"]
    if s in pfs:
        return s
    if s.lower() in _PORTFOLIO_ALIASES:
        return _PORTFOLIO_ALIASES[s.lower()]
    for p in pfs:
        if p.lower() == s.lower():
            return p
    raise ToolError(f"Unknown portfolio '{value}'.",
                    {"valid_portfolios": pfs, "did_you_mean": difflib.get_close_matches(s, pfs + list(_PORTFOLIO_ALIASES), 3)})


def _coerce(p: Param, v: Any) -> Any:
    if p.kind == "date":
        return normalize_date(v)
    if p.kind == "portfolio":
        return normalize_portfolio(v)
    if p.kind == "scenario" and v is not None:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        s = {"severe": "severe_adverse", "base": "baseline", "severely_adverse": "severe_adverse"}.get(s, s)
        if s not in p.enum:
            raise ToolError(f"Unknown scenario '{v}'.", {"valid_scenarios": p.enum})
        return s
    if v is None:
        return None
    if p.type == "integer":
        try:
            return int(float(v))
        except (TypeError, ValueError):
            raise ToolError(f"Argument '{p.name}' must be an integer, got {v!r}.")
    if p.type == "number":
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ToolError(f"Argument '{p.name}' must be a number, got {v!r}.")
    if p.type == "boolean" and isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    if p.type == "object" and isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            raise ToolError(f"Argument '{p.name}' must be a JSON object.")
    if p.enum and v not in p.enum:
        raise ToolError(f"Argument '{p.name}' must be one of {p.enum}, got {v!r}.")
    if p.type == "string" and not isinstance(v, str):
        return str(v)
    return v


# ---------------------------------------------------------------------------

def jsonable(x: Any) -> Any:
    """Convert numpy/pandas values to plain JSON types; money and ratios rounded for readability."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 2) if abs(f) >= 100 else round(f, 6)
    if hasattr(x, "isoformat"):
        return x.isoformat()[:10]
    return x


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    arguments: dict
    ok: bool
    result: Any = None
    error: str | None = None
    hint: Any = None
    repairs: list[str] = field(default_factory=list)
    timestamp: str = ""
    duration_ms: float = 0.0

    def provenance(self) -> dict:
        digest = hashlib.sha256(json.dumps(self.result, sort_keys=True, default=str).encode()).hexdigest()
        return {"call_id": self.call_id, "tool_name": self.tool_name, "parameters": self.arguments,
                "timestamp": self.timestamp, "ok": self.ok, "result_sha256": digest}

    def to_message(self) -> dict:
        """What goes back to the model."""
        if self.ok:
            return {"ok": True, "tool": self.tool_name, "call_id": self.call_id, "result": self.result}
        return {"ok": False, "tool": self.tool_name, "error": self.error, "hint": self.hint}


CALL_LOG: list[ToolResult] = []


def _log(res: ToolResult) -> None:
    CALL_LOG.append(res)
    try:
        with open(config.output_dir() / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({**res.provenance(), "duration_ms": res.duration_ms, "repairs": res.repairs,
                                "error": res.error}, default=str) + "\n")
    except OSError:
        pass


def call_tool(name: str, arguments: Any = None, log: bool = True) -> ToolResult:
    """Validate, repair, execute and log one tool call."""
    _load_tools()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cid = "call_" + uuid.uuid4().hex[:10]
    t0 = time.perf_counter()
    repairs: list[str] = []

    def fail(msg, hint=None, args=None):
        r = ToolResult(cid, name, args or {}, False, error=msg, hint=hint, repairs=repairs, timestamp=ts,
                       duration_ms=round((time.perf_counter() - t0) * 1000, 1))
        if log:
            _log(r)
        return r

    if name not in TOOLS:
        return fail(f"Unknown tool '{name}'.", {"did_you_mean": difflib.get_close_matches(name, list(TOOLS), 3)})
    spec = TOOLS[name]
    if arguments is None or arguments == "":
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
            repairs.append("parsed JSON string arguments")
        except json.JSONDecodeError:
            return fail("Arguments are not valid JSON.", {"expected_schema": spec.schema()["function"]["parameters"]})
    if not isinstance(arguments, dict):
        return fail("Arguments must be a JSON object.", {"expected_schema": spec.schema()["function"]["parameters"]})

    by_name = {p.name: p for p in spec.params}
    clean: dict = {}
    for k, v in arguments.items():
        key = k
        if k not in by_name:
            close = difflib.get_close_matches(k, list(by_name), 1, 0.75)
            if close and close[0] not in arguments:
                key = close[0]
                repairs.append(f"renamed argument '{k}' -> '{key}'")
            else:
                repairs.append(f"dropped unknown argument '{k}'")
                continue
        p = by_name[key]
        try:
            nv = _coerce(p, v)
        except ToolError as e:
            return fail(str(e), e.hint, arguments)
        if nv != v and not (v is None and nv is None):
            repairs.append(f"{key}: {v!r} -> {nv!r}")
        clean[key] = nv
    # Sensible defaults for missing comparison dates.
    for p in spec.params:
        if p.name not in clean and p.required:
            if p.name == "previous_date":
                clean[p.name] = available_dates()[0]
                repairs.append(f"defaulted previous_date to {clean[p.name]}")
            elif p.name == "current_date":
                clean[p.name] = available_dates()[-1]
                repairs.append(f"defaulted current_date to {clean[p.name]}")
            else:
                return fail(f"Missing required argument '{p.name}'.",
                            {"expected_schema": spec.schema()["function"]["parameters"]}, arguments)
    try:
        out = jsonable(spec.func(**clean))
    except ToolError as e:
        return fail(str(e), e.hint, clean)
    except KeyError as e:
        return fail(f"Not found: {e}", None, clean)
    except Exception as e:  # never crash the agent loop; report a structured error instead
        return fail(f"{type(e).__name__}: {e}", "Check the arguments against the tool schema.", clean)
    res = ToolResult(cid, name, clean, True, result=out, repairs=repairs, timestamp=ts,
                     duration_ms=round((time.perf_counter() - t0) * 1000, 1))
    if log:
        _log(res)
    return res


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    _load_tools()
    names = names or list(TOOLS)
    return [TOOLS[n].schema() for n in names if n in TOOLS]


_LOADED = False


def _load_tools() -> None:
    global _LOADED
    if not _LOADED:
        from risklens.tools import (allowance_tools, credit_tools, macro_tools, model_tools,  # noqa: F401
                                    portfolio_tools, reconciliation_tools)
        _LOADED = True


def as_dict(r: ToolResult) -> dict:
    return asdict(r)
