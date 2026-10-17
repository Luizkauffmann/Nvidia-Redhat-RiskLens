"""Tool layer foundations: registry, argument validation, provenance, caching.

Every function an agent may call is registered with ``@tool``. The registry
derives a JSON schema from the signature (types, enums, defaults), so the
same definitions can be handed to an LLM for tool calling in Phase 3.

``call_tool(name, arguments)`` is the single entry point agents use. It

* normalises forgiving inputs (``"2026-07"``, ``"July 2026"`` -> ``"2026-07-31"``;
  ``"cre"`` -> ``"CRE"``) because small models often get formats slightly wrong,
* rejects bad arguments with a structured error that lists the allowed values
  (so the model can recover on the next turn),
* executes the tool and makes the result JSON-safe,
* returns and logs a provenance record: tool name, arguments, timestamp,
  duration, status and a hash of the result (Rule 4: every number is traceable).

Tools must never read ``data/ground_truth`` — see tests/test_tools.py.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import pickle
import re
import time
import types
import typing
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from src.cecl.attribution import AllowanceAttributor, AttributionResult
from src.cecl.engine import ECLEngine
from src.data.loader import DataStore
from src.utils import config

FRAMEWORKS = ["CECL", "IFRS9"]
NUMBER_NEUTRAL_CONFIGS = {"agents", "reconciliation"}   # editing these never changes cached analytics
Framework = Literal["CECL", "IFRS9"]

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ToolError(Exception):
    """Raised by tools for problems the caller can fix (bad id, bad combo)."""

    def __init__(self, message: str, error_type: str = "invalid_arguments", hint: Any = None):
        super().__init__(message)
        self.error_type = error_type
        self.hint = hint


# --------------------------------------------------------------------------- #
# Shared context (data, engine, cached analytics)
# --------------------------------------------------------------------------- #


class ToolContext:
    """Holds the data store, engine and cached analytics shared by all tools.

    Attribution and Shapley results are expensive (11-35 s) and deterministic,
    so they are cached in memory and on disk. The disk key includes the data
    files' sizes/mtimes and the config files' contents, so regenerating data
    or editing methodology invalidates the cache automatically.
    """

    def __init__(self, data_dir: Path | str | None = None, cache_dir: Path | str | None = None,
                 disk_cache: bool = True):
        self.store = DataStore(data_dir)
        self.engine = ECLEngine()
        self.attributor = AllowanceAttributor(self.store, self.engine)
        self.cache_dir = Path(cache_dir) if cache_dir else config.path("outputs") / "cache"
        self.disk_cache = disk_cache
        self._mem: dict[str, Any] = {}

    # ------------------------------------------------------------ metadata
    @cached_property
    def dates(self) -> list[str]:
        return self.store.available_dates()

    @cached_property
    def portfolios(self) -> list[str]:
        return sorted(self.store.loans["portfolio"].dropna().unique().tolist())

    @cached_property
    def product_types(self) -> list[str]:
        return sorted(self.store.loans["product_type"].dropna().unique().tolist())

    @cached_property
    def macro_variables(self) -> list[str]:
        return list(config.methodology()["macro"]["variables"])

    @cached_property
    def scenarios(self) -> list[str]:
        return list(config.methodology()["macro"]["scenarios"])

    @cached_property
    def model_ids(self) -> list[str]:
        return sorted(self.store.model_metadata["model_id"].tolist())

    @cached_property
    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.store.dir.glob("*")):
            stat = p.stat()
            h.update(f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        for p in sorted(config.config_dir().glob("*.yaml")):
            if p.stem in NUMBER_NEUTRAL_CONFIGS:      # don't change any cached number
                continue
            h.update(p.read_bytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------ caching
    def _cached(self, key: str, compute: Callable[[], Any]) -> Any:
        if key in self._mem:
            return self._mem[key]
        path = self.cache_dir / f"{key}_{self._fingerprint}.pkl"
        if self.disk_cache and path.exists():
            try:
                with path.open("rb") as fh:
                    self._mem[key] = pickle.load(fh)
                return self._mem[key]
            except Exception:  # corrupt/old cache -> recompute
                path.unlink(missing_ok=True)
        value = compute()
        self._mem[key] = value
        if self.disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(value, fh)
            tmp.replace(path)
        return value

    def attribution(self, previous_date: str, current_date: str, framework: str = "CECL",
                    order: str = "default") -> AttributionResult:
        def compute():
            if order == "reverse":
                steps = [s["id"] for s in self.attributor.cfg["steps"]][::-1]
                return self.attributor.run(previous_date, current_date, framework,
                                           step_order=steps, macro_order=self.macro_variables[::-1])
            return self.attributor.run(previous_date, current_date, framework)
        return self._cached(f"attr_{order}_{framework}_{previous_date}_{current_date}", compute)

    def macro_shapley(self, previous_date: str, current_date: str, framework: str = "CECL") -> dict[str, Any]:
        return self._cached(f"macroshap_{framework}_{previous_date}_{current_date}",
                            lambda: self.attributor.macro_shapley(self.attribution(previous_date, current_date, framework)))

    def driver_shapley(self, previous_date: str, current_date: str, framework: str = "CECL") -> dict[str, Any]:
        return self._cached(f"drvshap_{framework}_{previous_date}_{current_date}",
                            lambda: self.attributor.driver_shapley(previous_date, current_date, framework))

    # ------------------------------------------------------------ data helpers
    def summary_row(self, as_of_date: str, framework: str) -> pd.Series:
        s = self.store.summary
        row = s[(pd.to_datetime(s["as_of_date"]) == pd.Timestamp(as_of_date)) & (s["framework"] == framework)]
        if row.empty:
            raise ToolError(f"No official {framework} run for {as_of_date}", "not_found",
                            {"available_dates": self.dates})
        return row.iloc[0]

    def results(self, as_of_date: str, framework: str) -> pd.DataFrame:
        return self.store.results_at(as_of_date, framework)


_CONTEXT: ToolContext | None = None


def get_context() -> ToolContext:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = ToolContext()
    return _CONTEXT


def set_context(ctx: ToolContext | None) -> None:
    """Point every tool at a different data directory (tests, other books)."""
    global _CONTEXT
    _CONTEXT = ctx


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #


def to_jsonable(obj: Any, ndigits: int = 6) -> Any:
    """Recursively convert numpy/pandas objects to plain JSON types.

    Floats are rounded to ``ndigits`` decimals (dollar amounts keep cents;
    rates keep enough precision). NaN/inf become None.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, ndigits)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict("records"), ndigits)
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict(), ndigits)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), ndigits)
    if obj is pd.NaT:
        return None
    return str(obj)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return to_jsonable(df.reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

DATE_PARAMS = {"as_of_date", "previous_date", "current_date"}


@dataclass
class ToolSpec:
    name: str
    func: Callable[..., dict[str, Any]]
    description: str
    agents: list[str]
    category: str
    slow: bool = False

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.func)

    def json_schema(self, ctx: ToolContext | None = None) -> dict[str, Any]:
        """OpenAI-style function schema (also accepted by NIM / vLLM)."""
        props: dict[str, Any] = {}
        required: list[str] = []
        hints = typing.get_type_hints(self.func)
        for pname, p in self.signature.parameters.items():
            spec = _type_schema(hints.get(pname, str))
            enum = _enum_for(pname, ctx)
            if enum is not None and spec.get("type") in ("string", ["string", "null"]):
                spec["enum"] = enum + ([None] if spec.get("type") == ["string", "null"] else [])
            doc = PARAM_DOCS.get(pname)
            if doc:
                spec["description"] = doc
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                spec["default"] = p.default
            props[pname] = spec
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": {"type": "object", "properties": props, "required": required,
                                            "additionalProperties": False}}}


REGISTRY: dict[str, ToolSpec] = {}

PARAM_DOCS = {
    "as_of_date": "Reporting date (month-end), e.g. '2026-08-31'.",
    "previous_date": "Earlier reporting date (month-end), e.g. '2026-07-31'.",
    "current_date": "Later reporting date (month-end), e.g. '2026-08-31'.",
    "framework": "Allowance framework. CECL (US GAAP lifetime) is the default.",
    "portfolio": "Optional portfolio filter. Omit for the whole book.",
    "n": "Number of rows to return.",
    "loan_id": "Loan identifier, e.g. 'LN_0048760'.",
    "counterparty_id": "Borrower identifier, e.g. 'CP_040408'.",
    "model_id": "Model identifier, e.g. 'CRE_PD_v4'.",
    "variable": "Macroeconomic variable.",
    "scenario": "Optional macro scenario filter.",
}


def _is_union(origin: Any) -> bool:
    return origin is typing.Union or origin is types.UnionType


def _type_schema(tp: Any) -> dict[str, Any]:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if _is_union(origin):
        non_null = [a for a in args if a is not type(None)]
        inner = _type_schema(non_null[0]) if len(non_null) == 1 else {}
        if "type" in inner and isinstance(inner["type"], str):
            inner["type"] = [inner["type"], "null"]
        return inner
    if origin is Literal:
        return {"type": "string", "enum": list(args)}
    if origin in (list, typing.List):
        return {"type": "array", "items": _type_schema(args[0]) if args else {}}
    if origin in (dict, typing.Dict):
        return {"type": "object", "additionalProperties": _type_schema(args[1]) if len(args) > 1 else {}}
    return {int: {"type": "integer"}, float: {"type": "number"}, bool: {"type": "boolean"},
            str: {"type": "string"}}.get(tp, {"type": "string"})


def _enum_for(pname: str, ctx: ToolContext | None) -> list[str] | None:
    if ctx is None:
        return None
    try:
        return {
            "as_of_date": ctx.dates, "previous_date": ctx.dates, "current_date": ctx.dates,
            "portfolio": ctx.portfolios, "variable": ctx.macro_variables, "scenario": ctx.scenarios,
        }.get(pname)
    except FileNotFoundError:
        return None


def tool(category: str, agents: list[str], slow: bool = False):
    """Register a deterministic analytics function as an agent-callable tool."""
    def deco(func: Callable[..., dict[str, Any]]):
        doc = inspect.getdoc(func) or ""
        description = doc.split("\n\n")[0].replace("\n", " ").strip()
        REGISTRY[func.__name__] = ToolSpec(func.__name__, func, description, agents, category, slow)
        return func
    return deco


def list_tools(agent: str | None = None) -> list[str]:
    return [n for n, s in REGISTRY.items() if agent is None or agent in s.agents or "all" in s.agents]


def tool_schemas(agent: str | None = None, ctx: ToolContext | None = None) -> list[dict[str, Any]]:
    ctx = ctx or get_context()
    return [REGISTRY[n].json_schema(ctx) for n in list_tools(agent)]


# --------------------------------------------------------------------------- #
# Argument normalisation & validation
# --------------------------------------------------------------------------- #

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                       "sep", "oct", "nov", "dec"], start=1)}


def normalize_date(value: Any, ctx: ToolContext) -> str:
    """Map '2026-08-31', '2026-08', '08/31/2026', 'August 2026', 'aug' to an available date."""
    if value is None:
        raise ToolError("Date is required", hint={"available_dates": ctx.dates})
    text = str(value).strip()
    if text in ctx.dates:
        return text
    tokens = [t for t in re.split(r"[\s,/._-]+", text.lower()) if t]
    month = next((num for t in tokens if t.isalpha() for name, num in _MONTHS.items() if t.startswith(name)), None)
    if month is not None:                       # "August 2026", "aug", "Jul-26"
        years = [int(t) for t in tokens if t.isdigit() and len(t) == 4]
        candidates = [d for d in ctx.dates if int(d[5:7]) == month and (not years or int(d[:4]) == years[0])]
        if len(candidates) == 1:
            return candidates[0]
    else:                                       # "2026-08", "08/31/2026", "2026-08-30"
        try:
            ts = pd.Timestamp(text)
            for d in ctx.dates:
                if (int(d[:4]), int(d[5:7])) == (ts.year, ts.month):
                    return d
        except (ValueError, TypeError):
            pass
    raise ToolError(f"No reporting date matches '{value}'", hint={"available_dates": ctx.dates})


def _normalize_choice(value: Any, allowed: list[str], what: str) -> str:
    if value in allowed:
        return value
    key = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {"commercial real estate": "CRE", "c and i": "C&I", "ci": "C&I", "c&i": "C&I",
               "commercial and industrial": "C&I", "cards": "Credit Card", "credit cards": "Credit Card",
               "card": "Credit Card", "mortgages": "Mortgage", "auto loan": "Auto", "auto loans": "Auto",
               "consumer loans": "Consumer", "personal loan": "Consumer", "personal loans": "Consumer",
               "unemployment": "unemployment_rate", "gdp": "gdp_growth", "rates": "interest_rate",
               "interest rates": "interest_rate", "hpi": "hpi_growth", "house prices": "hpi_growth",
               "cre prices": "cre_price_growth", "cre price index": "cre_price_growth",
               "severe": "severe_adverse", "severely adverse": "severe_adverse", "base": "baseline"}
    for a in allowed:
        if key == a.lower().replace("_", " ").replace("-", " "):
            return a
    if key in aliases and aliases[key] in allowed:
        return aliases[key]
    raise ToolError(f"Unknown {what} '{value}'", hint={f"allowed_{what}s": allowed})


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    params = spec.signature.parameters
    hints = typing.get_type_hints(spec.func)
    unknown = set(arguments) - set(params)
    if unknown:
        raise ToolError(f"Unexpected argument(s) {sorted(unknown)} for {spec.name}",
                        hint={"allowed_arguments": list(params)})
    out: dict[str, Any] = {}
    for pname, p in params.items():
        if pname not in arguments or arguments[pname] is None:
            if p.default is inspect.Parameter.empty:
                raise ToolError(f"Missing required argument '{pname}' for {spec.name}",
                                hint={"required": [n for n, q in params.items() if q.default is inspect.Parameter.empty],
                                      **({"available_dates": ctx.dates} if pname in DATE_PARAMS else {})})
            out[pname] = p.default
            continue
        v = arguments[pname]
        tp = hints.get(pname, str)
        if pname in DATE_PARAMS:
            v = normalize_date(v, ctx)
        elif pname == "framework":
            v = _normalize_choice(str(v).upper().replace(" ", ""), FRAMEWORKS, "framework")
        elif pname == "portfolio":
            v = _normalize_choice(v, ctx.portfolios, "portfolio")
        elif pname == "variable":
            v = _normalize_choice(v, ctx.macro_variables, "variable")
        elif pname == "scenario":
            v = _normalize_choice(v, ctx.scenarios, "scenario")
        else:
            v = _coerce(pname, v, tp)
        out[pname] = v
    if "previous_date" in out and "current_date" in out and out["previous_date"] >= out["current_date"]:
        raise ToolError("previous_date must be earlier than current_date",
                        hint={"available_dates": ctx.dates})
    return out


def _coerce(pname: str, v: Any, tp: Any) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    base = [a for a in args if a is not type(None)][0] if _is_union(origin) else tp
    if typing.get_origin(base) is Literal:
        allowed = list(typing.get_args(base))
        return _normalize_choice(v, allowed, pname)
    try:
        if base is int:
            if isinstance(v, bool):
                raise ValueError
            return int(float(v))
        if base is float:
            return float(v)
        if base is bool:
            if isinstance(v, str):
                if v.lower() in ("true", "yes", "1"):
                    return True
                if v.lower() in ("false", "no", "0"):
                    return False
                raise ValueError
            return bool(v)
        if typing.get_origin(base) is dict:
            if isinstance(v, str):
                v = json.loads(v)
            return {str(k): float(x) for k, x in dict(v).items()}
        if base is str:
            return str(v).strip()
    except (ValueError, TypeError, json.JSONDecodeError):
        raise ToolError(f"Argument '{pname}' has the wrong type (got {type(v).__name__} {v!r})",
                        hint={"expected_type": _type_schema(tp)})
    return v


# --------------------------------------------------------------------------- #
# Execution with provenance
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallRecord:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]           # as requested by the caller
    parameters: dict[str, Any]          # after normalisation (what actually ran)
    timestamp: str
    duration_ms: float
    status: Literal["ok", "error"]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    result_sha256: str | None = None
    caller: str | None = None
    normalized: bool = False            # True if arguments were repaired
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def provenance(self) -> dict[str, Any]:
        """Compact citation an agent can attach to a finding."""
        return {"call_id": self.call_id, "tool_name": self.tool_name, "parameters": self.parameters,
                "timestamp": self.timestamp, "result_sha256": self.result_sha256}


class ToolCallLog:
    """In-memory list of calls plus an optional JSONL file (for the demo UI and evaluation)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.records: list[ToolCallRecord] = []

    def add(self, rec: ToolCallRecord) -> None:
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            slim = {k: v for k, v in rec.to_dict().items() if k != "result"}
            with self.path.open("a") as fh:
                fh.write(json.dumps(slim, default=str) + "\n")

    def stats(self) -> dict[str, Any]:
        n = len(self.records)
        ok = sum(r.status == "ok" for r in self.records)
        by_type: dict[str, int] = {}
        for r in self.records:
            if r.error:
                by_type[r.error["error_type"]] = by_type.get(r.error["error_type"], 0) + 1
        return {"calls": n, "succeeded": ok, "failed": n - ok,
                "success_rate": ok / n if n else None,
                "repaired_arguments": sum(r.normalized for r in self.records),
                "errors_by_type": by_type,
                "mean_duration_ms": float(np.mean([r.duration_ms for r in self.records])) if n else None}


DEFAULT_LOG = ToolCallLog()


def call_tool(name: str, arguments: dict[str, Any] | str | None = None, *, caller: str | None = None,
              log: ToolCallLog | None = DEFAULT_LOG, ctx: ToolContext | None = None) -> ToolCallRecord:
    """Validate, execute and record one tool call. Never raises for caller mistakes."""
    ctx = ctx or get_context()
    started = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    raw = arguments
    rec = ToolCallRecord(call_id=uuid.uuid4().hex[:12], tool_name=str(name), arguments={}, parameters={},
                         timestamp=ts, duration_ms=0.0, status="error", caller=caller)
    try:
        if isinstance(arguments, str):           # models sometimes send a JSON string
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolError(f"Arguments are not valid JSON: {exc.msg}", "malformed_arguments",
                                {"example": {"previous_date": ctx.dates[0], "current_date": ctx.dates[-1]}})
        arguments = dict(arguments or {})
        rec.arguments = to_jsonable(arguments)
        if name not in REGISTRY:
            close = [t for t in REGISTRY if name and (name.lower() in t or t in name.lower())]
            raise ToolError(f"Unknown tool '{name}'", "unknown_tool",
                            {"did_you_mean": close[:5], "available_tools": sorted(REGISTRY)})
        spec = REGISTRY[name]
        params = validate_arguments(spec, arguments, ctx)
        rec.parameters = to_jsonable(params)
        rec.normalized = any(str(arguments.get(k)) != str(v) for k, v in params.items() if k in arguments)
        result = spec.func(**params)
        result = to_jsonable(result)
        rec.result = result
        rec.result_sha256 = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
        rec.status = "ok"
    except ToolError as exc:
        rec.error = {"error_type": exc.error_type, "message": str(exc), "hint": to_jsonable(exc.hint)}
    except Exception as exc:  # genuine bug or data problem: report, don't crash the agent loop
        rec.error = {"error_type": "execution_error", "message": f"{type(exc).__name__}: {exc}", "hint": None}
    rec.duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    if raw is not None and not rec.arguments and isinstance(raw, dict):
        rec.arguments = to_jsonable(raw)
    if log is not None:
        log.add(rec)
    return rec


# --------------------------------------------------------------------------- #
# Small helpers shared by tool modules
# --------------------------------------------------------------------------- #


def ew_mean(values: pd.Series, weights: pd.Series) -> float | None:
    """Weighted mean ignoring NaN values (None if no weight)."""
    m = values.notna() & weights.notna()
    w = weights[m].sum()
    return float((values[m] * weights[m]).sum() / w) if w else None


def pct_change(prev: float | None, curr: float | None) -> float | None:
    if prev in (None, 0) or curr is None:
        return None
    return 100.0 * (curr - prev) / abs(prev)


def driver_labels() -> dict[str, str]:
    return dict(config.methodology()["attribution"]["drivers"])


def step_columns(attr: AttributionResult) -> list[str]:
    return [s["id"] for s in attr.steps]


def driver_of_column(attr: AttributionResult) -> dict[str, str]:
    return {s["id"]: s["driver"] for s in attr.steps}
