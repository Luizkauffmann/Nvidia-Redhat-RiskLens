"""Numeric grounding check: every number in an answer must come from a tool result.

The agent may round ($19,238,021.81 -> "$19.24M") and change units
(0.0521 -> "5.2%", 52.8 bps -> "53 bps"), but it may not compute new figures.
For each number in the answer we accept it if some number in this run's tool
results (or the question) matches after unit conversion, within the rounding
precision the answer used.

Ignored: dates, IDs (CP_040408, CRE_PD_v4), list numbering, years, and small
bare integers (counts and ranks like "top 5", "3 notches"); see ``_IGNORE_BARE_INT_BELOW``.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

_IGNORE_BARE_INT_BELOW = 100

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_MONTH_YEAR = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b", re.I)
_QUARTER = re.compile(r"\b(?:Q[1-4]\s*\d{2,4}|\d{4}\s*Q[1-4])\b", re.I)
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s+")
_NUM = re.compile(
    r"(?<![\w.])(?P<neg>[-−–]\s?)?(?P<cur>\$\s?)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>%|percentage points|pp\b|bps\b|bp\b|basis points|[KkMB]\b|million\b|billion\b|thousand\b|mn\b|bn\b))?",
)
_SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "M": 1e6, "million": 1e6, "mn": 1e6, "B": 1e9, "billion": 1e9, "bn": 1e9}
_NUM_IN_STR = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class NumberMention:
    text: str
    value: float
    kind: str              # money | percent | bps | pp | number
    decimals: int
    grounded: bool | None = None
    matched: float | None = None


@dataclass
class GroundingReport:
    status: str                                     # PASS | FAIL | NO_NUMBERS
    checked: int
    grounded: int
    ungrounded: list[dict[str, Any]] = field(default_factory=list)
    mentions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_numbers(text: str) -> list[NumberMention]:
    clean = _LIST_MARK.sub(" ", text)
    for pat in (_DATE, _MONTH_YEAR, _QUARTER):
        clean = pat.sub(" ", clean)
    out = []
    for m in _NUM.finditer(clean):
        raw = m.group("num")
        unit = (m.group("unit") or "").strip()
        cur = bool(m.group("cur"))
        val = float(raw.replace(",", ""))
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if m.group("neg"):
            val = -val
        if unit in ("%",):
            kind = "percent"
        elif unit in ("percentage points", "pp"):
            kind = "pp"
        elif unit in ("bps", "bp", "basis points"):
            kind = "bps"
        elif cur or unit in _SCALE:
            kind = "money"
            val *= _SCALE.get(unit, 1.0)
        else:
            kind = "number"
            if dec == 0 and (abs(val) < _IGNORE_BARE_INT_BELOW or 1900 <= abs(val) <= 2100):
                continue
        out.append(NumberMention(text=m.group(0).strip(), value=val, kind=kind, decimals=dec))
    return out


def _leaves(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            yield float(obj)
    elif isinstance(obj, str):
        for s in _NUM_IN_STR.findall(obj):
            try:
                yield float(s)
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)


def _tolerance(m: NumberMention) -> float:
    unit = {"money": _scale_of(m.text)}.get(m.kind, 1.0)
    return 0.5 * 10 ** (-m.decimals) * unit * 1.0001 + 1e-9


def _scale_of(text: str) -> float:
    for u, s in sorted(_SCALE.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\d\s?{re.escape(u)}\b", text):
            return s
    return 1.0


def check_grounding(answer: str, sources: Iterable[Any]) -> GroundingReport:
    """Check every number in ``answer`` against the numbers found in ``sources`` (tool results, question)."""
    vals = sorted({abs(v) for src in sources for v in _leaves(src)})
    import bisect

    def near(target: float, tol: float) -> float | None:
        t = abs(target)
        i = bisect.bisect_left(vals, t - tol)
        if i < len(vals) and vals[i] <= t + tol:
            return vals[i]
        return None

    mentions = extract_numbers(answer)
    for m in mentions:
        tol = _tolerance(m)
        if m.kind == "money":
            cands = [(m.value, tol)]
        elif m.kind == "percent":
            cands = [(m.value, tol), (m.value / 100.0, tol / 100.0)]           # 19.4% <- 19.42 or 0.194
        elif m.kind == "pp":
            cands = [(m.value, tol), (m.value / 100.0, tol / 100.0)]
        elif m.kind == "bps":
            cands = [(m.value, tol), (m.value / 1e4, tol / 1e4)]
        else:
            cands = [(m.value, tol), (m.value / 100.0, tol / 100.0)]
        for target, t in cands:
            hit = near(target, t)
            if hit is not None:
                m.grounded, m.matched = True, hit
                break
        else:
            m.grounded = False
    ung = [asdict(m) for m in mentions if not m.grounded]
    status = "NO_NUMBERS" if not mentions else ("PASS" if not ung else "FAIL")
    return GroundingReport(status=status, checked=len(mentions), grounded=len(mentions) - len(ung),
                           ungrounded=ung, mentions=[asdict(m) for m in mentions])
