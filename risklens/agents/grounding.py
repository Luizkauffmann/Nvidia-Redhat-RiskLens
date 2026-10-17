"""Grounding check: every figure in an answer must come from a tool result.

Numbers in the answer are parsed with their units ($, M/K/B, %, pp) and
matched against every number in the tool results of the same run, allowing
for the rounding implied by how the figure is written. Anything that does not
match is reported as unsupported so the agent can revise or the UI can flag it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM = re.compile(
    r"(?P<neg>[-−])?\s*(?P<cur>\$)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|pp\b|percentage points?|bps|[mM]\b|[mM]illion|[kK]\b|[bB]\b|[bB]illion|bn\b)?")
_SKIP_CONTEXT = re.compile(r"(\d{4}-\d{2}-\d{2}|[A-Z]{2,}_\w*\d+|\b(?:19|20)\d{2}\b|\bv\d+\b|\bQ\d\b)")


def _flatten(obj, out: list[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        # Figures embedded in text fields (e.g. reason codes "DSCR 1.33 -> 1.16").
        for t in re.findall(r"-?\d+(?:\.\d+)?", obj):
            out.append(float(t))


@dataclass
class GroundingReport:
    checked: int = 0
    supported: int = 0
    unsupported: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict:
        return {"checked": self.checked, "supported": self.supported, "unsupported": self.unsupported, "ok": self.ok}


def _candidates(text_num: str, unit: str | None, is_cur: bool) -> tuple[list[float], float]:
    """Possible raw values for a written number and the tolerance its precision implies."""
    x = float(text_num.replace(",", ""))
    dec = len(text_num.split(".")[1]) if "." in text_num else 0
    half = 0.5 * 10 ** (-dec)
    u = (unit or "").lower()
    if u in ("m", "million"):
        return [x * 1e6], half * 1e6 + 1
    if u in ("k",):
        return [x * 1e3], half * 1e3 + 1
    if u in ("b", "billion", "bn"):
        return [x * 1e9], half * 1e9 + 1
    if u in ("%", "pp", "percentage point", "percentage points"):
        return [x, x / 100], half + 1e-9
    if u == "bps":
        return [x / 100, x / 1e4], half + 1e-9
    return [x], max(half, 0.01 if is_cur else half) + 1e-9


def check(answer: str, tool_results: list) -> GroundingReport:
    pool: list[float] = []
    for r in tool_results:
        _flatten(r, pool)
    # Signed and absolute values, plus common rescalings.
    vals = set()
    for v in pool:
        for s in (v, -v, abs(v)):
            vals.add(s)
            vals.add(s * 100)
    vals = sorted(vals)
    import bisect

    def supported(cands, tol):
        for c in cands:
            i = bisect.bisect_left(vals, c - tol)
            if i < len(vals) and vals[i] <= c + tol:
                return True
        return False

    rep = GroundingReport()
    masked = _SKIP_CONTEXT.sub(lambda mt: " " * len(mt.group(0)), answer)
    for mt in _NUM.finditer(masked):
        num, unit, cur = mt.group("num"), mt.group("unit"), bool(mt.group("cur"))
        raw = float(num.replace(",", ""))
        # Plain small integers (list numbers, counts of notches/days) are not checked.
        if not cur and not unit and "." not in num and raw < 1000:
            continue
        # List markers like "1." at line start.
        start = mt.start("num")
        if not cur and not unit and re.match(r"^\s*$", masked[max(0, masked.rfind("\n", 0, start) + 1):start]) and masked[mt.end("num"):mt.end("num") + 1] == ".":
            continue
        rep.checked += 1
        cands, tol = _candidates(num, unit, cur)
        if mt.group("neg"):
            cands = cands + [-c for c in cands]
        if supported(cands, tol):
            rep.supported += 1
        else:
            rep.unsupported.append(mt.group(0).strip())
    return rep
