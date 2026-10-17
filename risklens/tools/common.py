"""Helpers shared by tool modules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from risklens import config
from risklens.data.store import DataStore, get_store
from risklens.tools.registry import ToolError


def store() -> DataStore:
    return get_store()


def check_order(prev: str, cur: str) -> None:
    if prev == cur:
        raise ToolError("previous_date and current_date are the same.", {"available_dates": store().dates})
    if prev > cur:
        raise ToolError("previous_date must be earlier than current_date.",
                        {"swap_to": {"previous_date": cur, "current_date": prev}})


def loans(as_of: str, portfolio: str | None = None) -> pd.DataFrame:
    df = store().loans(as_of)
    return df if portfolio is None else df[df["portfolio"] == portfolio]


def wavg(values, weights) -> float | None:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = ~np.isnan(v) & ~np.isnan(w)
    if not m.any() or w[m].sum() == 0:
        return None
    return float(np.average(v[m], weights=w[m]))


def pct(a: float | None, b: float | None) -> float | None:
    if a in (None, 0) or b is None:
        return None
    return (b - a) / abs(a) * 100


def portfolios() -> list[str]:
    return list(config.methodology()["portfolios"])


def driver_labels() -> dict[str, str]:
    return dict(config.methodology()["attribution"]["drivers"])
