"""Macroeconomic scenario handling: forecast paths -> annual deviations.

The scenario dataset holds quarterly forecast paths per (as_of_date, scenario).
The methodology uses the path for the reasonable & supportable (R&S) period,
then reverts straight-line to the long-run anchor (deviation 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quarterly_deviation(path: pd.DataFrame, variables: list[str], anchors: dict[str, float],
                        rs_quarters: int, reversion_quarters: int, n_quarters: int) -> np.ndarray:
    """Return (n_quarters, n_vars) deviations from anchor after R&S + reversion."""
    path = path.sort_values("forecast_quarter")
    if len(path) < rs_quarters:
        raise ValueError(f"Scenario path has {len(path)} quarters; R&S needs {rs_quarters}")
    anchor = np.array([anchors[v] for v in variables], dtype=float)
    raw = path[variables].to_numpy(dtype=float)[:rs_quarters] - anchor
    out = np.zeros((n_quarters, len(variables)))
    out[:rs_quarters] = raw
    last = raw[rs_quarters - 1]
    for k in range(1, reversion_quarters + 1):
        q = rs_quarters + k - 1
        if q < n_quarters:
            out[q] = last * (1.0 - k / reversion_quarters)
    return out


def annual_deviation(path: pd.DataFrame, variables: list[str], anchors: dict[str, float],
                     rs_quarters: int, reversion_quarters: int, n_years: int) -> np.ndarray:
    """Average quarterly deviations into (n_years, n_vars) annual deviations."""
    q = quarterly_deviation(path, variables, anchors, rs_quarters, reversion_quarters, n_years * 4)
    return q.reshape(n_years, 4, len(variables)).mean(axis=1)


def scenario_annual_deviations(macro: pd.DataFrame, macro_cfg: dict, n_years: int) -> dict[str, np.ndarray]:
    """Annual deviation matrix for every scenario in a single-vintage macro frame."""
    out = {}
    for scenario, path in macro.groupby("scenario"):
        out[scenario] = annual_deviation(
            path, macro_cfg["variables"], macro_cfg["long_run_anchor"],
            macro_cfg["reasonable_supportable_quarters"], macro_cfg["reversion_quarters"], n_years)
    return out
