"""Reconciliation: components sum to the reported change, and the residual
check is genuine (it catches input changes the attribution does not explain)."""
from __future__ import annotations

import copy

import pytest

from src.cecl.attribution import AllowanceAttributor

from tests.conftest import CURR, PREV


def test_components_reconcile_to_reported_change(attribution):
    s = attribution.summary
    assert s["explained_change"] + s["residual"] == pytest.approx(s["change"], abs=1e-6)
    assert abs(s["residual"]) < 1.0                       # dollars
    assert sum(d["impact"] for d in s["drivers"]) == pytest.approx(s["change"], abs=1.0)


def test_steps_telescope(attribution):
    prev = attribution.summary["previous_allowance"]
    for step in attribution.steps:
        prev += step["impact"]
        assert step["allowance_after"] == pytest.approx(prev, abs=1e-4)
    assert prev == pytest.approx(attribution.summary["current_allowance"], abs=1.0)


def test_loan_level_contributions_sum_to_modeled_change(attribution):
    lc = attribution.loan_contributions
    s = attribution.summary
    overlay = next(st["impact"] for st in attribution.steps if st["id"] == "overlay")
    assert lc["ecl_change"].sum() == pytest.approx(s["change"] - overlay, abs=1.0)
    by_port = attribution.portfolio_contributions["total"].sum()
    assert by_port == pytest.approx(s["change"], abs=1.0)


def test_residual_detects_unattributed_input_change(store, engine):
    """If an engine input is left out of every swap group, the residual is
    no longer ~0 — proving the reconciliation is not forced."""
    att = AllowanceAttributor(store, engine)
    att.cfg = copy.deepcopy(att.cfg)
    att.cfg["credit_fields"] = [f for f in att.cfg["credit_fields"] if f != "rating_notch"]
    res = att.run(PREV, CURR, "CECL")
    assert abs(res.summary["residual"]) > 100_000


def test_any_step_order_still_reconciles(store, engine, attribution):
    """Reversing the walk changes how interactions are assigned, never the total."""
    att = AllowanceAttributor(store, engine)
    steps = [s["id"] for s in att.cfg["steps"]][::-1]
    rev = att.run(PREV, CURR, "CECL", step_order=steps, macro_order=att.variables[::-1])
    assert abs(rev.summary["residual"]) < 1.0
    assert rev.summary["change"] == pytest.approx(attribution.summary["change"], abs=1e-6)
    assert rev.summary["method"]["step_order"] == steps


def test_shapley_values_sum_exactly(store, engine, attribution):
    att = AllowanceAttributor(store, engine)
    macro = att.macro_shapley(attribution)
    seq_macro = sum(s["impact"] for s in attribution.steps if s["step"] == "macro")
    assert sum(v["impact"] for v in macro["variables"]) == pytest.approx(seq_macro, abs=1e-3)
    assert macro["engine_runs"] == 2 ** len(att.variables)
    assert macro["by_portfolio"].to_numpy().sum() == pytest.approx(seq_macro, abs=1e-3)
    drv = att.driver_shapley(PREV, CURR, "CECL")
    assert sum(d["impact"] for d in drv["drivers"]) == pytest.approx(attribution.summary["change"], abs=1e-3)
    assert drv["by_portfolio"].to_numpy().sum() == pytest.approx(attribution.summary["change"], abs=1e-3)
    assert drv["drivers"][0]["driver"] == "macroeconomic"


def test_invalid_step_order_rejected(store, engine):
    with pytest.raises(ValueError):
        AllowanceAttributor(store, engine).run(PREV, CURR, "CECL", step_order=["macro", "overlay"])
