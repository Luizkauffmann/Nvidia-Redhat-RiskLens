"""The allowance moves for the *intended* reasons (ground truth vs attribution)."""
from __future__ import annotations

import pandas as pd

from tests.conftest import CURR, PREV


def drivers(attribution) -> dict[str, float]:
    return {d["driver"]: d["impact"] for d in attribution.summary["drivers"]}


def test_august_allowance_materially_higher(attribution):
    s = attribution.summary
    assert s["current_allowance"] > s["previous_allowance"]
    assert s["change_pct"] > 10.0


def test_primary_driver_is_macro(attribution, ground_truth):
    d = drivers(attribution)
    assert max(d, key=d.get) == ground_truth["design"]["primary_driver"] == "macroeconomic"
    assert d["macroeconomic"] / attribution.summary["change"] > 0.4


def test_driver_signs_match_story(attribution):
    d = drivers(attribution)
    assert d["credit_deterioration"] > 0
    assert d["portfolio_growth"] > 0
    assert d["paydowns_runoff"] < 0
    assert d["model_methodology"] > 0          # scenario weights moved toward severe


def test_model_change_contributes_nothing_and_weights_do(attribution, ground_truth):
    steps = {s["id"]: s["impact"] for s in attribution.steps}
    assert ground_truth["design"]["model_changed"] is False
    assert steps["model_change"] == 0.0
    assert steps["overlay"] == 0.0
    assert steps["scenario_weights"] > 0


def test_primary_macro_variable_is_unemployment(attribution, ground_truth):
    top = attribution.summary["macro_variables"][0]["variable"]
    assert top == ground_truth["design"]["primary_macro_variable"]


def test_cre_is_largest_contributing_portfolio(attribution, ground_truth):
    total = attribution.portfolio_contributions["total"]
    assert total.idxmax() == ground_truth["design"]["primary_portfolio"]
    credit = attribution.portfolio_contributions["credit_migration"]
    assert credit["CRE"] / credit.sum() > 0.5


def test_largest_borrower_and_cohort_are_top_contributors(attribution, ground_truth):
    by_cp = attribution.loan_contributions.groupby("counterparty_id")["ecl_change"].sum().sort_values(ascending=False)
    assert by_cp.index[0] == ground_truth["design"]["largest_borrower"]
    top10 = set(by_cp.index[:10])
    assert len(top10 & set(ground_truth["design"]["deteriorated_cre_borrowers"])) >= 7


def test_cohort_pd_roughly_doubles_from_about_two_percent(store, ground_truth):
    ids = ground_truth["design"]["deteriorated_cre_borrowers"]

    def ew_pd(d):
        r = store.results_at(d, "CECL")
        r = r[r.counterparty_id.isin(ids)]
        return float((r.pd_12m * r.ead).sum() / r.ead.sum())

    p0, p1 = ew_pd(PREV), ew_pd(CURR)
    assert 0.012 < p0 < 0.03
    assert 0.035 < p1 < 0.07
    assert p1 / p0 > 2.0


def test_stored_official_results_match_fresh_run(store, engine):
    for d in (PREV, CURR):
        fresh = engine.run(store.engine_inputs(d), "CECL")
        stored = store.summary
        row = stored[(stored.framework == "CECL") & (pd.to_datetime(stored.as_of_date) == pd.Timestamp(d))].iloc[0]
        assert abs(row["allowance"] - fresh.allowance) < 1e-6
