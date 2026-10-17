"""Synthetic data integrity and reproducibility."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.generator import DemoDataGenerator
from src.utils import config

from tests.conftest import CURR, PREV


def test_dimensions_and_keys(store):
    loans = store.loans
    assert store.available_dates() == [PREV, CURR]
    assert len(store.loans_at(PREV)) == config.settings()["n_loans"]
    assert not loans.duplicated(["as_of_date", "loan_id"]).any()
    cps = store.counterparties
    assert not cps.duplicated(["as_of_date", "counterparty_id"]).any()
    for d in (PREV, CURR):
        assert store.loans_at(d)["counterparty_id"].isin(store.counterparties_at(d)["counterparty_id"]).all()


def test_taxonomy_matches_methodology(store):
    prods = config.methodology()["products"]
    assert set(store.loans["product_type"]) == set(prods)
    for p, spec in prods.items():
        assert set(store.loans.loc[store.loans.product_type == p, "portfolio"]) == {spec["portfolio"]}


def test_balances_and_terms_valid(store):
    l = store.loans
    assert (l["current_balance"] >= 0).all()
    termed = l["remaining_term_months"].notna()
    assert (l.loc[termed, "remaining_term_months"] >= 1).all()
    assert (l.loc[l.product_type == "Credit Card", "remaining_term_months"].isna()).all()


def test_no_model_version_change(store):
    a = store.model_assignments_at(PREV).set_index("product_type")[["pd_model_id", "lgd_model_id", "ead_model_id"]]
    b = store.model_assignments_at(CURR).set_index("product_type")[["pd_model_id", "lgd_model_id", "ead_model_id"]]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())
    cre = store.loans[store.loans.portfolio == "CRE"]
    assert set(cre["pd_model_id"]) == {"CRE_PD_v4"}


def test_scenarios_weights_and_macro_direction(store):
    for d in (PREV, CURR):
        assert sum(store.weights_at(d).values()) == pytest.approx(1.0)
    assert store.weights_at(CURR)["severe_adverse"] > store.weights_at(PREV)["severe_adverse"]
    m0, m1 = store.macro_at(PREV), store.macro_at(CURR)
    for s in ("baseline", "adverse", "severe_adverse"):
        assert m1.loc[m1.scenario == s, "unemployment_rate"].mean() > m0.loc[m0.scenario == s, "unemployment_rate"].mean()
        assert m1.loc[m1.scenario == s, "cre_price_growth"].mean() < m0.loc[m0.scenario == s, "cre_price_growth"].mean()


def test_injected_cre_story_present(store, ground_truth):
    g = ground_truth["design"]
    assert len(g["deteriorated_cre_borrowers"]) == 20
    new = store.loans_at(CURR).loc[~store.loans_at(CURR).loan_id.isin(store.loans_at(PREV).loan_id)]
    assert new.loc[new.portfolio == "CRE", "current_balance"].sum() == pytest.approx(g["new_cre_exposure"], abs=0.01)


def test_generation_is_reproducible():
    a = DemoDataGenerator().build_july()[0]
    b = DemoDataGenerator().build_july()[0]
    pd.testing.assert_frame_equal(a, b)
