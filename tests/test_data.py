import numpy as np

from risklens import config
from risklens.data.generator import generate
from tests.conftest import CUR, PREV


def test_dimensions(store):
    j, a = store.loans(PREV), store.loans(CUR)
    assert len(j) == config.settings()["n_loans"] == 50_000
    assert set(j["portfolio"]) == set(config.methodology()["portfolios"])
    assert j["loan_id"].is_unique and a["loan_id"].is_unique
    assert set(j["counterparty_id"]) | set(a["counterparty_id"]) <= set(store.counterparties["counterparty_id"])
    required = ["as_of_date", "loan_id", "counterparty_id", "portfolio", "product_type", "industry", "state",
                "origination_date", "maturity_date", "current_balance", "undrawn_amount", "ead", "pd_12m", "lgd", "ecl",
                "internal_rating", "fico", "ltv", "dti", "dscr", "days_past_due", "delinquency_bucket", "default_flag",
                "pd_model_id", "lgd_model_id", "ead_model_id", "scenario_set_id"]
    assert not set(required) - set(a.columns)


def test_not_applicable_fields_are_null(store):
    j = store.loans(PREV)
    retail = j[j["segment"] == "RETAIL"]
    com = j[j["segment"] == "COMMERCIAL"]
    assert retail["dscr"].isna().all() and retail["internal_rating"].isna().all()
    assert com["fico"].isna().all() and com["internal_rating"].notna().all()


def test_injected_story(store, gt):
    a = store.loans(CUR)
    new_cre = a[a["loan_id"].isin(gt["new_cre_loan_ids"])]
    assert len(new_cre) > 0 and abs(new_cre["current_balance"].sum() - 40e6) < 1
    assert gt["cre_cohort_size"] == 20
    assert gt["cohort_pd_12m"]["july"] < 0.025 and 0.04 < gt["cohort_pd_12m"]["august"] < 0.065
    assert gt["model_changed"] is False and gt["scenario_weights_changed"] is True
    a_alw, j_alw = store.reported_allowance(CUR), store.reported_allowance(PREV)
    assert 95e6 < j_alw < 110e6 and 115e6 < a_alw < 130e6


def test_macro_worse_in_august(store):
    rs = config.methodology()["macro"]["reasonable_supportable_quarters"]
    m = store.macro[store.macro["forecast_quarter"] <= rs]
    g = m.groupby(["as_of_date", "scenario"])[["unemployment_rate", "cre_price_growth", "gdp_growth"]].mean()
    for s in ("baseline", "adverse", "severe_adverse"):
        assert g.loc[(CUR, s), "unemployment_rate"] > g.loc[(PREV, s), "unemployment_rate"]
        assert g.loc[(CUR, s), "cre_price_growth"] < g.loc[(PREV, s), "cre_price_growth"]
    w = store.weights(CUR)
    assert abs(sum(w.values()) - 1) < 1e-12


def test_generation_is_reproducible(tmp_path, store):
    out = generate(tmp_path)
    assert abs(out["summary"][PREV]["allowance"] - store.reported_allowance(PREV)) < 0.01
    assert abs(out["summary"][CUR]["allowance"] - store.reported_allowance(CUR)) < 0.01
    import pandas as pd
    a = pd.read_parquet(tmp_path / f"loans_{CUR}.parquet")
    assert np.allclose(a["ecl"].to_numpy(), store.loans(CUR)["ecl"].to_numpy())
