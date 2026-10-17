"""Engine correctness: hand-calculated toy cases + structural properties."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cecl.engine import ECLEngine, EngineInputs
from src.cecl.scenarios import quarterly_deviation
from src.utils import config

from tests.conftest import CURR, PREV

METHOD = config.methodology()
ANCHOR = METHOD["macro"]["long_run_anchor"]
VARS = METHOD["macro"]["variables"]


def flat_macro(dev: dict | None = None) -> pd.DataFrame:
    """All scenarios at the long-run anchor (+ optional constant deviation)."""
    rows = []
    for s in METHOD["macro"]["scenarios"]:
        for q in range(1, 13):
            rows.append({"scenario": s, "forecast_quarter": q,
                         **{v: ANCHOR[v] + (dev or {}).get(v, 0.0) for v in VARS}})
    return pd.DataFrame(rows)


def toy_loan(**over) -> pd.DataFrame:
    base = {
        "loan_id": "T1", "counterparty_id": "C1", "portfolio": "CRE", "product_type": "Commercial Real Estate",
        "current_balance": 1_000_000.0, "undrawn_amount": 0.0, "remaining_term_months": 24.0,
        "interest_rate": 0.06, "rating_notch": 10, "dscr": 1.45, "ltv": 0.62, "watchlist_flag": 0,
        "days_past_due": 0.0, "default_flag": 0, "origination_pd": 0.0077, "origination_rating_notch": 10,
        "pd_model_id": "CRE_PD_v4", "lgd_model_id": "CRE_LGD_v3", "ead_model_id": "EAD_BAL_v1",
    }
    base.update(over)
    return pd.DataFrame([base])


WEIGHTS = {"baseline": 0.6, "adverse": 0.3, "severe_adverse": 0.1}


def test_toy_bullet_loan_matches_hand_calculation():
    """2-year bullet CRE loan at feature centers, macro at anchor: closed form."""
    m = config.models()
    pd_annual = 1 / (1 + np.exp(-m["pd_models"]["CRE_PD_v4"]["intercept"]))
    lgd = m["lgd_models"]["CRE_LGD_v3"]["base"]
    expected_cecl = 1_000_000 * lgd * (1 - (1 - pd_annual) ** 2)          # no discounting
    expected_12m = 1_000_000 * lgd * pd_annual / 1.06 ** 0.5               # IFRS 9 stage 1, EIR discount

    inp = EngineInputs("2026-07-31", toy_loan(), flat_macro(), WEIGHTS)
    cecl = ECLEngine().run(inp, "CECL").loans.iloc[0]
    ifrs = ECLEngine().run(inp, "IFRS9").loans.iloc[0]
    assert cecl["ecl"] == pytest.approx(expected_cecl, rel=1e-10)
    assert ifrs["ifrs9_stage"] == 1
    assert ifrs["ecl"] == pytest.approx(expected_12m, rel=1e-10)


def test_defaulted_loan_reserved_at_lgd_times_ead():
    inp = EngineInputs("2026-07-31", toy_loan(days_past_due=120.0, default_flag=1), flat_macro(), WEIGHTS)
    row = ECLEngine().run(inp, "CECL").loans.iloc[0]
    assert row["defaulted"]
    assert row["ecl"] == pytest.approx(1_000_000 * row["lgd"], rel=1e-12)


def test_worse_macro_raises_ecl_and_reverts_after_rs_period():
    eng = ECLEngine()
    good = eng.run(EngineInputs("d", toy_loan(), flat_macro(), WEIGHTS), "CECL").total_ecl
    bad = eng.run(EngineInputs("d", toy_loan(), flat_macro({"unemployment_rate": 2.0}), WEIGHTS), "CECL").total_ecl
    assert bad > good
    path = flat_macro({"unemployment_rate": 2.0})
    q = quarterly_deviation(path[path.scenario == "baseline"], VARS, ANCHOR, 8, 4, 20)
    assert np.allclose(q[:8, 0], 2.0) and np.allclose(q[12:], 0.0)
    assert q[8, 0] == pytest.approx(1.5) and q[11, 0] == pytest.approx(0.0)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ECLEngine().run(EngineInputs("d", toy_loan(), flat_macro(), {"baseline": 0.5, "adverse": 0.3, "severe_adverse": 0.1}))


def test_cecl_card_undrawn_not_reserved_but_ifrs9_is():
    card = toy_loan(portfolio="Credit Card", product_type="Credit Card", current_balance=2_000.0, undrawn_amount=8_000.0,
                    remaining_term_months=np.nan, utilization=0.2, fico=715, dti=0.33,
                    pd_model_id="CC_PD_v5", lgd_model_id="CC_LGD_v3", ead_model_id="EAD_CCF_v2")
    inp = EngineInputs("d", card, flat_macro(), WEIGHTS)
    assert ECLEngine().run(inp, "CECL").loans.iloc[0]["ead"] == pytest.approx(2_000.0)
    assert ECLEngine().run(inp, "IFRS9").loans.iloc[0]["ead"] == pytest.approx(2_000.0 + 0.30 * 8_000.0)


@pytest.mark.parametrize("framework", ["CECL", "IFRS9"])
def test_portfolio_run_properties(store, engine, framework):
    res = engine.run(store.engine_inputs(PREV), framework)
    l = res.loans
    assert len(l) == 50_000
    assert (l["ecl"] >= 0).all() and l["ecl"].notna().all()
    assert (l["pd_12m"].between(0, 1)).all() and (l["lgd"].between(0, 1)).all()
    # severity ordering of scenarios
    assert res.scenario_ecl["baseline"] < res.scenario_ecl["adverse"] < res.scenario_ecl["severe_adverse"]
    assert res.allowance == pytest.approx(res.total_ecl + res.overlay_total)


def test_ifrs9_below_cecl_and_stages_consistent(store, engine):
    inp = store.engine_inputs(CURR)
    cecl, ifrs = engine.run(inp, "CECL"), engine.run(inp, "IFRS9")
    assert ifrs.total_ecl < cecl.total_ecl
    l = ifrs.loans
    assert set(l["ifrs9_stage"].unique()) <= {1, 2, 3}
    assert (l.loc[l["ifrs9_stage"] == 1, "measurement"] == "12M").all()
    assert (l.loc[l["ifrs9_stage"] > 1, "measurement"] == "LIFETIME").all()
    assert (l.loc[l["defaulted"], "ifrs9_stage"] == 3).all()


def test_engine_is_deterministic(store, engine):
    a = engine.run(store.engine_inputs(PREV), "CECL").allowance
    b = engine.run(store.engine_inputs(PREV), "CECL").allowance
    assert a == b


@pytest.mark.parametrize("framework", ["CECL", "IFRS9"])
def test_macro_sweep_matches_full_runs(store, engine, framework):
    """The fast macro path (used by Shapley and sensitivities) is identical to full runs."""
    import dataclasses
    inp = store.engine_inputs(CURR)
    other = store.macro_at(PREV)
    sweep = engine.macro_sweep(inp, [inp.macro, other], framework)
    full_cur = engine.run(inp, framework)
    full_other = engine.run(dataclasses.replace(inp, macro=other), framework)
    assert sweep[0]["allowance"] == pytest.approx(full_cur.allowance, abs=1e-6)
    assert sweep[1]["allowance"] == pytest.approx(full_other.allowance, abs=1e-6)
    assert np.allclose(sweep[0]["loan_ecl"], full_cur.loans["ecl"].to_numpy(), atol=1e-9)
