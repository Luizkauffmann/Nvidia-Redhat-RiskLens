"""Cross-check the vectorised engine against a slow, independent scalar
implementation of the documented formula on a random sample of real loans."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils import config

from tests.conftest import CURR

M = config.methodology()
MODELS = config.models()


def _dev_by_year(path, n_years):
    """Quarterly path -> annual deviation dicts, R&S then straight-line reversion."""
    mc = M["macro"]
    rs, rv = mc["reasonable_supportable_quarters"], mc["reversion_quarters"]
    path = path.sort_values("forecast_quarter")
    q = []
    for i in range(n_years * 4):
        if i < rs:
            row = path.iloc[i]
            q.append({v: row[v] - mc["long_run_anchor"][v] for v in mc["variables"]})
        elif i < rs + rv:
            k = i - rs + 1
            last = q[rs - 1]
            q.append({v: last[v] * (1 - k / rv) for v in mc["variables"]})
        else:
            q.append({v: 0.0 for v in mc["variables"]})
    return [{v: sum(q[4 * y + j][v] for j in range(4)) / 4 for v in mc["variables"]} for y in range(n_years)]


def _feature_sum(loan, spec):
    tot = 0.0
    for f, fs in spec.get("features", {}).items():
        x = loan.get(f)
        x = fs["center"] if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
        tot += fs["coef"] * (x - fs["center"])
    return tot


def scalar_ecl(loan, macro, weights, framework, stage):
    p = M["products"][loan["product_type"]]
    pdm = MODELS["pd_models"][loan["pd_model_id"]]
    lgm = MODELS["lgd_models"][loan["lgd_model_id"]]
    dpd = 0 if math.isnan(loan["days_past_due"]) else loan["days_past_due"]
    logit = pdm["intercept"] + _feature_sum(loan, pdm)
    if 30 <= dpd < 60:
        logit += M["delinquency_logit"]["30"]
    elif 60 <= dpd < 90:
        logit += M["delinquency_logit"]["60"]
    defaulted = loan["default_flag"] == 1 or dpd >= M["calculation"]["default_dpd"]

    rem = loan["remaining_term_months"]
    if p["amortization"] == "REVOLVING":
        h = p["behavioral_life_years"] if math.isnan(rem) else min(p["behavioral_life_years"], rem / 12)
    else:
        h = rem / 12
    h = min(max(h, 1 / 12), M["calculation"]["max_life_years"])
    n_years = math.ceil(h)

    ccf_spec = M["ead_models"][loan["ead_model_id"]][framework]
    ccf = ccf_spec.get(loan["product_type"], 0.0) if isinstance(ccf_spec, dict) else ccf_spec
    ead0 = loan["current_balance"] + ccf * (loan["undrawn_amount"] or 0.0)
    lgd0 = lgm["base"] + _feature_sum(loan, lgm)
    discount = M["calculation"]["discount"][framework]

    total = 0.0
    for s, w in weights.items():
        devs = _dev_by_year(macro[macro.scenario == s], n_years)
        surv, ecl_life, ecl_12 = 1.0, 0.0, 0.0
        for y in range(n_years):
            frac = min(1.0, max(0.0, h - y))
            shift = sum(pdm["macro_sensitivity"].get(v, 0.0) * devs[y][v] for v in devs[y])
            pdy = 1.0 if (defaulted and y == 0) else 1 / (1 + math.exp(-(logit + shift)))
            sf = (1 - pdy) ** frac
            marginal = surv * (1 - sf)
            surv *= sf
            lgd = lgd0 + sum(lgm["macro_sensitivity"].get(v, 0.0) * devs[y][v] for v in devs[y])
            lgd = min(max(lgd, lgm["floor"]), lgm["cap"])
            mid = y + frac / 2
            ead = ead0 * (max(0.0, min(1.0, 1 - mid / h)) if p["amortization"] == "AMORTIZING" else 1.0)
            df = (1 + loan["interest_rate"]) ** -mid if discount else 1.0
            c = marginal * lgd * ead * df
            ecl_life += c
            if y == 0:
                ecl_12 = c
        lifetime = framework == "CECL" or stage >= 2
        total += w * (ecl_life if lifetime else ecl_12)
    return total


@pytest.mark.parametrize("framework", ["CECL", "IFRS9"])
def test_vectorised_engine_matches_scalar_reference(store, engine, framework):
    inp = store.engine_inputs(CURR)
    res = engine.run(inp, framework).loans.set_index("loan_id")
    loans = inp.loans.set_index("loan_id", drop=False)
    rng = np.random.default_rng(7)
    sample = list(rng.choice(loans.index, 150, replace=False))
    # always include every product and some defaulted loans
    sample += list(loans.groupby("product_type").head(2).index) + list(loans[loans.default_flag == 1].index[:5])
    for lid in sample:
        loan = loans.loc[lid].to_dict()
        ref = scalar_ecl(loan, inp.macro, inp.weights, framework, int(res.loc[lid, "ifrs9_stage"]))
        assert res.loc[lid, "ecl"] == pytest.approx(ref, rel=1e-9, abs=1e-6), lid
