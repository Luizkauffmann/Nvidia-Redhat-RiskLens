import math

import numpy as np
import pandas as pd

from risklens import config
from risklens.cecl.engine import SCENARIOS, RunInputs, annual_deviation, base_lgd, base_logit, run_cecl
from tests.conftest import CUR, PREV


def test_rerun_matches_booked(store):
    for d in (PREV, CUR):
        assert abs(store.run(d).allowance - store.reported_allowance(d)) < 0.01
        assert np.allclose(store.run(d).ecl, store.loans(d)["ecl"].to_numpy())


def test_scalar_reference_one_loan(store):
    """Independent scalar re-implementation of the formula for a few loans."""
    inp = store.inputs(CUR)
    meth = config.methodology()
    sample = inp.loans[~inp.loans["default_flag"]].groupby("product_type").head(1)
    res = run_cecl(RunInputs(CUR, sample, inp.macro_dev, inp.weights, []))
    variables = meth["macro"]["variables"]
    for k, (_, ln) in enumerate(sample.reset_index(drop=True).iterrows()):
        prod = meth["products"][ln["product_type"]]
        life = prod.get("life_years") if prod["profile"] == "revolving" else ln["remaining_term_months"] / 12
        life = min(max(life, 1 / 12), meth["calculation"]["max_life_years"])
        n_years = math.ceil(life)
        dev = annual_deviation(inp.macro_dev, n_years)
        logit = base_logit(pd.DataFrame([ln]))[0]
        lgd0, lo, hi = base_lgd(pd.DataFrame([ln]))
        ccf = config.models()["ead_models"][ln["ead_model_id"]]["ccf"]
        ccf = ccf.get(ln["product_type"], 0) if isinstance(ccf, dict) else ccf
        ead0 = ln["current_balance"] + ccf * ln["undrawn_amount"]
        total = 0.0
        for si, s in enumerate(SCENARIOS):
            surv, ecl = 1.0, 0.0
            for y in range(n_years):
                frac = min(max(life - y, 0), 1)
                shift = sum(meth["macro"]["pd_beta"][ln["portfolio"]][v] * dev[si, vi, y] for vi, v in enumerate(variables))
                pd_y = 1 / (1 + math.exp(-(logit + shift)))
                lgd = lgd0[0]
                spec = meth["macro"]["lgd_collateral"].get(ln["portfolio"])
                if spec:
                    lgd = lgd + spec["kappa"] * -dev[si, variables.index(spec["variable"]), y]
                lgd = min(max(lgd, lo[0]), hi[0])
                t = y + 0.5 * frac
                shape = max(1 - t / life, 0) if prod["profile"] == "amortising" else 1.0
                ead = ead0 * shape * (1 - prod["prepay"]) ** y
                d = surv * (1 - (1 - pd_y) ** frac)
                ecl += d * lgd * ead
                surv -= d
            total += inp.weights[s] * ecl
        assert abs(total - res.ecl[k]) < 1e-6 * max(1, total), ln["product_type"]


def test_worse_macro_and_weights_raise_ecl(store):
    inp = store.inputs(PREV)
    base = run_cecl(inp).allowance
    dev = inp.macro_dev.copy()
    dev[:, 0, :] += 1.0   # +1pp unemployment
    assert run_cecl(RunInputs(PREV, inp.loans, dev, inp.weights, inp.overlays)).allowance > base
    sev = {"baseline": 0.0, "adverse": 0.0, "severe_adverse": 1.0}
    assert run_cecl(RunInputs(PREV, inp.loans, inp.macro_dev, sev, inp.overlays)).allowance > base


def test_defaulted_loans_full_loss(store):
    inp = store.inputs(CUR)
    res = run_cecl(inp)
    d = inp.loans["default_flag"].to_numpy()
    assert d.any()
    assert np.allclose(res.ecl[d], res.lgd[d] * res.ead[d])
    assert (res.pd_12m[d] == 1).all()


def test_overlays_added_once(store):
    res = store.run(CUR)
    assert abs(res.allowance - res.modeled_ecl - 1_900_000) < 1e-6
