from risklens.cecl import attribution as A
from tests.conftest import CUR, PREV


def test_sequential_telescopes_exactly(store):
    w = A.waterfall(PREV, CUR)
    assert abs(w["start_allowance"] - store.run(PREV).allowance) < 1e-6
    assert abs(w["end_allowance"] - store.run(CUR).allowance) < 1e-6
    assert abs(sum(r["impact"] for r in w["steps"]) - (w["end_allowance"] - w["start_allowance"])) < 1e-6


def test_reverse_and_shapley_sum_to_change(store):
    change = store.run(CUR).allowance - store.run(PREV).allowance
    rev = A.waterfall(PREV, CUR, order=("methodology", "macro", "credit", "growth", "runoff"))
    assert abs(sum(r["impact"] for r in rev["steps"]) - change) < 1e-6
    assert abs(sum(A.shapley_groups(PREV, CUR).values()) - change) < 1e-6


def test_macro_variables_sum_to_macro_step():
    w = A.waterfall(PREV, CUR)
    macro = next(r["impact"] for r in w["steps"] if r["step"] == "macro")
    assert abs(A.macro_variable_shapley(PREV, CUR)["macro_component"] - macro) < 1e-6


def test_ground_truth_recovered(gt):
    seq = A.driver_totals(A.waterfall(PREV, CUR)["steps"])
    assert max(seq, key=seq.get) == gt["primary_driver"]
    shap = A.shapley_groups(PREV, CUR)
    assert max(shap, key=shap.get) == "macro"
    assert seq["paydowns_runoff"] < 0 < seq["portfolio_growth"]
    mv = A.macro_variable_shapley(PREV, CUR)
    assert max(mv["by_variable"], key=mv["by_variable"].get) == gt["primary_macro_variable"]
    cre = {v: d["CRE"] for v, d in mv["by_variable_and_portfolio"].items()}
    assert max(cre, key=cre.get) == gt["primary_macro_variable_in_cre"]
