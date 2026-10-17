"""Story variants: config overlay, story-aware task checks, and one variant built end to end."""
from __future__ import annotations

import copy

import pytest

from scripts.generate_demo_data import main as generate
from src.data.loader import read_ground_truth
from src.evaluation.gym_dataset import task_row
from src.evaluation.story_variants import design_checks
from src.evaluation.tasks import build_tasks
from src.utils import config


@pytest.fixture
def variant(monkeypatch):
    def use(name: str | None):
        if name:
            monkeypatch.setenv("RISKLENS_STORY_VARIANT", name)
        else:
            monkeypatch.delenv("RISKLENS_STORY_VARIANT", raising=False)
        config.reload()
    yield use
    monkeypatch.delenv("RISKLENS_STORY_VARIANT", raising=False)
    config.reload()


def test_variant_overlays_only_its_keys(variant):
    variant(None)
    base = config.demo_story()
    variant("credit_led")
    story = config.demo_story()
    assert story["cre_cohort"]["n_borrowers"] == 60 != base["cre_cohort"]["n_borrowers"]
    assert story["cre_cohort"]["property_types"] == base["cre_cohort"]["property_types"]   # merged, not replaced
    assert story["product_mix"] == base["product_mix"]
    assert story["design"]["primary_driver"] == "credit_deterioration"


def test_every_variant_states_a_design():
    for name in config.list_story_variants():
        assert config.load_yaml(f"story_variants/{name}").get("design", {}).get("primary_driver"), name


def _checks(tasks, task_id):
    return next(t for t in tasks if t.task_id == task_id).checks


def test_flat_macro_and_unchanged_weights_change_the_expected_answers(ground_truth):
    base = build_tasks(ground_truth, include_paraphrases=False)
    assert "names unemployment_rate" in [c["desc"] for c in _checks(base, "q11_macro_variable")]
    assert len(_checks(base, "q12_weights")) == 2

    gt = copy.deepcopy(ground_truth)
    gt["observed_cecl"]["macro_variables"] = {k: 0.0 for k in gt["observed_cecl"]["macro_variables"]}
    gt["design"]["scenario_weights_changed"] = False
    flat = build_tasks(gt, include_paraphrases=False)
    assert [c["desc"] for c in _checks(flat, "q11_macro_variable")] == ["says no macro variable moved the allowance"]
    assert [c["desc"] for c in _checks(flat, "q12_weights")] == ["says weights unchanged"]


def test_design_checks_flag_a_story_that_did_not_play_out(ground_truth):
    rows = design_checks(ground_truth)
    assert all(r["ok"] for r in rows)             # base story: macro / CRE / unemployment, increase
    gt = copy.deepcopy(ground_truth)
    gt["design"]["primary_driver"] = "portfolio_growth"
    assert not next(r for r in design_checks(gt) if r["claim"] == "primary_driver")["ok"]


def test_rows_are_tagged_with_their_variant(variant, ground_truth):
    task = build_tasks(ground_truth, include_paraphrases=False)[0]
    variant(None)
    assert task_row(task, tools=[], prompt="p")["id"] == task.task_id
    variant("growth_led")
    row = task_row(task, tools=[], prompt="p")
    assert row["id"] == f"growth_led__{task.task_id}"
    assert row["verifier_metadata"]["story_variant"] == "growth_led"


def test_credit_led_variant_plays_out_as_designed(variant, monkeypatch, tmp_path):
    monkeypatch.setenv("RISKLENS_OUTPUT_DIR", str(tmp_path / "outputs"))
    variant("credit_led")
    generate(tmp_path / "generated", tmp_path / "ground_truth")
    gt = read_ground_truth(tmp_path / "ground_truth")
    assert gt["story_variant"] == "credit_led"
    assert gt["design"]["scenario_weights_changed"] is False
    assert all(r["ok"] for r in design_checks(gt)), design_checks(gt)
    assert gt["observed_cecl"]["drivers"]["macroeconomic"] == pytest.approx(0.0, abs=1e-6)
