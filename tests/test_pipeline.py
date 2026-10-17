"""The traditional CECL workflow (src/pipeline) used by the Colab notebook."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src.data.loader import DataStore
from src.pipeline.cecl_run import export_model_parameters, inventory, run_official, seed_source_data, validate_inputs
from src.utils import config
from tests.conftest import CURR, PREV


@pytest.fixture()
def workdir(generated, tmp_path):
    """Private copy of the generated data (runs write into it)."""
    d = tmp_path / "data"
    shutil.copytree(generated[0], d)
    return d


def test_inputs_pass_validation(store):
    for d in (PREV, CURR):
        v = validate_inputs(d, store)
        assert v["status"] == "PASS", [c for c in v["checks"] if c["status"] != "PASS"]


def test_validation_blocks_unapproved_model(workdir):
    loans = pd.read_parquet(workdir / "loan_snapshots.parquet")
    m = (pd.to_datetime(loans["as_of_date"]) == pd.Timestamp(CURR)) & loans["pd_model_id"].eq("CRE_PD_v4")
    loans.loc[m, "pd_model_id"] = "CRE_PD_v5"                         # candidate model, still IN_VALIDATION
    loans.to_parquet(workdir / "loan_snapshots.parquet", index=False)
    v = validate_inputs(CURR, DataStore(workdir))
    bad = {c["check"] for c in v["checks"] if c["status"] == "FAIL"}
    assert {"models_approved_for_production", "loan_models_match_assignments"} <= bad
    with pytest.raises(RuntimeError, match="Input validation FAIL"):
        run_official(CURR, data_dir=workdir)


def test_official_run_is_idempotent_and_logged(workdir, store):
    before = store.summary.copy()
    run_official(CURR, ["CECL"], data_dir=workdir)
    run_official(CURR, ["CECL"], data_dir=workdir)                     # re-run replaces, never duplicates
    st = DataStore(workdir)
    assert len(st.summary) == len(before)
    assert len(st.results) == len(store.results)
    row = st.summary[(st.summary.framework == "CECL") & (pd.to_datetime(st.summary.as_of_date) == pd.Timestamp(CURR))]
    ref = before[(before.framework == "CECL") & (pd.to_datetime(before.as_of_date) == pd.Timestamp(CURR))]
    assert row["allowance"].iloc[0] == pytest.approx(ref["allowance"].iloc[0], abs=1e-6)
    log = st.run_log
    assert {"run_id", "input_hash_loans", "model_versions", "validation_status"} <= set(log.columns)
    cur = log[log["run_id"] == f"CECL_{CURR.replace('-', '')}_OFFICIAL"]
    assert cur["input_hash_loans"].nunique() == 1                       # same inputs, same hash


def test_seed_skips_existing_data(workdir, tmp_path):
    r = seed_source_data(workdir, tmp_path / "gt")
    assert r["status"] == "skipped"
    tables = {t["table"]: t for t in inventory(DataStore(workdir))}
    assert tables["loans"]["present"] and tables["loans"]["as_of_dates"] == [PREV, CURR]


def test_model_parameter_export_matches_registry(store):
    a = export_model_parameters().sort_values(["model_id", "parameter"]).reset_index(drop=True)
    b = store.model_parameters.sort_values(["model_id", "parameter"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_paths_follow_environment(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg"
    shutil.copytree(config.CONFIG_DIR, cfg)
    monkeypatch.setenv("RISKLENS_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("RISKLENS_DATA_DIR", str(tmp_path / "drive_data"))
    config.reload()
    try:
        assert config.config_dir() == cfg
        assert config.path("generated") == tmp_path / "drive_data"
        assert DataStore().dir == tmp_path / "drive_data"
    finally:
        monkeypatch.undo()
        config.reload()
    assert config.config_dir() == config.CONFIG_DIR


def test_databank_export_is_complete_and_documented(store, tmp_path):
    from src.pipeline.cecl_run import DICTIONARY, export_databank
    written = {w["file"]: w for w in export_databank(tmp_path, store, frameworks=["CECL"])}
    for d in (PREV, CURR):
        assert written[f"{d}/loan_portfolio_{d}.csv"]["rows"] == len(store.loans_at(d))
        res = pd.read_csv(tmp_path / f"{d}/ecl_results_CECL_{d}.csv")
        row = store.summary[(store.summary.framework == "CECL") & (pd.to_datetime(store.summary.as_of_date) == pd.Timestamp(d))]
        assert res["ecl"].sum() + store.overlays_at(d)["amount"].sum() == pytest.approx(row["allowance"].iloc[0], abs=0.01)
    assert {"reference/model_registry.csv", "data_dictionary.csv", "manifest.csv"} <= set(written)
    for table, frame in (("loan_portfolio", store.loans), ("borrowers", store.counterparties), ("ecl_results", store.results)):
        assert set(frame.columns) <= set(DICTIONARY[table]), table
