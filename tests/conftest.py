"""Shared fixtures: generate the demo dataset once per test session into a
temporary directory so tests never depend on (or overwrite) local data."""
from __future__ import annotations

import pytest

from scripts.generate_demo_data import main as generate
from src.cecl.attribution import AllowanceAttributor
from src.cecl.engine import ECLEngine
from src.data.loader import DataStore, read_ground_truth

PREV, CURR = "2026-07-31", "2026-08-31"


@pytest.fixture(scope="session")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("risklens")
    data_dir, gt_dir = root / "generated", root / "ground_truth"
    generate(data_dir, gt_dir)
    return data_dir, gt_dir


@pytest.fixture(scope="session")
def store(generated) -> DataStore:
    return DataStore(generated[0])


@pytest.fixture(scope="session")
def ground_truth(generated) -> dict:
    return read_ground_truth(generated[1])


@pytest.fixture(scope="session")
def engine() -> ECLEngine:
    return ECLEngine()


@pytest.fixture(scope="session")
def attribution(store, engine):
    return AllowanceAttributor(store, engine).run(PREV, CURR, "CECL")
