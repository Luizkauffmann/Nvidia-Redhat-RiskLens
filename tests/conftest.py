import json

import pytest

from risklens import config
from risklens.data.store import get_store

PREV, CUR = "2026-07-31", "2026-08-31"


@pytest.fixture(scope="session", autouse=True)
def data_ready():
    if not (config.data_dir() / "macro_scenarios.csv").exists():
        from risklens.data.generator import generate
        generate()
    return True


@pytest.fixture(scope="session")
def store(data_ready):
    return get_store()


@pytest.fixture(scope="session")
def gt(data_ready):
    # Tests may read ground truth; tools and agents may not.
    with open(config.data_dir() / config.settings()["ground_truth_file"]) as f:
        return json.load(f)
