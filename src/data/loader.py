"""File-backed data store. Everything is plain Parquet/CSV — no platform lock-in.

Swapping pandas for cuDF later only requires changing the read functions here.
The ground-truth folder is deliberately NOT accessible through this class.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd

from src.cecl.engine import EngineInputs
from src.utils import config

FILES = {
    "loans": "loan_snapshots.parquet",
    "counterparties": "counterparties.parquet",
    "macro": "macro_scenarios.csv",
    "weights": "scenario_weights.csv",
    "overlays": "management_overlays.csv",
    "model_metadata": "model_metadata.csv",
    "model_assignments": "model_assignments.csv",
    "model_parameters": "model_parameters.csv",
    "results": "ecl_results.parquet",
    "summary": "allowance_summary.csv",
    "run_log": "run_log.csv",
}
DATE_COLS = {"as_of_date", "origination_date", "maturity_date", "approval_date", "model_run_date"}


def _as_date(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


class DataStore:
    def __init__(self, data_dir: Path | str | None = None):
        self.dir = Path(data_dir) if data_dir else config.path("generated")

    # ------------------------------------------------------------ writing
    def write(self, name: str, frame: pd.DataFrame) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / FILES[name]
        if p.suffix == ".parquet":
            frame.to_parquet(p, index=False)
        else:
            frame.to_csv(p, index=False)
        self.__dict__.pop(name, None)
        return p

    # ------------------------------------------------------------ reading
    def _read(self, name: str) -> pd.DataFrame:
        p = self.dir / FILES[name]
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run scripts/generate_demo_data.py first")
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        for c in DATE_COLS & set(df.columns):
            df[c] = pd.to_datetime(df[c])
        return df

    @cached_property
    def loans(self) -> pd.DataFrame: return self._read("loans")
    @cached_property
    def counterparties(self) -> pd.DataFrame: return self._read("counterparties")
    @cached_property
    def macro(self) -> pd.DataFrame: return self._read("macro")
    @cached_property
    def weights(self) -> pd.DataFrame: return self._read("weights")
    @cached_property
    def overlays(self) -> pd.DataFrame: return self._read("overlays")
    @cached_property
    def model_metadata(self) -> pd.DataFrame: return self._read("model_metadata")
    @cached_property
    def model_assignments(self) -> pd.DataFrame: return self._read("model_assignments")
    @cached_property
    def model_parameters(self) -> pd.DataFrame: return self._read("model_parameters")
    @cached_property
    def results(self) -> pd.DataFrame: return self._read("results")
    @cached_property
    def summary(self) -> pd.DataFrame: return self._read("summary")
    @cached_property
    def run_log(self) -> pd.DataFrame: return self._read("run_log")

    # ---------------------------------------------------------- by date
    def available_dates(self) -> list[str]:
        return sorted(f"{d:%Y-%m-%d}" for d in self.loans["as_of_date"].unique())

    def _at(self, frame: pd.DataFrame, as_of) -> pd.DataFrame:
        out = frame[frame["as_of_date"] == _as_date(as_of)]
        if out.empty:
            raise KeyError(f"No data for as_of_date={as_of}. Available: {self.available_dates()}")
        return out.reset_index(drop=True)

    def loans_at(self, as_of) -> pd.DataFrame: return self._at(self.loans, as_of)
    def counterparties_at(self, as_of) -> pd.DataFrame: return self._at(self.counterparties, as_of)
    def macro_at(self, as_of) -> pd.DataFrame: return self._at(self.macro, as_of)
    def overlays_at(self, as_of) -> pd.DataFrame: return self._at(self.overlays, as_of)
    def model_assignments_at(self, as_of) -> pd.DataFrame: return self._at(self.model_assignments, as_of)

    def weights_at(self, as_of) -> dict[str, float]:
        w = self._at(self.weights, as_of)
        return dict(zip(w["scenario"], w["weight"].astype(float)))

    def scenario_set_id(self, as_of) -> str:
        return str(self._at(self.weights, as_of)["scenario_set_id"].iloc[0])

    def results_at(self, as_of, framework: str = "CECL") -> pd.DataFrame:
        r = self._at(self.results, as_of)
        return r[r["framework"] == framework].reset_index(drop=True)

    def engine_inputs(self, as_of) -> EngineInputs:
        return EngineInputs(
            as_of_date=f"{_as_date(as_of):%Y-%m-%d}",
            loans=self.loans_at(as_of),
            macro=self.macro_at(as_of),
            weights=self.weights_at(as_of),
            overlays=self.overlays_at(as_of),
            scenario_set_id=self.scenario_set_id(as_of),
        )


def write_ground_truth(gt: dict, directory: Path | None = None) -> Path:
    d = directory or config.path("ground_truth")
    d.mkdir(parents=True, exist_ok=True)
    p = d / "ground_truth.json"
    p.write_text(json.dumps(gt, indent=2, default=str))
    return p


def read_ground_truth(directory: Path | None = None) -> dict:
    """Evaluation-only access. Tools and agents must never call this."""
    d = directory or config.path("ground_truth")
    return json.loads((d / "ground_truth.json").read_text())
