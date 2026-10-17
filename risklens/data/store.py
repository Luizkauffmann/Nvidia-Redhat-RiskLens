"""Read access to the published datasets (the only data tools may use).

The store never opens the hidden evaluation labels; tests enforce that the
tools and agents layers do not either.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from risklens import config
from risklens.cecl.engine import SCENARIOS, RunInputs, RunResult, run_cecl


class DataStore:
    def __init__(self, data_dir: Path | None = None):
        self.dir = Path(data_dir or config.data_dir())
        if not (self.dir / "macro_scenarios.csv").exists():
            raise FileNotFoundError(f"No data in {self.dir}. Run: python -m scripts.generate_data")
        self._loans: dict[str, pd.DataFrame] = {}
        self._runs: dict[str, RunResult] = {}
        self.macro = pd.read_csv(self.dir / "macro_scenarios.csv")
        self.overlays = pd.read_csv(self.dir / "management_overlays.csv")
        self.model_metadata = pd.read_csv(self.dir / "model_metadata.csv")
        with open(self.dir / "run_manifest.json") as f:
            self.manifest = {m["as_of_date"]: m for m in json.load(f)}
        self.dates = sorted(self.manifest)

    # ---- raw data ----------------------------------------------------------
    def loans(self, as_of: str) -> pd.DataFrame:
        if as_of not in self._loans:
            path = self.dir / f"loans_{as_of}.parquet"
            if not path.exists():
                raise KeyError(as_of)
            self._loans[as_of] = pd.read_parquet(path)
        return self._loans[as_of]

    @property
    def counterparties(self) -> pd.DataFrame:
        if not hasattr(self, "_cps"):
            self._cps = pd.read_parquet(self.dir / "counterparties.parquet")
        return self._cps

    def weights(self, as_of: str) -> dict[str, float]:
        sub = self.macro[self.macro["as_of_date"] == as_of]
        return {s: float(sub.loc[sub["scenario"] == s, "scenario_weight"].iloc[0]) for s in SCENARIOS}

    def overlay_list(self, as_of: str) -> list[dict]:
        sub = self.overlays[self.overlays["as_of_date"] == as_of]
        return sub[["overlay_id", "portfolio", "amount", "reason"]].to_dict("records")

    def macro_dev(self, as_of: str) -> np.ndarray:
        m = config.methodology()["macro"]
        sub = self.macro[self.macro["as_of_date"] == as_of].sort_values(["scenario", "forecast_quarter"])
        out = np.zeros((len(SCENARIOS), len(m["variables"]), m["forecast_quarters"]))
        for si, s in enumerate(SCENARIOS):
            rows = sub[sub["scenario"] == s]
            for vi, v in enumerate(m["variables"]):
                out[si, vi, :] = rows[v].to_numpy() - m["long_run_anchor"][v]
        return out

    # ---- engine runs ---------------------------------------------------------
    def inputs(self, as_of: str) -> RunInputs:
        return RunInputs(as_of, self.loans(as_of), self.macro_dev(as_of), self.weights(as_of), self.overlay_list(as_of))

    def run(self, as_of: str) -> RunResult:
        """Re-run the engine for a reporting date (should equal the booked run)."""
        if as_of not in self._runs:
            self._runs[as_of] = run_cecl(self.inputs(as_of))
        return self._runs[as_of]

    def reported_allowance(self, as_of: str) -> float:
        """Allowance as booked in the run manifest (the 'reported' figure)."""
        return float(self.manifest[as_of]["allowance"])


@lru_cache(maxsize=4)
def get_store(data_dir: str | None = None) -> DataStore:
    return DataStore(Path(data_dir) if data_dir else None)
