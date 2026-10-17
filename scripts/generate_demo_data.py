"""Generate the synthetic July/August 2026 dataset, book the official runs, and
write the ground truth. Uses the same pipeline steps as the Colab workflow.

    python -m scripts.generate_demo_data            (from the repo root)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cecl_run import (record_observed_ground_truth, run_official,  # noqa: E402
                                   seed_source_data)
from src.utils import config  # noqa: E402


def main(data_dir: Path | None = None, ground_truth_dir: Path | None = None) -> dict:
    t0 = time.time()
    seed = seed_source_data(data_dir, ground_truth_dir, overwrite=True)
    for p in ("results", "summary", "run_log"):         # fresh books for a fresh dataset
        f = Path(seed["data_dir"]) / {"results": "ecl_results.parquet", "summary": "allowance_summary.csv",
                                      "run_log": "run_log.csv"}[p]
        f.unlink(missing_ok=True)
    dates = config.settings()["reporting_dates"]
    for d in dates:
        run_official(d, data_dir=data_dir)
    gt = record_observed_ground_truth(dates[0], dates[1], data_dir, ground_truth_dir)
    print(f"Generated data in {seed['data_dir']} and ground truth in {seed['ground_truth']} ({time.time() - t0:.1f}s)")
    return gt


if __name__ == "__main__":
    main()
