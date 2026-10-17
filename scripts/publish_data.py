"""Publish the seeded synthetic dataset to data/published/ (committed to the repo).

The hackathon rules require synthetic data to be published with its
methodology. The working copy in data/generated/ stays git-ignored and is
regenerated on every machine; this script snapshots it for the record:

    python -m scripts.publish_data          # regenerate if needed, copy, write samples + checksums

Contents: every table in data/generated/, the ground truth (evaluation only),
small CSV samples for browsing on GitHub, and SHA256SUMS.json.
run_log.csv is excluded because it stores the wall-clock execution time.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generated"
GT = ROOT / "data" / "ground_truth" / "ground_truth.json"
OUT = ROOT / "data" / "published"
SKIP = {"run_log.csv", ".gitkeep"}


def main() -> None:
    if not (GEN / "loan_snapshots.parquet").exists() or not GT.exists():
        from scripts.generate_demo_data import main as generate
        generate()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(GEN.iterdir()):
        if f.is_file() and f.name not in SKIP:
            shutil.copy2(f, OUT / f.name)
    shutil.copy2(GT, OUT / "ground_truth.json")

    samples = OUT / "samples"
    samples.mkdir(exist_ok=True)
    loans = pd.read_parquet(GEN / "loan_snapshots.parquet")
    for d, g in loans.groupby("as_of_date"):
        g.groupby("product_type", group_keys=False).head(60).to_csv(samples / f"loans_{str(d)[:10]}_sample.csv", index=False)
    ecl = pd.read_parquet(GEN / "ecl_results.parquet")
    ecl[ecl["loan_id"].isin(loans["loan_id"].head(200))].to_csv(samples / "ecl_results_sample.csv", index=False)
    pd.read_parquet(GEN / "counterparties.parquet").head(300).to_csv(samples / "counterparties_sample.csv", index=False)

    sums = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
    (OUT / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Published {len(sums)} files ({size:.1f} MB) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
