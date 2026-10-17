"""Regenerate the published synthetic data (seeded, reproducible) and warm the attribution cache.

    python -m scripts.generate_data            # ~15 s data + ~60 s attribution cache
    python -m scripts.generate_data --no-cache # data only
"""
import argparse
import time

from risklens.data.generator import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="skip warming the attribution cache")
    args = ap.parse_args()
    t0 = time.time()
    out = generate()
    for d, s in out["summary"].items():
        print(f"{d}: {s['loans']:>6,} loans  balance ${s['balance'] / 1e9:,.2f}B  allowance ${s['allowance'] / 1e6:,.2f}M")
    print(f"data written in {time.time() - t0:.0f}s")
    if not args.no_cache:
        from scripts.precompute import main as warm
        warm()


if __name__ == "__main__":
    main()
