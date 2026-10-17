"""Warm the analytics cache (attribution walks and Shapley checks).

The first call to the slow tools takes ~10-45 s; afterwards they are served
from data/outputs/cache in milliseconds. Run this after generating data and
before a live demo.

    python -m scripts.precompute_analytics            # CECL
    python -m scripts.precompute_analytics --ifrs9    # also IFRS 9
"""
from __future__ import annotations

import argparse
import time

from src.tools import get_context


def main(frameworks: list[str]) -> None:
    ctx = get_context()
    dates = ctx.dates
    for fw in frameworks:
        for prev, curr in zip(dates, dates[1:]):
            for label, fn in (("attribution", lambda: ctx.attribution(prev, curr, fw)),
                              ("attribution (reverse order)", lambda: ctx.attribution(prev, curr, fw, order="reverse")),
                              ("macro variable Shapley", lambda: ctx.macro_shapley(prev, curr, fw)),
                              ("driver Shapley", lambda: ctx.driver_shapley(prev, curr, fw))):
                t = time.perf_counter()
                fn()
                print(f"{fw} {prev} -> {curr}  {label:<28} {time.perf_counter() - t:6.1f}s")
    print(f"Cache: {ctx.cache_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifrs9", action="store_true", help="also precompute IFRS 9")
    args = ap.parse_args()
    main(["CECL", "IFRS9"] if args.ifrs9 else ["CECL"])
