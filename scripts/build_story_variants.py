"""Build the story variants: data, official runs, ground truth, warm caches and Gym datasets.

    python -m scripts.build_story_variants                    # every config/story_variants/*.yaml
    python -m scripts.build_story_variants credit_led growth_led
    python -m scripts.build_story_variants --no-precompute     # skip cache warm-up (~30 s per variant)

Each variant runs in its own subprocess with RISKLENS_* env vars (src/evaluation/story_variants.py),
so the base story in data/generated is untouched. Afterwards every variant's intended outcome
(`design:`) is checked against the observed attribution and the results are written to
evidence/story_variants/summary.{json,md}. Exits non-zero if a story did not play out as designed.

Then collect rollouts per variant:
    run_gym_eval("vllm", split="train", variant="credit_led", num_repeats=4)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from src.evaluation.story_variants import dataset_dir, design_checks, variant_env, variant_root
from src.utils import config

EVIDENCE = config.ROOT / "evidence" / "story_variants"


def build(name: str, precompute: bool) -> None:
    env = {**os.environ, **variant_env(name)}
    steps = [["scripts.generate_demo_data"], *([["scripts.precompute_analytics"]] if precompute else []),
             ["scripts.build_gym_dataset", "--out", str(dataset_dir(name))]]
    for step in steps:
        subprocess.run([sys.executable, "-m", *step], cwd=config.ROOT, env=env, check=True)


def summarize(name: str) -> dict:
    gt = json.loads((variant_root(name) / "ground_truth" / "ground_truth.json").read_text())
    obs = gt["observed_cecl"]
    story = config.load_yaml(f"story_variants/{name}")
    return {"variant": name, "description": story.get("description", ""),
            "previous_allowance": obs["previous_allowance"], "current_allowance": obs["current_allowance"],
            "change": obs["change"], "drivers": obs["drivers"], "portfolio_change": obs["portfolio_change"],
            "macro_variables": obs["macro_variables"], "largest_borrower": gt["design"]["largest_borrower"],
            "scenario_weights_changed": gt["design"]["scenario_weights_changed"],
            "design_checks": design_checks(gt), "gym_dataset": str(dataset_dir(name).relative_to(config.ROOT))}


def to_markdown(rows: list[dict]) -> str:
    m = 1e6
    lines = ["# Story variants", "",
             "Same bank, same dates (2026-07-31 → 2026-08-31), different story. Built by "
             "`python -m scripts.build_story_variants`; every figure comes from the CECL engine's official runs "
             "and sequential attribution. `design` = what the variant config says should happen; "
             "`observed` = what the attribution found.", "",
             "| Variant | July $M | August $M | Change $M | Primary driver (observed) | Top portfolio | Top macro variable | Design checks |",
             "|---|---:|---:|---:|---|---|---|---|"]
    for r in rows:
        drv = max(r["drivers"], key=lambda k: abs(r["drivers"][k]))
        port = max(r["portfolio_change"], key=r["portfolio_change"].get)
        var = max(r["macro_variables"], key=r["macro_variables"].get)
        ok = sum(c["ok"] for c in r["design_checks"])
        lines.append(f"| `{r['variant']}` | {r['previous_allowance'] / m:.2f} | {r['current_allowance'] / m:.2f} | "
                     f"{r['change'] / m:+.2f} | {drv} ({r['drivers'][drv] / m:+.2f}) | {port} | {var} | "
                     f"{ok}/{len(r['design_checks'])} {'PASS' if ok == len(r['design_checks']) else 'FAIL'} |")
    drivers = list(config.methodology()["attribution"]["drivers"])     # fixed column order
    lines += ["", "## Driver attribution ($M)", "",
              "| Variant | " + " | ".join(drivers) + " |", "|---|" + "---:|" * len(drivers)]
    for r in rows:
        lines.append(f"| `{r['variant']}` | " + " | ".join(f"{r['drivers'].get(d, 0.0) / m:+.2f}" for d in drivers) + " |")
    lines += ["", "## Design checks", ""]
    for r in rows:
        lines.append(f"- **{r['variant']}**: {r['description']}")
        for c in r["design_checks"]:
            lines.append(f"  - {c['claim']}: expected `{c['expected']}`, observed `{c['observed']}` "
                         f"{'✓' if c['ok'] else '✗'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variants", nargs="*", help="default: all in config/story_variants")
    ap.add_argument("--no-precompute", action="store_true")
    args = ap.parse_args()
    names = args.variants or config.list_story_variants()
    for name in names:
        t0 = time.time()
        print(f"=== {name}")
        build(name, precompute=not args.no_precompute)
        print(f"=== {name} built in {time.time() - t0:.0f}s")
    built = [n for n in config.list_story_variants() if (variant_root(n) / "ground_truth" / "ground_truth.json").exists()]
    rows = [summarize(n) for n in built]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    (EVIDENCE / "summary.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    failed = [r["variant"] for r in rows if not all(c["ok"] for c in r["design_checks"])]
    if failed:
        sys.exit(f"stories that did not play out as designed: {failed}")


if __name__ == "__main__":
    main()
