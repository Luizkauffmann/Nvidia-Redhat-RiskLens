"""Story variants: the same bank and dates with a different July -> August story.

Each variant is ``config/story_variants/<name>.yaml`` merged over ``demo_story.yaml``
(``RISKLENS_STORY_VARIANT``). Its data, caches and ground truth live in their own folders, so
the base story is untouched and tools / Gym servers pick a variant purely through env vars:

    data/variants/<name>/{generated,outputs,ground_truth}
    resources_servers/risklens_cecl/data/variants/<name>/{example,train,validation}.jsonl

A variant states its intended outcome under ``design:``; ``design_checks`` compares that with
the observed attribution, so a story that does not play out as written is caught at build time.
Evaluation-only, like the ground truth it reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import config

ROOT = config.ROOT
GYM_DATA = ROOT / "resources_servers" / "risklens_cecl" / "data" / "variants"
HELD_OUT = ("property_shock",)       # never used for training data; the before/after test story


def variant_root(name: str) -> Path:
    return ROOT / "data" / "variants" / name


def variant_env(name: str) -> dict[str, str]:
    """Environment that points config, tools and Gym servers at one variant."""
    root = variant_root(name)
    return {"RISKLENS_STORY_VARIANT": name,
            "RISKLENS_DATA_DIR": str(root / "generated"),
            "RISKLENS_OUTPUT_DIR": str(root / "outputs"),
            "RISKLENS_GROUND_TRUTH_DIR": str(root / "ground_truth")}


def dataset_dir(name: str) -> Path:
    return GYM_DATA / name


def _top(values: dict[str, float], signed: bool = True) -> str:
    return max(values, key=(lambda k: values[k]) if signed else (lambda k: abs(values[k])))


def design_checks(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Intended outcome (``design``) vs observed CECL attribution; one row per stated claim."""
    d, obs = gt["design"], gt["observed_cecl"]
    observed = {"primary_driver": _top(obs["drivers"], signed=False),
                "primary_portfolio": _top(obs["portfolio_change"]),
                "primary_macro_variable": _top(obs["macro_variables"])}
    rows = [{"claim": "allowance_increases", "expected": True, "observed": obs["change"] > 0}]
    rows += [{"claim": k, "expected": d[k], "observed": observed[k]} for k in observed if k in d]
    for r in rows:
        r["ok"] = r["expected"] == r["observed"]
    return rows
