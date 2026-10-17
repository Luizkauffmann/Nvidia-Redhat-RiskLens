"""Configuration loading. All tunable settings live in config/*.yaml.

Locations can be redirected with environment variables, so the same code runs
from the repo (default), on Brev, or in Colab against a mounted folder:

    RISKLENS_CONFIG_DIR        config/*.yaml (methodology, models, settings, ...)
    RISKLENS_DATA_DIR          source data + official runs   (settings.paths.generated)
    RISKLENS_OUTPUT_DIR        caches, run logs, reports      (settings.paths.outputs)
    RISKLENS_GROUND_TRUTH_DIR  evaluation labels              (settings.paths.ground_truth)

Set them before the first config read, or call ``reload()`` afterwards.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"            # repo default; use config_dir() for the active one

ENV_PATHS = {"generated": "RISKLENS_DATA_DIR", "outputs": "RISKLENS_OUTPUT_DIR",
             "ground_truth": "RISKLENS_GROUND_TRUTH_DIR"}


def config_dir() -> Path:
    env = os.environ.get("RISKLENS_CONFIG_DIR")
    return Path(env).expanduser() if env else CONFIG_DIR


@lru_cache(maxsize=None)
def _load(directory: str, name: str) -> dict[str, Any]:
    path = Path(directory) / (name if name.endswith(".yaml") else f"{name}.yaml")
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_yaml(name: str) -> dict[str, Any]:
    return _load(str(config_dir()), name)


def reload() -> None:
    """Forget cached YAML (after editing config files or changing env vars)."""
    _load.cache_clear()


def settings() -> dict[str, Any]:
    return load_yaml("settings")


def methodology() -> dict[str, Any]:
    return load_yaml("methodology")


def models() -> dict[str, Any]:
    return load_yaml("models")


def demo_story() -> dict[str, Any]:
    return load_yaml("demo_story")


def reconciliation() -> dict[str, Any]:
    return load_yaml("reconciliation")


def path(key: str) -> Path:
    """Resolve a data/output path: env override first, else settings.paths relative to the repo root."""
    env = os.environ.get(ENV_PATHS.get(key, ""), "")
    return Path(env).expanduser() if env else ROOT / settings()["paths"][key]
