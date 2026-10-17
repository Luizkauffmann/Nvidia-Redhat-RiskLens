"""Configuration loading. All tunables live in config/*.yaml."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

try:  # optional: pick up a local .env
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("RISKLENS_CONFIG_DIR", REPO_ROOT / "config"))


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load config/<name>.yaml (cached)."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def settings() -> dict:
    return load("settings")


def methodology() -> dict:
    return load("methodology")


def models() -> dict:
    return load("models")


def story() -> dict:
    return load("story")


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def data_dir() -> Path:
    return _resolve(os.environ.get("RISKLENS_DATA_DIR", settings()["data_dir"]))


def output_dir() -> Path:
    p = _resolve(os.environ.get("RISKLENS_OUTPUT_DIR", settings()["output_dir"]))
    p.mkdir(parents=True, exist_ok=True)
    return p


def reporting_dates() -> list[str]:
    return list(settings()["reporting_dates"])


def product_portfolio() -> dict[str, str]:
    return {p: v["portfolio"] for p, v in methodology()["products"].items()}
