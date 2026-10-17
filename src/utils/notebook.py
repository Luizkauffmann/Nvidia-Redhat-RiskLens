"""Small helpers shared by the notebooks (local Jupyter first, Colab compatible)."""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from src.utils import config

ROOT = Path(__file__).resolve().parents[2]


def M(x: float | None) -> str:
    """$ millions for display."""
    return "n/a" if x is None else f"${x / 1e6:,.2f}M"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines from the repo's .env into os.environ (existing variables win)."""
    path = Path(path) if path else ROOT / ".env"
    loaded: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v and k not in os.environ:
                os.environ[k] = v
                loaded[k] = "***" if "KEY" in k or "TOKEN" in k else v
    return loaded


def secret(name: str, default: str | None = None) -> str | None:
    """Environment variable, then .env, then Colab Secrets."""
    if os.environ.get(name):
        return os.environ[name]
    load_dotenv()
    if os.environ.get(name):
        return os.environ[name]
    if in_colab():
        from google.colab import userdata  # type: ignore
        try:
            return userdata.get(name)
        except Exception:
            return default
    return default


def ensure_demo_data(force: bool = False) -> dict:
    """Generate the seeded July/August data and book the official runs if they are missing."""
    data = config.path("generated")
    gt = config.path("ground_truth") / "ground_truth.json"
    if force or not (data / "allowance_summary.csv").exists() or not gt.exists():
        from scripts.generate_demo_data import main as generate
        generate()
        return {"status": "generated", "data_dir": str(data)}
    return {"status": "present", "data_dir": str(data)}


def endpoint_up(base_url: str, timeout: float = 3.0) -> bool:
    """True if an OpenAI-compatible server answers GET {base_url}/models."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False
