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
            return userdata.get(name) or default
        except Exception:            # secret missing or notebook access not granted
            return default
    return default


SECRET_NAMES = ("NVIDIA_API_KEY", "RISKLENS_VLLM_URL", "RISKLENS_VLLM_MODEL", "VLLM_API_KEY",
                "RISKLENS_AGENT_BACKEND", "RISKLENS_GYM_POLICY")


def load_secrets() -> list[str]:
    """Copy known settings from .env (local) or Colab Secrets into os.environ; return the names found."""
    load_dotenv()
    found = []
    for name in SECRET_NAMES:
        value = secret(name)
        if value:
            os.environ[name] = value
            found.append(name)
    return found


def ensure_demo_data(force: bool = False) -> dict:
    """Generate the seeded July/August data and book the official runs if they are missing."""
    data = config.path("generated")
    gt = config.path("ground_truth") / "ground_truth.json"
    if force or not (data / "allowance_summary.csv").exists() or not gt.exists():
        from scripts.generate_demo_data import main as generate
        generate()
        return {"status": "generated", "data_dir": str(data)}
    return {"status": "present", "data_dir": str(data)}


GYM_MIN_PYTHON = (3, 13, 14)


def ensure_gym_cli(verbose: bool = True) -> str | None:
    """Make the NeMo Gym ``gym`` CLI available; return its path (None if it could not be installed).

    * already on PATH            -> use it
    * this Python >= 3.13.14     -> ``pip install nemo-gym`` into this environment
    * older Python (e.g. Colab)  -> ``uv`` builds a Python 3.13 venv at ``.venv-gym/`` just for the CLI and
                                    puts it on PATH; the notebook kernel keeps its own Python. The Gym
                                    servers run in their own venvs either way.
    """
    import shutil
    import subprocess

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    found = shutil.which("gym")
    if found:
        return found
    venv_bin = ROOT / ".venv-gym" / "bin"
    if (venv_bin / "gym").exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        return str(venv_bin / "gym")
    try:
        if sys.version_info[:3] >= GYM_MIN_PYTHON:
            log("Installing nemo-gym into this environment ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo-gym>=0.7.0", "uv"], check=True)
        else:
            log(f"Python {sys.version.split()[0]} is older than NeMo Gym needs; building a Python 3.13 venv with uv "
                "(about a minute) ...")
            if not shutil.which("uv"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
            uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
            venv = ROOT / ".venv-gym"
            subprocess.run([uv, "venv", "-q", "--seed", "--python", "3.13", str(venv)], check=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", str(venv / "bin" / "python"), "nemo-gym>=0.7.0", "uv"],
                           check=True)
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    except subprocess.CalledProcessError as exc:
        log(f"Could not install NeMo Gym ({exc}). The in-process evaluator still works.")
        return None
    return shutil.which("gym")


def endpoint_up(base_url: str, timeout: float = 3.0) -> bool:
    """True if an OpenAI-compatible server answers GET {base_url}/models."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False
