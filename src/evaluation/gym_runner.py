"""Drive NeMo Gym from Python (notebooks, scripts): pick a policy, collect rollouts, read rewards.

    from src.evaluation.gym_runner import POLICIES, run_gym_eval, load_rollouts
    out = run_gym_eval(POLICIES["mock"], split="example")   # starts all Gym servers, runs, stops them
    df = load_rollouts(out)

Policies (``POLICIES``):

* ``mock``    deterministic RiskLens planner behind an OpenAI-compatible endpoint (no GPU, no key)
* ``vllm``    Nemotron 3.5 Lightning on local vLLM / Red Hat AI (``deploy/brev_serve_nemotron.sh``)
* ``nvidia``  Nemotron 3.5 Lightning on the NVIDIA API catalog (``NVIDIA_API_KEY``)

The ``gym`` CLI must be on PATH (``pip install nemo-gym``, Python >= 3.13.14). Run from the repo root
so Gym finds ``resources_servers/risklens_cecl``.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "resources_servers" / "risklens_cecl" / "data"
AGENT = "risklens_cecl_simple_agent"


@dataclass(frozen=True)
class Policy:
    name: str
    model: str
    url: str
    api_key_env: str | None = None
    serve_mock: bool = False

    def api_key(self) -> str:
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise RuntimeError(f"Set {self.api_key_env} for the {self.name} policy")
            return key
        return "none"


POLICIES: dict[str, Policy] = {
    "mock": Policy("mock", "mock-planner", "http://127.0.0.1:8011/v1", serve_mock=True),
    "vllm": Policy("vllm", os.environ.get("RISKLENS_VLLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"),
                   os.environ.get("RISKLENS_VLLM_URL", "http://localhost:8000/v1"), "VLLM_API_KEY"),
    "nvidia": Policy("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", "https://integrate.api.nvidia.com/v1",
                     "NVIDIA_API_KEY"),
}
# vLLM started without --api-key accepts any key
if not os.environ.get("VLLM_API_KEY"):
    POLICIES["vllm"] = replace(POLICIES["vllm"], api_key_env=None)


def _clean_env() -> dict[str, str]:
    """Environment for gym subprocesses: drop the kernel's PYTHONPATH/PYTHONHOME (Colab sets them),
    which would leak packages into Gym's own Python 3.13 venvs."""
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}


def gym_available() -> bool:
    return shutil.which("gym") is not None


def ensure_dataset(agent: str | None = None, force: bool = False) -> dict[str, Path]:
    """Build the Gym JSONL files if missing (or ``force``)."""
    paths = {s: DATA_DIR / f"{s}.jsonl" for s in ("example", "train", "validation")}
    if force or not all(p.exists() for p in paths.values()):
        from src.evaluation.gym_dataset import build_rows, write_jsonl
        for split, rows in build_rows(agent=agent).items():
            write_jsonl(rows, paths[split])
    return paths


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@contextlib.contextmanager
def mock_policy_server(port: int = 8011) -> Iterator[subprocess.Popen | None]:
    """Run ``scripts/serve_policy.py --provider mock`` for the duration of the block."""
    if _port_open("127.0.0.1", port):
        yield None                      # already running (e.g. started by hand)
        return
    proc = subprocess.Popen([sys.executable, "-m", "scripts.serve_policy", "--provider", "mock", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        yield proc
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


def _model_args(policy: Policy) -> list[str]:
    return ["--resources-server", "risklens_cecl", "--model-type", "vllm_model",
            "--model", policy.model, "--model-url", policy.url, "--model-api-key", policy.api_key()]


def _redact(cmd: list[str]) -> str:
    out = list(cmd)
    if "--model-api-key" in out:
        i = out.index("--model-api-key") + 1
        out[i] = "none" if out[i] == "none" else "***"
    return " ".join(out)


@contextlib.contextmanager
def gym_servers(policy: Policy, log_path: Path | str | None = None, timeout_s: float = 1200) -> Iterator[Path]:
    """``gym env start`` for the RiskLens environment and the given policy; stopped when the block exits.

    The first start creates a venv per server, which can take a few minutes; later starts reuse them.
    Yields the server log path.
    """
    if not gym_available():
        raise RuntimeError("NeMo Gym CLI not found. Install it with `pip install nemo-gym` (Python >= 3.13.14).")
    log_path = Path(log_path) if log_path else ROOT / "data" / "outputs" / "gym" / "servers.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gym", "env", "start", *_model_args(policy)]
    print("$", _redact(cmd))
    mock = mock_policy_server(int(policy.url.rsplit(":", 1)[1].split("/")[0])) if policy.serve_mock else contextlib.nullcontext()
    with mock, log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env=_clean_env())
        try:
            t0 = time.time()
            while True:
                text = log_path.read_text(errors="replace")
                if "servers ready" in text:
                    break
                if proc.poll() is not None:
                    raise RuntimeError(f"gym env start exited ({proc.returncode}); see {log_path}:\n{text[-2000:]}")
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"Gym servers not ready after {timeout_s:.0f}s; see {log_path}")
                time.sleep(2)
            print(f"Gym servers ready in {time.time() - t0:.0f}s (log: {log_path.relative_to(ROOT)})")
            yield log_path
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)


def collect_rollouts(input_path: Path | str, output_path: Path | str, limit: int | None = None,
                     num_repeats: int = 1, concurrency: int | None = None, temperature: float | None = None,
                     stream: bool = True) -> Path:
    """``gym eval run --no-serve`` against running servers (see ``gym_servers``)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gym", "eval", "run", "--no-serve", "--agent", AGENT, "--input", str(input_path),
           "--output", str(output_path), "--num-repeats", str(num_repeats)]
    if limit:
        cmd += ["--limit", str(limit)]
    if concurrency:
        cmd += ["--concurrency", str(concurrency)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=not stream, env=_clean_env())
    if proc.returncode != 0:
        tail = "" if stream else f":\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        raise RuntimeError(f"gym eval run failed (exit {proc.returncode}){tail}")
    return output_path


def run_gym_eval(policy: Policy | str, split: str = "example", output: Path | str | None = None,
                 limit: int | None = None, num_repeats: int = 1, concurrency: int | None = None,
                 temperature: float | None = None, stream: bool = True, variant: str | None = None) -> Path:
    """Start the Gym servers, collect rollouts for one split (example | train | validation | hint), stop the servers.

    ``variant`` runs a story variant (``scripts/build_story_variants.py`` must have built it): the servers
    read that variant's data through env vars and the rows come from its dataset folder.
    """
    policy = POLICIES[policy] if isinstance(policy, str) else policy
    if variant:
        from src.evaluation.story_variants import dataset_dir, variant_env
        paths = {split: dataset_dir(variant) / f"{split}.jsonl"}
        env, tag = variant_env(variant), f"{variant}_"
    else:
        paths, env, tag = ensure_dataset(), {}, ""
        paths.setdefault(split, DATA_DIR / f"{split}.jsonl")      # extra splits, e.g. hint.jsonl
    if not paths[split].exists():
        raise FileNotFoundError(f"{paths[split]} missing (scripts.build_story_variants / scripts.build_hint_rows)")
    output = Path(output) if output else ROOT / "data" / "outputs" / "gym" / f"{policy.name}_{tag}{split}_rollouts.jsonl"
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)          # inherited by the gym subprocesses via _clean_env()
    try:
        with gym_servers(policy):
            return collect_rollouts(paths[split].relative_to(ROOT), output, limit, num_repeats, concurrency,
                                    temperature, stream)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_rollouts(path: Path | str) -> pd.DataFrame:
    """One row per rollout: task, reward, components, tool use, failed checks, answer."""
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            r = json.loads(line)
            meta = r.get("verifier_metadata") or {}
            calls = [o.get("name") for o in (r.get("response") or {}).get("output", []) if o.get("type") == "function_call"]
            rows.append({"task_id": meta.get("task_id"), "category": meta.get("category"),
                         "reward": r.get("reward"), **{k: r.get(k) for k in
                         ("correctness", "grounding", "tool_selection", "call_validity", "efficiency", "tool_calls", "tool_errors")},
                         "tools": calls, "failed_checks": [c["desc"] for c in r.get("checks", []) if not c.get("passed")],
                         "answer": r.get("final_answer", "")})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    cols = ["reward", "correctness", "grounding", "tool_selection", "call_validity", "efficiency", "tool_calls"]
    return {"rollouts": len(df), **{f"mean_{c}": round(float(df[c].mean()), 4) for c in cols if c in df},
            "pass_rate_reward_ge_0.8": round(float((df["reward"] >= 0.8).mean()), 4) if len(df) else None}
