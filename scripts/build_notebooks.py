"""Generate the RiskLens notebooks from this file (the notebooks are build outputs; edit here).

    python -m scripts.build_notebooks          # writes notebooks/0*.ipynb (outputs cleared)

Notebooks are local-first (Jupyter on your laptop or a Brev box, run from the repo checkout) and
still work in Google Colab: the bootstrap cell clones the repo when it isn't already there.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
REPO = "Luizkauffmann/Nvidia-Redhat-RiskLens"

BOOTSTRAP = f'''# RiskLens bootstrap: finds the repo checkout (local Jupyter) or clones it (Colab)
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/{REPO}.git"

def _find_repo():
    env = os.environ.get("RISKLENS_REPO")
    for p in ([Path(env)] if env else []) + [Path.cwd(), *Path.cwd().parents]:
        if (p / "config" / "gym.yaml").exists() and (p / "src" / "tools").exists():
            return p.resolve()
    return None

REPO_DIR = _find_repo()
if REPO_DIR is None:
    if not IN_COLAB:
        raise SystemExit("Start Jupyter from the Nvidia-Redhat-RiskLens checkout (or set RISKLENS_REPO).")
    REPO_DIR = Path("/content/Nvidia-Redhat-RiskLens")
    if (REPO_DIR / ".git").exists():                 # runtime restarted: refresh the existing clone
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "-q", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt"), "matplotlib"], check=True)

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.utils import config
config.reload()
from src.utils.notebook import M, ensure_demo_data, endpoint_up, load_secrets, secret
print("secrets:", load_secrets() or "none (.env locally, Colab Secrets in Colab)")
print("repo   :", REPO_DIR, "| python", sys.version.split()[0], "| colab" if IN_COLAB else "| local")
print("data   :", ensure_demo_data())
'''


def badge(name: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


def notebook(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    return nb


# --------------------------------------------------------------------------- #
# 01 · Traditional CECL workflow
# --------------------------------------------------------------------------- #
WORKFLOW = [
    md("""
# RiskLens 01 · The CECL month-end run

""" + badge("01_CECL_Workflow.ipynb") + """

This is the **traditional CECL run** a bank's allowance team does every month-end, before any AI is involved. RiskLens's agents never run these steps; they investigate the booked results through deterministic tools (step 5), so every number they quote traces back to this run.

| Step | What happens |
|---|---|
| 0 | Find the repo, load settings, make sure the seeded demo data exists |
| 1 | Source extracts: loans, borrowers, macro scenarios, scenario weights, overlays, model registry |
| 2 | Pre-run controls: data quality, model approval status, weights, macro coverage |
| 3 | Official run: PD / LGD / EAD by assigned model, lifetime ECL per scenario, probability weighting, overlays. Booked with a run log |
| 4 | Allowance movement analysis: waterfall, segments, macro variables, top borrowers, reconciliation |
| 5 | Hand-off: the tool schemas the agents get |

**Local or Colab.** Locally, start Jupyter from the repo root (`jupyter lab`). In Colab, the first cell clones the repo. Then run top to bottom. Data lives in `data/`, regenerated from a fixed seed, so it is identical on every machine. No GPU and no model are needed here.
"""),
    md("## 0 · Setup"),
    code(BOOTSTRAP),
    code('''
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from src.data.loader import DataStore
from src.pipeline import cecl_run

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.max_colwidth", 80)
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False})
DATES = config.settings()["reporting_dates"]
PREV, CURR = DATES[-2], DATES[-1]
store = DataStore()
print("config :", config.config_dir())
print("data   :", config.path("generated"))
print("periods:", DATES)
'''),
    md("""
## 1 · Source data
In a bank these extracts come from loan servicing, credit and finance systems. Here a seeded generator stands in for them (`python -m scripts.generate_demo_data` rebuilds them from scratch).
"""),
    code('''
display(pd.DataFrame(cecl_run.inventory(store)))
loans = store.loans.assign(date=store.loans.as_of_date.dt.strftime("%Y-%m-%d"))
book = loans.groupby(["portfolio", "date"]).agg(loans=("loan_id", "size"), balance=("current_balance", "sum")).unstack("date")
book.apply(lambda c: c.map(M) if c.name[0] == "balance" else c.map("{:,}".format))
'''),
    md("""
### Risk model registry
Each product is scored by a **PD**, an **LGD** and an **EAD** model. The registry holds versions and approval status; `config/models.yaml` holds the scorecard parameters the engine uses.
"""),
    code('''
display(store.model_metadata)
display(store.model_assignments_at(CURR).drop(columns=["as_of_date"]))
mp = store.model_parameters
mp[mp.model_id == "CRE_PD_v4"].set_index("parameter")[["value"]].T
'''),
    md("### Macroeconomic scenarios (reasonable & supportable forecast, then reversion)"),
    code('''
macro = store.macro.assign(vintage=store.macro.as_of_date.dt.strftime("%Y-%m-%d"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, var in zip(axes, ["unemployment_rate", "cre_price_growth"]):
    for (scen, vint), g in macro.groupby(["scenario", "vintage"]):
        ax.plot(g.forecast_quarter, g[var], marker="o", ls="-" if vint == CURR else "--", label=f"{scen} · {vint}")
    ax.set_title(f"{var}: July (dashed) vs August forecast")
    ax.tick_params(axis="x", rotation=45)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
display(store.weights.assign(as_of_date=store.weights.as_of_date.dt.strftime("%Y-%m-%d")))
display(store.overlays.assign(as_of_date=store.overlays.as_of_date.dt.strftime("%Y-%m-%d")))
'''),
    md("""
## 2 · Pre-run controls
A FAIL blocks the official run, e.g. a loan scored with `CRE_PD_v5`, which is still in validation.
"""),
    code('''
val = {d: cecl_run.validate_inputs(d, store) for d in DATES}
print({d: v["status"] for d, v in val.items()})
pd.DataFrame({d: {c["check"]: c["status"] for c in v["checks"]} for d, v in val.items()})
'''),
    md("""
## 3 · Official run
Per loan: lifetime ECL = Σ marginal PD × LGD × EAD over the remaining life, for each scenario path. Then probability-weight the scenarios and add the management overlays. CECL (lifetime) and IFRS 9 (12-month / lifetime by stage) are both booked.
"""),
    code('''
for d in DATES:
    t = time.time()
    out = cecl_run.run_official(d)
    for r in out["runs"]:
        print(f"{r['run_id']:26s} allowance {M(r['allowance']):>10}  ({r['loan_count']:,} loans, controls {r['validation_status']})  {time.time() - t:4.1f}s")
store = DataStore()            # re-read the booked results
store.run_log[["run_id", "executed_at_utc", "allowance", "scenario_set_id", "input_hash_loans", "validation_status"]].tail(4)
'''),
    code('''
r = store.results
r = r[r.framework == "CECL"].assign(date=lambda x: x.as_of_date.dt.strftime("%Y-%m-%d"))
by_port = r.groupby(["portfolio", "date"]).ecl.sum().unstack()
ax = (by_port[[PREV, CURR]] / 1e6).plot.bar(title="Modeled CECL ECL by portfolio ($M)", rot=0)
ax.set_ylabel("$M"); plt.show()
by_port["change"] = by_port[CURR] - by_port[PREV]
by_port.apply(lambda c: c.map(M))
'''),
    md("""
### Databank export
A readable CSV copy for people (`data/outputs/databank/`): loans, borrowers and official results per date, plus reference tables and a data dictionary. The engine keeps reading Parquet.
"""),
    code('''
written = cecl_run.export_databank(config.path("outputs") / "databank", store, frameworks=["CECL", "IFRS9"])
pd.DataFrame(written).head(12)
'''),
    md("""
## 4 · Allowance movement analysis
These cells use the **same deterministic tools the agents call** (`src/tools`), so the notebook, the agents and the NeMo Gym environment all see the same numbers. The first call computes the attribution walk and Shapley checks (1–2 minutes) and caches them in `data/outputs/cache`.
"""),
    code('''
from src.tools import ToolCallLog, call_tool, list_tools
LOG = ToolCallLog(config.path("outputs") / "notebook_tool_calls.jsonl")

def tool(name, **args):
    rec = call_tool(name, args, caller="workflow_notebook", log=LOG)
    if rec.status != "ok":
        raise RuntimeError(rec.error)
    return rec.result

cmp_ = tool("compare_allowance", previous_date=PREV, current_date=CURR)
print(f"Allowance {M(cmp_['previous_allowance'])} -> {M(cmp_['current_allowance'])}  change {M(cmp_['change'])} ({cmp_['change_pct']:.1f}%)")
'''),
    code('''
wf = tool("calculate_portfolio_attribution", previous_date=PREV, current_date=CURR)
order = list(dict.fromkeys(s["driver"] for s in wf["steps"]))
imp = {d["driver"]: d for d in wf["drivers"]}
labels = [PREV] + [imp[d]["label"] for d in order] + [CURR]
deltas = [imp[d]["impact"] for d in order]
heights = [wf["previous_allowance"]] + deltas + [wf["current_allowance"]]
bottoms = [0] + [wf["previous_allowance"] + sum(deltas[:i]) + min(deltas[i], 0) for i in range(len(deltas))] + [0]
colors = ["#7f8c8d"] + ["#c0392b" if v > 0 else "#27ae60" for v in deltas] + ["#2c3e50"]
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(range(len(labels)), [abs(h) / 1e6 for h in heights], bottom=[b / 1e6 for b in bottoms], color=colors)
for i, h in enumerate(heights):
    ax.text(i, (bottoms[i] + abs(h)) / 1e6 + 0.5, M(h), ha="center", fontsize=8)
ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
ax.set_ylim(min(wf["previous_allowance"], wf["current_allowance"]) / 1e6 * 0.85, max(np.cumsum([wf["previous_allowance"]] + deltas)) / 1e6 * 1.05)
ax.set_title(f"CECL allowance waterfall (residual {M(wf['residual'])})"); ax.set_ylabel("$M")
plt.tight_layout(); plt.show()
pd.DataFrame(wf["drivers"])[["label", "impact", "share_of_change_pct"]]
'''),
    code('''
seg = tool("calculate_segment_attribution", previous_date=PREV, current_date=CURR)
mat = pd.DataFrame([{"portfolio": s["segment"], **s["drivers"], "total": s["total_change"]} for s in seg["segments"]]).set_index("portfolio")
grid = mat.drop(columns="total") / 1e6
lim = float(np.abs(grid.values).max())
fig, ax = plt.subplots(figsize=(10, 3.8))
im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=20, ha="right"); ax.set_yticks(range(grid.shape[0]), grid.index)
for (i, j), v in np.ndenumerate(grid.values):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Allowance change by portfolio × driver ($M)"); fig.colorbar(im); plt.tight_layout(); plt.show()
mat.apply(lambda c: c.map(M))
'''),
    code('''
mv = tool("calculate_variable_attribution", previous_date=PREV, current_date=CURR)   # exact Shapley, 64 projections
v = pd.DataFrame(mv["variables"]).sort_values("impact")
ax = (v.set_index("variable")["impact"] / 1e6).plot.barh(title=f"Macro impact by variable (Shapley), total {M(mv['macro_total'])}")
ax.set_xlabel("$M"); plt.show()
v[["variable", "impact", "sequential_impact", "share_of_macro_pct"]]
'''),
    code('''
top = tool("identify_top_contributors", previous_date=PREV, current_date=CURR, n=10)
pd.DataFrame(top["contributors"])[["rank", "counterparty_id", "customer_name", "portfolio", "balance_current",
                                    "ecl_previous", "ecl_current", "ecl_change", "pd_12m_previous", "pd_12m_current",
                                    "main_driver", "reason_codes"]]
'''),
    code('''
mod = tool("compare_model_versions", previous_date=PREV, current_date=CURR)
w = tool("compare_scenario_weights", previous_date=PREV, current_date=CURR)
print("model versions changed:", mod["any_model_version_changed"], "| impact", M(mod["allowance_impact_of_model_change"]))
print("candidates not in production:", [m["model_id"] for m in mod["candidate_models_not_in_production"]])
print("scenario weights changed:", w["weights_changed"], "-> toward", w["weight_shifted_toward"], "| impact", M(w["allowance_impact_of_weight_change"]))
'''),
    code('''
rec = tool("reconcile_allowance_change", previous_date=PREV, current_date=CURR)
print(f"Reported {M(rec['reported_change'])} | explained {M(rec['explained_change'])} | residual {M(rec['residual'])} "
      f"| {rec['pct_explained']:.2f}% explained -> {rec['overall_status']}")
display(pd.DataFrame(rec["controls"])[["control", "status", "value"]])
pd.DataFrame(rec["order_robustness"]["drivers"])[["label", "sequential", "reverse_sequential", "shapley"]]
'''),
    md("""
### What-if: scenario weights
Weighted ECL is linear in the weights, so this is an exact re-weighting of the booked scenario results. A what-if, not a booked run.
"""),
    code('''
wi = tool("run_scenario_analysis", as_of_date=CURR, weights={"baseline": 0.50, "adverse": 0.30, "severe_adverse": 0.20})
print("official:", M(wi["official_allowance"]), "| 50/30/20:", M(wi["custom"]["allowance"]), f"({wi['custom']['difference_pct']:+.1f}%)")
{k: M(v) for k, v in wi["single_scenario_allowance"].items()}
'''),
    md("""
## 5 · Hand-off to the agents
Agents get **only these tool schemas**; each specialist sees a focused subset (`config/agents.yaml`). Every call is validated, repaired if needed (`"Aug 2026"` → `2026-08-31`) and logged with provenance. The NeMo Gym environment (notebook 03) exposes exactly these tools as HTTP endpoints.
"""),
    code('''
print({a: len(list_tools(a)) for a in ["orchestrator", "portfolio_agent", "credit_agent", "macro_agent", "model_risk_agent", "reconciliation_agent"]})
r = call_tool("calculate_pd_migration", {"previous_date": "July 2026", "current_date": "aug", "portfolio": "commercial real estate"}, caller="demo", log=LOG)
print("normalised parameters:", r.parameters)
print("provenance:", r.provenance())
bad = call_tool("calculate_pd_migration", {"previous_date": "2025-01-31", "current_date": CURR}, log=LOG)
print("structured error:", bad.error)
LOG.stats()
'''),
    md("""
**Next:** `02_Agent_Investigator.ipynb` asks Nemotron to explain this movement; `03_NeMo_Gym_Environment.ipynb` scores it.

**Try a model change:** edit `config/models.yaml` (for example point CRE at `CRE_PD_v5` parameters), re-run steps 3–4, and watch the Model Risk tools and the reconciliation pick it up. `git checkout config/models.yaml` restores the defaults.
"""),
]


# --------------------------------------------------------------------------- #
# 02 · Agent
# --------------------------------------------------------------------------- #
BACKEND_CELL = '''
# Which model answers. Local default: Nemotron 3.5 Lightning on vLLM (deploy/brev_serve_nemotron.sh).
#   vllm   -> RISKLENS_VLLM_URL (default http://localhost:8000/v1; a Brev box works too)
#   nvidia -> NVIDIA API catalog, needs NVIDIA_API_KEY in .env
#   mock   -> deterministic planner, no model (offline demo / CI)
#   auto   -> vllm if it answers, else nvidia if a key is set, else mock
BACKEND = os.environ.get("RISKLENS_AGENT_BACKEND", "auto")
VLLM_URL = os.environ.get("RISKLENS_VLLM_URL", "http://localhost:8000/v1")
ENABLE_THINKING = True                      # Nemotron reasoning mode: slower, plans better

if BACKEND == "auto":
    BACKEND = "vllm" if endpoint_up(VLLM_URL) else ("nvidia" if secret("NVIDIA_API_KEY") else "mock")
overrides = {"provider": BACKEND, "enable_thinking": ENABLE_THINKING}
if BACKEND == "vllm":
    overrides["base_url"] = VLLM_URL
    overrides["model"] = os.environ.get("RISKLENS_VLLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
elif BACKEND == "nvidia":
    os.environ["NVIDIA_API_KEY"] = secret("NVIDIA_API_KEY") or ""

from src.llm.provider import get_provider
provider = get_provider(overrides)
print("backend:", BACKEND, "|", provider.describe())
'''

AGENT = [
    md("""
# RiskLens 02 · Ask Nemotron why the allowance moved

""" + badge("02_Agent_Investigator.ipynb") + """

An **agent on top of the booked CECL runs**. Nemotron 3.5 Lightning (30B total / 3B active, within the 33B limit) plans the investigation and calls RiskLens's deterministic tools. A grounding check verifies that every number in its answer came from a tool result, and each answer is scored with the **same verifier the NeMo Gym environment uses**.

**Local or Colab.** Secrets come from `.env` locally or from Colab Secrets (key icon) in Colab: `NVIDIA_API_KEY`, `RISKLENS_VLLM_URL`. Colab's GPUs are too small for Nemotron, so in Colab use `nvidia` or point `vllm` at your Brev box. Pick the model in the backend cell:

| Backend | What you need |
|---|---|
| `vllm` (default) | Nemotron served by vLLM / Red Hat AI on this machine or a Brev box: `bash deploy/brev_serve_nemotron.sh`, then `RISKLENS_VLLM_URL` if it isn't `localhost:8000` |
| `nvidia` | `NVIDIA_API_KEY=nvapi-...` in `.env` (free key from build.nvidia.com) |
| `mock` | nothing: deterministic planner, for testing without a model |
"""),
    md("## 1 · Setup"),
    code(BOOTSTRAP),
    md("## 2 · Connect to the model"),
    code(BACKEND_CELL),
    code('''
# Smoke test: one plain turn + one tool call
from src.tools import tool_schemas
r = provider.chat([{"role": "user", "content": "Reply with exactly: RiskLens ready"}])
print(f"chat  : {(r.content or '').strip()[:60]!r}  ({r.latency_ms:.0f} ms)")
tools = [s for s in tool_schemas() if s["function"]["name"] == "compare_allowance"]
r = provider.chat([{"role": "user", "content": "How much did the CECL allowance change between 2026-07-31 and 2026-08-31?"}], tools=tools)
print("tools :", [(t.name, t.arguments) for t in r.tool_calls] or (r.content or "")[:200], "| format:", r.raw_tool_format)
'''),
    md("""
## 3 · Ask
`ask()` runs one investigation and shows the answer, a grounding badge, the **Gym reward** when the question is in the task bank, and the trace: every model turn and tool call with its parameters and `call_id`.
"""),
    code('''
import json
import pandas as pd
from IPython.display import Markdown, display
from src.agents.orchestrator import build_orchestrator
from src.evaluation import verifier
from src.evaluation.tasks import build_tasks
from src.tools import ToolCallLog

LOG = ToolCallLog(config.path("outputs") / "agent_tool_calls.jsonl")
agent = build_orchestrator(provider, log=LOG)
TASKS = {t.question: t.to_dict() for t in build_tasks()}
RUNS = []

def ask(question, show_trace=True):
    run = agent.run(question)
    RUNS.append(run)
    g = (run.grounding or {}).get("status", "n/a")
    badge = {"PASS": "✅ every number traced to a tool result", "NO_NUMBERS": "✅ no figures quoted",
             "FAIL": "⚠️ unsupported numbers: " + ", ".join(u["text"] for u in (run.grounding or {}).get("ungrounded", []))}.get(g, g)
    score = ""
    if question in TASKS:
        res = verifier.score(*verifier.from_agent_run(run), TASKS[question])
        run.events.append({"type": "gym_reward", **res.to_dict()})
        missed = [c["desc"] for c in res.checks if not c["passed"]]
        score = f" · Gym reward **{res.reward:.2f}**" + (f" (missed: {', '.join(missed)})" if missed else "")
    display(Markdown(f"**Q: {question}**\\n\\n{run.answer}\\n\\n---\\n*{run.provider.get('model')} · {run.status} · "
                     f"{run.duration_s}s · {len(run.tool_records)} tool calls · grounding: {badge}*{score}"))
    if show_trace:
        rows = []
        for e in run.events:
            if e["type"] == "llm":
                rows.append({"step": "model", "detail": ", ".join(e["tool_calls"]) or "final answer", "status": e["tool_format"], "ms": e["latency_ms"]})
            elif e["type"] == "tool":
                rows.append({"step": "tool", "detail": f"{e['name']}({json.dumps(e.get('parameters') or {})})",
                             "status": "duplicate" if e.get("duplicate") else e["status"], "ms": e.get("duration_ms")})
                if e.get("error"):
                    rows.append({"step": "  hint", "detail": f"{e['error']['message']} -> {e['error'].get('hint')}", "status": "", "ms": None})
            elif e["type"] == "grounding":
                rows.append({"step": "grounding", "detail": ", ".join(e["ungrounded"]) or "all numbers supported", "status": e["status"], "ms": None})
        with pd.option_context("display.max_colwidth", 140):
            display(pd.DataFrame(rows))
    return run
'''),
    md("### The demo questions"),
    code('_ = ask("Why did our allowance increase in August? What were the major drivers?")'),
    code('_ = ask("Which macroeconomic variable had the largest impact on the allowance?")'),
    code('_ = ask("What happened specifically in Commercial Real Estate?")'),
    code('_ = ask("Which borrowers contributed most to the increase? Who should I investigate first?")'),
    code('_ = ask("Could this increase just be because we changed models?")'),
    code('_ = ask("Does the attribution reconcile to the total allowance change?")'),
    md("### Your own question"),
    code('_ = ask("Explain borrower CP_040408: what changed and how much did it add to the allowance?")'),
    md("""
## 4 · Tool-calling scorecard
Reliability metrics for the **Tool-Calling** track: success rate, repaired arguments, recovered errors, tool calls written as text, grounding, and the Gym reward where the question is in the task bank.
"""),
    code('''
def gym_reward(run):
    e = next((e for e in run.events if e["type"] == "gym_reward"), None)
    return e["reward"] if e else None

summary = pd.DataFrame([{"question": r.question[:60], "gym_reward": gym_reward(r), **r.metrics()} for r in RUNS])
display(summary[["question", "gym_reward", "tool_calls", "tool_errors", "recovered_after_error", "repaired_arguments",
                 "text_fallback_tool_calls", "revisions", "grounding", "llm_turns", "total_tokens", "duration_s"]])
LOG.stats()
'''),
]


# --------------------------------------------------------------------------- #
# 03 · NeMo Gym
# --------------------------------------------------------------------------- #
GYM = [
    md("""
# RiskLens 03 · NeMo Gym: a verifiable environment for CECL investigations

""" + badge("03_NeMo_Gym_Environment.ipynb") + """

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) runs agents against **environments** and scores them with a **verifier**, for evaluation and for RL training. RiskLens is a natural fit: we own the simulator, so we know the right answer to every question.

| NeMo Gym concept | RiskLens |
|---|---|
| Dataset (JSONL, one task per row) | the 15 demo questions + paraphrases; each row carries the system prompt, the agent's tool schemas and `verifier_metadata` (expected facts from the seeded ground truth) |
| Resources server (tools + verifier) | `resources_servers/risklens_cecl/app.py`: all 30 deterministic tools as `POST /<tool>` endpoints, plus `/verify` |
| Agent harness | Gym's `simple_agent` (model ↔ tools loop) |
| Model server | Nemotron 3.5 Lightning on vLLM (local / Brev) or the NVIDIA API, through Gym's `vllm_model`; or our mock planner |

**Reward** (`config/gym.yaml`, each component in [0, 1]):
`0.55·correctness + 0.20·grounding + 0.10·tool_selection + 0.10·call_validity + 0.05·efficiency`, × 0.25 if the answer used no successful tool call (every episode replays the same story, so memorised answers must not pay), 0 with no answer.

**Local or Colab.** NeMo Gym needs Python ≥ 3.13.14. The setup cell installs it: into this environment if the Python is new enough, otherwise (e.g. Colab) into a separate Python 3.13 venv that only the `gym` command uses. The first server start builds one venv per Gym server (a few minutes); later starts take seconds. In Colab, use the `nvidia` policy or your Brev vLLM URL for section 5.
"""),
    md("## 0 · Setup"),
    code(BOOTSTRAP),
    code('''
import json, shutil
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display
from src.evaluation import gym_runner, verifier
from src.evaluation.gym_dataset import build_rows, write_jsonl
from src.evaluation.local_eval import evaluate
from src.evaluation.trajectories import rollouts_to_sft
from src.llm.provider import get_provider

pd.set_option("display.max_colwidth", 90)
from src.utils.notebook import ensure_gym_cli
GYM = ensure_gym_cli()          # installs NeMo Gym if missing (a Python 3.13 venv when the kernel is older, e.g. Colab)
print("gym CLI:", GYM or "unavailable: sections 1-3 still run in-process")
if GYM:
    print(subprocess.run(["gym", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
'''),
    md("""
## 1 · The dataset
`build_rows()` turns the task bank into Gym rows. `example` = the 15 canonical demo questions, `train` = those plus paraphrases, `validation` = held-out paraphrases.
"""),
    code('''
splits = build_rows()
for name, rows in splits.items():
    write_jsonl(rows, gym_runner.DATA_DIR / f"{name}.jsonl")
    print(f"{name:<11}{len(rows):>3} rows -> {(gym_runner.DATA_DIR / f'{name}.jsonl').relative_to(REPO_DIR)}")

row = splits["example"][4]
p = row["responses_create_params"]
print("\\nquestion :", p["input"][1]["content"])
print("tools    :", len(p["tools"]), "->", ", ".join(t["name"] for t in p["tools"][:6]), "...")
print("system   :", p["input"][0]["content"][:300].replace("\\n", " "), "...")
pd.DataFrame(row["verifier_metadata"]["checks"])
'''),
    md("""
## 2 · The reward function, by example
Three answers to *"By how many dollars did the allowance change?"*: a grounded one, one with invented numbers, and one that names the right figure without calling any tool.
"""),
    code('''
from src.tools import call_tool
task = next(r["verifier_metadata"] for r in splits["example"] if r["id"] == "q03_dollar_change")
rec = call_tool("compare_allowance", {"previous_date": "2026-07-31", "current_date": "2026-08-31"})
calls = [verifier.ToolCallTrace("compare_allowance", {}, {"status": "ok", "result": rec.result})]
c = rec.result
cases = {
    "grounded":     (f"The allowance increased by {M(c['change'])} ({c['change_pct']:.1f}%), from {M(c['previous_allowance'])} to {M(c['current_allowance'])}. [compare_allowance]", calls),
    "hallucinated": ("The allowance increased by $25.17M, driven by $412.9M of new loans.", calls),
    "no tools":     (f"The allowance increased by {M(c['change'])}.", []),
}
rows = []
for name, (answer, cs) in cases.items():
    res = verifier.score(answer, cs, task)
    rows.append({"case": name, "reward": res.reward, **res.components, "answer": answer})
pd.DataFrame(rows).set_index("case")
'''),
    md("""
## 3 · Baseline in-process (no servers)
`evaluate()` runs each row with RiskLens's own agent loop and the same verifier. The mock planner is a keyword-matching script: a floor, not a model. Note where it breaks on paraphrases.
"""),
    code('''
mock_local = evaluate(get_provider({"provider": "mock"}), split="train")
print(json.dumps(mock_local.summary(), indent=1))
mock_local.frame().sort_values("reward")[["task_id", "reward", "c_correctness", "c_grounding", "tool_calls", "checks_passed"]].head(8)
'''),
    md("""
## 4 · The same evaluation through real NeMo Gym servers
`run_gym_eval()` runs `gym env start` (resources server + `simple_agent` + `vllm_model` pointed at the policy), then `gym eval run --no-serve`, then stops the servers. For the mock policy it also starts `scripts/serve_policy.py`, an OpenAI-compatible endpoint, so the whole pipeline runs without a GPU.

Equivalent shell commands (from the repo root):
```bash
python -m scripts.serve_policy --provider mock --port 8011 &
gym env start --resources-server risklens_cecl --model-type vllm_model \\
    --model mock-planner --model-url http://127.0.0.1:8011/v1 --model-api-key none
gym eval run --no-serve --agent risklens_cecl_simple_agent \\
    --input resources_servers/risklens_cecl/data/train.jsonl --output data/outputs/gym/mock_train_rollouts.jsonl
```
"""),
    code('''
mock_path = gym_runner.run_gym_eval("mock", split="train", stream=False)
mock_gym = gym_runner.load_rollouts(mock_path)
print(gym_runner.summarize(mock_gym))
parity = mock_gym.set_index("task_id")["reward"].to_frame("gym").join(mock_local.frame().set_index("task_id")["reward"].rename("in_process"))
print("max |gym - in-process| reward difference:", float((parity.gym - parity.in_process).abs().max()))
mock_gym[["task_id", "reward", "tool_calls", "tools", "failed_checks"]]
'''),
    md("""
## 5 · Nemotron 3.5 Lightning in the environment
Point Gym's model server at Nemotron. `vllm` = local or Brev vLLM (`deploy/brev_serve_nemotron.sh`), `nvidia` = NVIDIA API catalog. `NUM_REPEATS > 1` samples each task several times, which shows how consistent the model is (and is what RL needs).
"""),
    code('''
POLICY = os.environ.get("RISKLENS_GYM_POLICY", "auto")      # auto | vllm | nvidia | mock
NUM_REPEATS = int(os.environ.get("RISKLENS_GYM_REPEATS", "2"))
SPLIT = "example"

vllm = gym_runner.POLICIES["vllm"]
if POLICY == "auto":
    POLICY = "vllm" if endpoint_up(vllm.url) else ("nvidia" if secret("NVIDIA_API_KEY") else None)
if POLICY == "nvidia":
    os.environ["NVIDIA_API_KEY"] = secret("NVIDIA_API_KEY") or ""
print("policy:", POLICY or "none reachable: start vLLM or add NVIDIA_API_KEY to .env, then re-run this cell")

results = {"mock": mock_gym}
if POLICY:
    path = gym_runner.run_gym_eval(POLICY, split=SPLIT, num_repeats=NUM_REPEATS, stream=False)
    results[POLICY] = gym_runner.load_rollouts(path)
    print(gym_runner.summarize(results[POLICY]))
    display(results[POLICY].groupby("task_id")[["reward", "correctness", "grounding", "tool_calls"]].mean().sort_values("reward"))
'''),
    md("## 6 · Compare policies"),
    code('''
comps = ["reward", "correctness", "grounding", "tool_selection", "call_validity", "efficiency"]
table = pd.DataFrame({name: df[df.task_id.isin([r["id"] for r in splits["example"]])][comps].mean() for name, df in results.items()}).T
display(table.round(3))
ax = table.T.plot.bar(rot=20, title="NeMo Gym reward components on the 15 demo questions", figsize=(10, 4))
ax.set_ylim(0, 1.05); plt.tight_layout(); plt.show()
if len(results) > 1:
    by_cat = pd.concat({n: df.groupby("category")["reward"].mean() for n, df in results.items()}, axis=1)
    display(by_cat.round(3))
'''),
    md("""
## 7 · Inspect a rollout
What the policy actually did: tool calls in order, the verifier's checks and the final answer.
"""),
    code('''
name = next(iter(n for n in results if n != "mock"), "mock")
df = results[name]
worst = df.sort_values("reward").iloc[0]
print(f"[{name}] {worst.task_id}  reward={worst.reward:.3f}  tools={worst.tools}")
print("failed checks:", worst.failed_checks or "none")
display(Markdown(worst.answer or "_(no answer)_"))
'''),
    md("""
## 8 · From rollouts to training data
Rollouts that pass the verifier (reward ≥ 0.8) become supervised fine-tuning examples in OpenAI chat format with tool calls: grounded, correct investigations only. This is the hand-off to LoRA SFT (Fine-Tuned track) and, with the same resources server as the reward, to RL with NeMo RL (GRPO) on Brev.
"""),
    code('''
paths = sorted((REPO_DIR / "data" / "outputs" / "gym").glob("*_rollouts.jsonl"))
sft_path = REPO_DIR / "data" / "outputs" / "gym" / "sft_trajectories.jsonl"
sft = rollouts_to_sft([p for p in paths if not p.name.startswith("mock")] or paths, min_reward=0.8, out=sft_path)
print(f"{len(sft)} examples -> {sft_path.relative_to(REPO_DIR)}  (from {[p.name for p in paths]})")
if sft:
    ex = sft[0]
    print("roles:", [m["role"] for m in ex["messages"]], "| task:", ex["metadata"]["task_id"], "| reward:", ex["metadata"]["reward"])
'''),
    md("""
**Next steps**
- `python -m scripts.build_gym_dataset --agent credit_agent` builds the same environment with a specialist's 15-tool allow-list, so each Phase 4 specialist can be evaluated (and later trained) on its own slice.
- Add story variants (credit-led, growth-led, a real model change) to the generator so the policy has to investigate rather than memorise; each variant ships its own ground truth and Gym rows.
- RL: NeMo RL's GRPO recipes consume a Gym resources server as the reward. Full RL on Nemotron 3.5 Lightning needs far more GPUs than a hackathon box, so the weekend deliverable is the environment, the baseline scores and the SFT set.
"""),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in {"01_CECL_Workflow.ipynb": WORKFLOW, "02_Agent_Investigator.ipynb": AGENT,
                        "03_NeMo_Gym_Environment.ipynb": GYM}.items():
        nbf.write(notebook(cells), OUT / name)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
