# RiskLens AI — working notes for Claude

Multi-agent CECL / IFRS 9 allowance investigator for the NVIDIA × Red Hat Small Models Hack (models ≤ 33B params).

## Golden rule
LLMs reason, route and explain. **Every number comes from deterministic code** (`src/cecl`, exposed to agents only through `src/tools`). Never let an agent compute or invent financial figures.

## Commands
```bash
pip install -r requirements.txt
python -m scripts.generate_demo_data   # rebuild data/generated + data/ground_truth (~20 s, seeded)
python -m scripts.precompute_analytics # warm attribution/Shapley cache before a demo (~75 s once)
python -m scripts.phase1_report        # sanity summary
python -m pytest                       # must stay green
```

## Layout
- `docs/PROJECT_ARCHITECTURE.md` — team-facing system map, data inputs, Nemotron backends, evaluation/fine-tuning/NeMo Gym plan, team workflow. Keep its status table current.
- `config/` — all tunables. `methodology.yaml` = ECL math + attribution order; `models.yaml` = scorecards by model ID; `demo_story.yaml` = injected July→August story; `reconciliation.yaml` = PASS/WARNING/FAIL thresholds.
- `src/cecl/engine.py` — CECL (lifetime, undiscounted) and IFRS 9 (staged, EIR-discounted) engine.
- `src/cecl/attribution.py` — sequential counterfactual waterfall (residual measured, never forced) + exact Shapley checks (macro variables, driver groups).
- `src/tools/` — 30 JSON tools; agents call them only via `call_tool` (validation, argument repair, provenance). Per-agent allow-lists in `config/agents.yaml`.
- Notebooks: `notebooks/RiskLens_CECL_Workflow.ipynb` (traditional run) and `RiskLens_Agent.ipynb` (ask Nemotron). Rebuild their Drive copies when they change.
- `src/pipeline/cecl_run.py` — traditional CECL workflow (seed → validate → book official run + run log). Used by `scripts/generate_demo_data.py` and `notebooks/RiskLens_CECL_Workflow.ipynb` (Colab, data on Google Drive via `RISKLENS_*_DIR` env vars).
- `src/llm/` — `get_provider()` (mock | nvidia | vllm | openai, from `config/agents.yaml` llm + `RISKLENS_LLM_*` env). `src/agents/base.py` — `ToolAgent` loop; `orchestrator.py` — Phase 3 agent + prompt; `grounding.py` — every number in an answer must match a tool result.
- `src/data/` — seeded generator and `DataStore` (Parquet/CSV, no platform SDK).
- `data/ground_truth/` — evaluation only. Tools and agents must never read it.

## Conventions
- Keep `product_type` labels exactly as in `methodology.yaml` (e.g. "Commercial Real Estate"); `portfolio` is the reporting group (CRE, C&I, …).
- New tools: decorate with `@tool(category, agents)`, give a one-line docstring (it becomes the schema description), raise `ToolError` with a `hint` for fixable mistakes, then regenerate the lists in `config/agents.yaml`.
- Tools return JSON-compatible dicts, never prose. Agents turn them into explanations, separating FACT from INTERPRETATION and keeping tool provenance.
- New engine behaviour needs a test; engine changes must keep `tests/test_engine_reference.py` (scalar cross-check) passing.
- Changing the story or calibration means regenerating data and re-checking `tests/test_attribution.py` against ground truth.
- Work incrementally; don't rewrite modules wholesale.

## Status
Phase 3 done: orchestrator agent + grounding check, tested with mock and scripted fake endpoints (108 tests); not yet run against live Nemotron (NVIDIA API is blocked from the dev sandbox). Story: CECL $99.06M → $118.30M (+$19.24M); macro primary under sequential, reverse and Shapley. Next: Phase 4 specialists (reuse `ToolAgent` with each agent's allow-list) + Streamlit UI.
