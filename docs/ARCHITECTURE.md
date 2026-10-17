# Architecture & decisions (Phases 1–2)

## Layers

```
config/*.yaml ─┐
               ├─> src/data/generator.py ──> data/generated/*  (+ data/ground_truth/, evaluation only)
               │                                   │
               └─> src/cecl/engine.py  <── src/data/loader.py (DataStore)
                         │
                   src/cecl/attribution.py   (sequential counterfactual waterfall)
                         │
        Phase 2:   src/tools/*  (30 JSON tools via call_tool: the only thing agents call)
        Phase 3+:  src/agents/*, src/llm/*, app/streamlit_app.py
```

The agents never compute numbers. Every quantitative claim must come from a tool that calls the engine or the attribution.

## ECL engine (`src/cecl/engine.py`)

`ECL_i = Σ_y MarginalPD_i,y · LGD_i,y · EAD_i,y · DF_i,y`, using annual steps and a fractional stub year.

| Component | Implementation |
|---|---|
| PD | Logistic scorecard per model ID (`config/models.yaml`), plus a delinquency add-on, plus a macro logit shift |
| Macro | Quarterly paths for 8 reasonable & supportable quarters, then straight-line reversion over 4 quarters, averaged to annual |
| LGD | Linear in features (LTV), plus macro sensitivity, with floor and cap |
| EAD | Balance + CCF × undrawn. CECL CCF is 0 for unconditionally cancellable card lines. Amortizing loans use a linear amortization profile |
| Life | Contractual remaining term; revolving products use a behavioral life (3 years); capped at 30 years |
| Discounting | CECL: none (loss-rate style, permitted under ASC 326). IFRS 9: EIR |
| CECL | Lifetime ECL for all loans |
| IFRS 9 | Stage 1 = 12-month; Stage 2/3 = lifetime. SICR = base PD ≥ 2× origination PD, ≥ 3-notch downgrade, ≥ 30 DPD, or watchlist. Stage 3 = default or ≥ 90 DPD |
| Scenarios | Baseline / adverse / severe adverse, probability-weighted |
| Allowance | Σ ECL + management overlays |

The engine is checked against a hand calculation (toy bullet loan) and against an independent loop-based implementation on about 170 real loans (`tests/test_engine_reference.py`).

## Attribution (`src/cecl/attribution.py`)

The walk goes: previous inputs → exits → balances/terms → new loans → credit attributes → macro (one variable at a time) → scenario weights → model IDs → overlays. Each step re-runs the engine.

- Impacts telescope, so they sum to the change by construction.
- **The residual is real.** The final hybrid state is compared with an independent run on the actual current inputs. Any engine input that changed but isn't in a swap group (`methodology.yaml → attribution.*_fields`) shows up as a residual.
- **Order matters for interactions.** The order is configurable (`step_order`, `macro_order`) and reported in `method.step_order`. Two order-independent checks complement the walk:
  - `macro_shapley`: exact Shapley over the 6 macro variables (2^6 = 64 projections, ~19 s), per loan, so portfolio × variable splits add up.
  - `driver_shapley`: exact Shapley over the 5 driver groups (2^5 = 32 engine runs, ~31 s).
  The reconciliation tool compares the forward walk, the reversed walk and Shapley, and flags a WARNING if they disagree on the primary driver.
- The step logic is a set of *switches* (`_apply`): the state after a set of switches does not depend on their order, which is what makes reordering and Shapley coalitions well defined.
- Results are produced at loan level, so borrower and portfolio drill-downs are exact sums of the same numbers.

## Engine performance

`ECLEngine.run` is split into `_prepare` (scorecards, horizons, EAD and LGD profiles: everything macro-independent) and `_project` (one scenario path). `macro_sweep` reuses one preparation for many macro frames, about 0.25 s per extra frame on 50k loans instead of 0.75 s per full run. A test checks it is numerically identical to full runs.

## Tool layer (`src/tools`)

| Piece | Role |
|---|---|
| `@tool(category, agents)` | Registers a function; the JSON schema comes from its signature (types, `Literal` enums, defaults), with live enums for dates, portfolios, variables and scenarios |
| `call_tool(name, args)` | Single entry point. Parses JSON-string args, normalises dates and names, validates, runs, makes output JSON-safe, returns a `ToolCallRecord` |
| Errors | `unknown_tool`, `invalid_arguments`, `malformed_arguments`, `not_found`, `execution_error`, each with a `hint` the model can use to retry |
| Provenance | `call_id`, tool, requested vs normalised parameters, timestamp, duration, status, `result_sha256` (deterministic: the same call gives the same hash) |
| `ToolCallLog` | In-memory + optional JSONL log with stats (success rate, repaired args, errors by type) for the tool-calling evaluation |
| `ToolContext` | Shares one `DataStore`, engine and attributor; caches attribution and Shapley in memory and on disk (`data/outputs/cache`, keyed by data files + numeric config) |
| `config/agents.yaml` | Per-agent allow-lists (11–15 tools each for specialists). A test keeps them in sync with the `@tool` tags |

Tools never compute from the ground truth (a test greps `src/tools`). "Reported" figures come from the stored official runs; "explained" figures come from engine re-runs. The reconciliation is therefore a genuine comparison (tests tamper with the booked allowance and expect PASS / WARNING / FAIL at the configured thresholds).

## Decisions

| Decision | Why |
|---|---|
| Plain Parquet/CSV and pandas, no platform SDK | Runs on Brev or a laptop; cuDF can be swapped in inside `loader.py` later |
| Legacy `product_type` labels kept; `portfolio` added as a reporting group | Continuity with the legacy data; matches the demo's CRE / C&I / Mortgage / Auto / Card buckets |
| Ground truth kept in a separate folder and only readable via `read_ground_truth()` | Lets us claim the agents found the cause independently |
| Story knobs in `config/demo_story.yaml` | Reproducible, adjustable scenario; the seed lives in `settings.yaml` |
| Commercial maturities mostly renew (85%) | Without renewals, CRE balloon maturities swamped the story with runoff |
| IFRS 9 SICR uses ex-macro base PD | Keeps staging a credit-quality signal; macro still moves PD levels |
| Scenario weights shift (10% → 12% severe), and no model change | Lets the Model Risk Agent answer "models didn't change, but the weights did" |
| CRE cohort shock softened (DSCR drop 0.07–0.12, value decline 4–7%, watchlist 30%) | With the original shock, macro led only in the default order (reversing the walk made credit primary) and cohort PD hit 5.9%. Now cohort PD is 2.0% → 5.2%, the change is +$19.24M, and macro leads under every method |
| Reported change uses the sequential walk; Shapley is a robustness check | The walk gives exact loan-level, step-by-step numbers that agents can drill into; Shapley shows how much of each driver is interaction |
| Exit reasons inferred (matured / defaulted / paid off) from the previous snapshot | The snapshots don't store exit reasons; inference is deterministic and labelled as such |
| Scenario what-ifs re-weight stored scenario ECL | Weighted ECL is linear in the weights, so this is exact and instant |
