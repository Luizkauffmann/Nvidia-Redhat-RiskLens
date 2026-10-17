# Legacy Dataiku package: usage audit

Source: `ecl_allowance_workbench_claude_handoff.zip` (Dataiku DSS Standard Webapp, V3).
Method: cross-referenced every dataset name against the backend, the frontend API calls and the four recipes, then checked which configuration fields the calculation code actually reads.

## Datasets

| Dataset | Read by | Actually drives a calculation? | RiskLens decision |
|---|---|---|---|
| portfolio_snapshot | generator, backend | Yes | **Ported** → `loan_snapshots.parquet` (same product labels, new fields: DSCR, utilization, model IDs) |
| counterparty_snapshot | generator, backend | Yes | **Ported** → `counterparties.parquet` |
| macroeconomic_scenarios | generator, backend | Yes | **Ported, reshaped**: wide quarterly paths per vintage/scenario (6 variables, not 11 × 11 geographies) |
| scenario_config | backend | **No**, only a version label is built from it | **Replaced** by `scenario_weights.csv` (weights are data, per date) |
| stage_rules | backend | **No**, staging is hard-coded in the generator | **Replaced** by the `ifrs9:` block in `methodology.yaml`, which the engine reads |
| model_group_config | generator, backend | Display/metadata only | **Replaced** by `model_metadata.csv` + `model_assignments.csv` + `config/models.yaml` |
| ratings_mapping | generator only | No | Folded into generator constants |
| installment_schedule | generator only | **No**, nothing consumes it | Dropped (engine uses the amortization profile) |
| default_recovery_history | generator only | **No** | Dropped (LGD is parametric; could feed LGD calibration later) |
| cecl_run_manifest | backend | Yes (history) | **Replaced** by `allowance_summary.csv` |
| cecl_instrument_results | backend | Yes (history) | **Replaced** by `ecl_results.parquet` (loan × framework) |
| cecl_attribution_results | backend | Yes | **Replaced**: legacy amounts were rescaled to sum to the total (see below) |
| cecl_feature_contributions | generator only | **No**, not shown in the UI | Replaced by `pd_logit_components()` (exact, additive on logit scale) |
| ecl_method_config | backend | Partly (FIXED / curve / matrix branches) | Concept kept as `methodology.yaml → products`; only scorecard PD is implemented |
| ecl_math_config | backend | **No**, displayed only; all 8 rows have the same semantics | Concept kept as `methodology.yaml → calculation` (horizon, discounting, formula), which the engine executes |
| ecl_execution_profiles | backend | No (Python is the only real engine) | Dropped |
| fixed_parameter_values, lifetime_pd_curves, transition_matrices, prepayment_curves | backend | Yes, but only in the V3 sandbox run path | Deferred: possible future `pd_method` options |
| ecl_run_requests, cecl_governance_status, cecl_release_log | backend | Empty, simulated workflow | Dropped for the hackathon |

## Code

| File | Notes |
|---|---|
| `recipes/generate_demo_data.py` / `dataiku_generate_demo_data_recipe.py` | Two near-duplicate generators (about 2,100 diff lines apart). The PD/LGD/macro ideas were ported; the rest was rewritten. |
| `recipes/dataiku_generate_training_datasets_recipe.py` | Writes training datasets that are not in the package. Dropped. |
| `recipes/generate_ecl_v3_datasets.py` | Writes the V3 configuration CSVs. Dropped. |
| `webapp/backend.py` | Imports `dataiku`, so it can't run outside DSS. Route ideas (`/overview`, `/compare`) map onto Phase 2 tools. |
| `webapp/app.js`, `body.html`, `style.css` | Seven-tab UI. Layout ideas reusable for Streamlit; the code is not. |

## Findings that changed the design

1. **The attribution was forced to balance.** `build_attribution` multiplied the components by `target / sum` (accepted when the factor was between 0.45 and 1.75), used an unexplained `× 2.2` on the credit effect, and hard-coded model version = 0. A reconciliation check on that output always passes. RiskLens uses a sequential counterfactual walk whose residual is measured, not forced (see `test_residual_detects_unattributed_input_change`).
2. **All shipped runs were IFRS 9.** The CECL branch existed but was never run. RiskLens runs both and defaults to CECL.
3. **The private SAS reference deck** was not in the package and must stay out of this repository.
