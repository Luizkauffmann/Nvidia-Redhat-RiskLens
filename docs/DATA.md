# Data card: RiskLens synthetic CECL dataset

**Status:** synthetic, published openly in [`data/published/`](../data/published) under Apache-2.0 together with the code that produces it.
**Generator:** [`src/data/generator.py`](../src/data/generator.py) and the official-run pipeline [`src/pipeline/cecl_run.py`](../src/pipeline/cecl_run.py), driven by [`config/demo_story.yaml`](../config/demo_story.yaml), [`config/methodology.yaml`](../config/methodology.yaml), [`config/models.yaml`](../config/models.yaml) and the seed in [`config/settings.yaml`](../config/settings.yaml) (`seed: 20260831`).
**Reproduce:** `python -m scripts.generate_demo_data` writes the working copy to `data/generated/` (git-ignored); `python -m scripts.publish_data` snapshots it to `data/published/` with CSV samples and `SHA256SUMS.json`. Regenerating produces byte-identical tables (only `run_log.csv`, which records the wall-clock execution time, differs, so it is not published).

## Why synthetic

Loan-level bank data with borrower identities, ratings and loss estimates is confidential. Public loan datasets do not come with CECL allowance runs, macro scenario vintages or model registries, and none has a known answer to "why did the allowance move". Generating the data lets us inject a known story and test whether the agents find it. **No real borrower, bank or market data is used.** Names are generated from small word lists; any resemblance to real entities is coincidental. Macro levels are plausible round numbers chosen for the story, not forecasts.

## Files

| File | Rows | Stands in for |
|---|---:|---|
| `loan_snapshots.parquet` | 99,892 (50,000 July + 49,892 August) | Loan servicing / core banking extract, 38 fields |
| `counterparties.parquet` | 58,067 (one row per borrower per date) | Borrower master, bureau data, financial spreading |
| `macro_scenarios.csv` | 72 | Economic forecast vendor: 2 vintages × 3 scenarios × 12 quarters, 6 variables |
| `scenario_weights.csv` | 6 | ALLL committee scenario weights, with approver and date |
| `management_overlays.csv` | 4 | Qualitative overlays per date |
| `model_metadata.csv` | 18 | Model inventory: PD / LGD / EAD models, versions, status |
| `model_assignments.csv` | 16 | Which model each product used on each date |
| `model_parameters.csv` | 100 | Scorecard and LGD parameters by model id |
| `ecl_results.parquet` | 199,784 | **Booked output**: per-loan CECL and IFRS 9 results (PD, LGD, EAD, ECL by scenario, stage) |
| `allowance_summary.csv` | 4 | Booked allowance per date and framework |
| `ground_truth.json` | – | The injected story, for **evaluation only** |
| `samples/*.csv` | – | Small CSV extracts for browsing on GitHub |

### Loan snapshot fields

| Field group | Fields | Applies to |
|---|---|---|
| Keys | `as_of_date`, `loan_id`, `counterparty_id`, `scenario_set_id` | all |
| Product | `portfolio`, `product_type`, `segment`, `property_type`, `industry`, `state` | all (`property_type` CRE only) |
| Contract | `origination_date`, `maturity_date`, `original_term_months`, `remaining_term_months`, `original_balance`, `current_balance`, `credit_limit`, `undrawn_amount`, `utilization`, `interest_rate` | all (limits and utilization for revolving) |
| Collateral | `collateral_value`, `ltv` | mortgage, auto, CRE |
| Retail credit | `fico`, `dti`, `origination_fico` | retail |
| Commercial credit | `rating_notch`, `internal_rating`, `origination_rating_notch`, `dscr`, `leverage`, `watchlist_flag` | commercial |
| Delinquency | `days_past_due`, `delinquency_bucket`, `default_flag` | all |
| Models | `pd_model_id`, `lgd_model_id`, `ead_model_id`, `origination_pd` | all |

Fields that do not apply to a product are null rather than zero. Rating notch 1 = AAA … 20 = CC.

## The July book

| Product | Loans | Portfolio |
|---|---:|---|
| Mortgage | 7,000 | Mortgage |
| Auto Loan | 11,000 | Auto |
| Personal Loan | 7,000 | Consumer |
| Credit Card | 18,000 | Credit Card |
| Commercial Term Loan | 2,900 | C&I |
| Revolving Credit Facility | 1,200 | C&I |
| Equipment Lease | 1,600 | C&I |
| Commercial Real Estate | 1,300 | CRE |

Total balance $7.61B. A small share starts delinquent (retail 2.3% past due, 0.3% defaulted; commercial 0.5%).

## The injected July → August story

All set in `config/demo_story.yaml`; the design and the realised events are recorded in `ground_truth.json`.

| Driver | What changes | Effect on CECL allowance (sequential waterfall) |
|---|---|---|
| **Macroeconomic** | The August forecast vintage is worse in every scenario. Peak unemployment deviation: baseline +0.40 → +1.10 pp, adverse +2.2 → +3.0, severe +4.3 → +5.0. GDP lower; CRE price growth −2 → −4 (baseline), −25 → −28 (severe); rates stay elevated | **+$10.34M**, the primary driver. Unemployment is the top variable (+$4.25M), then CRE prices (+$3.07M) |
| **Credit deterioration** | 20 CRE office/retail borrowers ($5–12M each; the largest, `CP_040408`, $18M): 1–2 notch downgrades (largest: 3, BB+ → B+), DSCR −0.07 to −0.12 (largest −0.17), property values −4% to −7% (LTV up), ~30% added to the watchlist, one 30–59 DPD. Background: retail FICO drift, delinquency roll and cure, 1.5% commercial up/downgrades | **+$7.48M**. Cohort EAD-weighted PD ~2.0% → ~5.2% |
| **Portfolio growth** | 1,592 new loans, including 12 new CRE loans totalling exactly $40M | **+$2.19M** |
| **Paydowns / runoff** | Payoffs by product, maturities (85% of matured commercial loans renewed), amortisation, 25% of defaulted loans charged off; 1,700 exits, $74.6M | **−$2.66M** |
| **Model / methodology** | Scenario weights 60/30/10 → 58/30/12 toward severe. **Same model versions** (CRE_PD_v4 both months; CRE_PD_v5 is in validation only). Overlays unchanged at $1.9M | **+$1.88M** |

Booked CECL allowance: **$99.06M → $118.30M (+$19.24M, +19.4%)**, residual $0.00. By portfolio, CRE carries $13.89M of the change. IFRS 9 on the same data: $43.89M → $57.11M.

## Methodology used for the booked results

Per loan, lifetime (CECL) or 12-month / lifetime by stage (IFRS 9):

```
ECL_i = Σ_s w_s Σ_y  MarginalPD_{i,s,y} · LGD_{i,s,y} · EAD_{i,y} · DF_{i,y}
PD_{i,s,y} = logistic(scorecard_logit_i + macro_shift_{portfolio,s,y})
CECL: lifetime, undiscounted.   IFRS 9: Stage 1 = 12-month, Stage 2/3 = lifetime, EIR-discounted.
Defaulted loans (90+ DPD): PD = 1.   Allowance = Σ ECL + management overlays.
```

Macro paths use an 8-quarter reasonable & supportable period, then revert to the long-run anchor. Parameters are in `config/methodology.yaml` and `config/models.yaml`; the engine is described in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Intended use and limits

- Built to evaluate explanation, multi-agent and tool-calling systems for allowance analysis. It is not calibrated to any real portfolio and must not be used to estimate real credit losses.
- One designed story. Real allowance movements have more noise, data-quality issues and interacting causes.
- `ground_truth.json` is published so results can be checked, but tools, agents and the NeMo Gym resources server never read it (enforced by tests); expected facts reach the verifier only through dataset rows.
