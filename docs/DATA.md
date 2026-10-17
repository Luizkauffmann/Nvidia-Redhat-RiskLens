# Data card: RiskLens synthetic CECL dataset

**Status:** synthetic, generated at the hackathon, published openly in [`data/published/`](../data/published) under Apache-2.0 together with the code that produced it.
**Generator:** [`risklens/data/generator.py`](../risklens/data/generator.py), driven by [`config/story.yaml`](../config/story.yaml), [`config/methodology.yaml`](../config/methodology.yaml), [`config/models.yaml`](../config/models.yaml) and the seed in [`config/settings.yaml`](../config/settings.yaml) (`seed: 20261017`).
**Reproduce:** `python -m scripts.generate_data`. Re-running produces the same loans and the same allowance to the cent (a test checks this). `SHA256SUMS.json` lists checksums of the committed files.

## Why synthetic

Loan-level bank data with borrower identities, ratings and loss estimates is confidential. Public loan datasets do not come with CECL allowance runs, macro scenario vintages or model registries, and they have no known answer to "why did the allowance move". Generating the data ourselves lets us inject a known story and then test whether the agent finds it. **No real borrower, bank or market data is used.** Names are generated from small word lists; any resemblance to real entities is coincidental. Macro levels are plausible round numbers chosen for the story, not forecasts.

## Files

| File | Rows | Description |
|---|---:|---|
| `loans_2026-07-31.parquet` | 50,000 | July month-end loan tape with booked CECL results |
| `loans_2026-08-31.parquet` | 51,798 | August month-end loan tape (429 loans exited, 2,227 originated) |
| `counterparties.parquet` | 50,508 | 45,060 retail customers, 5,448 commercial companies |
| `macro_scenarios.csv` | 72 | 2 forecast vintages × 3 scenarios × 12 quarters; one column per variable (6), plus scenario weight |
| `management_overlays.csv` | 4 | Qualitative overlays per date |
| `model_metadata.csv` | 16 | Model registry: PD, LGD, EAD models, status, dates |
| `run_manifest.json` | 2 | Booked official run per date: models, weights, overlays, allowance |
| `ground_truth.json` | – | The injected story, for evaluation only |
| `samples/*.csv` | – | Small CSV extracts for browsing on GitHub |

## Loan tape columns

| Column | Applies to | Meaning |
|---|---|---|
| `as_of_date` | all | Reporting date |
| `loan_id`, `counterparty_id` | all | Ids (`LN_0000001`, `CP_000001`) |
| `product_type`, `portfolio`, `segment` | all | 8 products grouped into 6 portfolios (CRE, C&I, Mortgage, Auto, Consumer, Credit Card); RETAIL / COMMERCIAL |
| `industry`, `property_type`, `state` | commercial / CRE / all | Industry (commercial), property type (CRE), US state |
| `origination_date`, `maturity_date`, `remaining_term_months` | all | Contract dates |
| `current_balance`, `undrawn_amount`, `utilization` | all / revolving | Drawn balance, undrawn commitment, utilization |
| `interest_rate`, `property_value` | all / secured | Contract rate; collateral value for LTV |
| `fico`, `dti` | retail | Credit score, debt-to-income |
| `ltv` | mortgage, auto, CRE | Loan-to-value |
| `dscr`, `leverage`, `rating_notch`, `internal_rating`, `watchlist_flag` | commercial | Debt service coverage, leverage, internal rating (notch 1 = AAA … 20 = CC) |
| `days_past_due`, `delinquency_bucket`, `default_flag` | all | Delinquency; default = 90+ DPD |
| `pd_model_id`, `lgd_model_id`, `ead_model_id`, `scenario_set_id` | all | Models and macro vintage used |
| `pd_12m`, `pd_lifetime`, `lgd`, `ead`, `ecl` | all | Booked results: scenario-weighted 12-month PD, lifetime PD, first-year LGD, exposure, lifetime ECL |
| `stage` | – | Always null: IFRS 9 staging does not apply under CECL |

Fields that do not apply to a product are null rather than zero (e.g. `dscr` for credit cards, `fico` for CRE).

## How the July book is built

| Product | Loans | Balance draw | Credit attributes |
|---|---:|---|---|
| Mortgage | 7,000 | lognormal, median $220k | FICO ~ N(745, 45), LTV U(0.45, 0.95) |
| Auto Loan | 11,000 | median $22k | FICO ~ N(705, 60), LTV U(0.60, 1.15) |
| Personal Loan | 7,000 | median $10k | FICO ~ N(690, 60) |
| Credit Card | 18,000 | limit median $8k × utilization Beta(2,5) | FICO ~ N(710, 65) |
| Commercial Term Loan | 2,900 | median $420k | rating ~ N(10.6, 1.6) notches, DSCR ~ N(1.5, 0.25), leverage |
| Revolving Credit Facility | 1,200 | commitment median $900k × utilization Beta(3,4) | as above |
| Equipment Lease | 1,600 | median $200k | as above |
| Commercial Real Estate | 1,300 | median $1.8M (+ 20-borrower cohort) | rating, DSCR, LTV U(0.45, 0.72) |

A small share of the book starts delinquent (retail: 2.3% past due, 0.3% in default). C&I facilities are spread over ~4,000 companies, a quarter of them in economic groups.

## The injected July → August story

Everything below is set in `config/story.yaml` and recorded in `ground_truth.json`.

| Driver | What changes | Expected effect |
|---|---|---|
| **Macroeconomic** | August forecast vintage is worse in every scenario: baseline unemployment peak deviation +0.40 → +1.35 pp, adverse +2.2 → +3.3, severe +4.3 → +5.0; GDP lower; CRE price growth −2 → −4 (baseline) and −25 → −28 (severe); rates elevated | Largest driver overall; unemployment the top variable; CRE prices the top variable inside CRE |
| **Credit deterioration** | 20 CRE office/retail borrowers ($5–12M each, largest $18M): 1–3 notch downgrades (largest: 3, BB+ → B+), DSCR −0.07 to −0.17, property values −4% to −8% (LTV up), ~30% added to the watchlist. Background: retail FICO drift, delinquency roll/cure, 1.5% commercial up/downgrades | CRE cohort EAD-weighted 12-month PD 1.8% → 5.5%; CRE the largest segment; `CP_046990` the largest contributor |
| **Portfolio growth** | 2,227 new loans including 12 new CRE loans totalling exactly $40M (multifamily/industrial) | Positive |
| **Paydowns / runoff** | Monthly payoffs by product, maturities (85% of matured commercial loans renewed), amortisation, 25% of defaulted loans charged off | Negative |
| **Model / methodology** | Scenario weights 60/30/10 → 58/30/12 (toward severe). **Same model versions** in both months (CRE_PD_v4; CRE_PD_v5 is in validation only). Overlays unchanged at $1.9M | Small positive; "was it a model change?" → no |

Resulting booked allowance: **$101.58M → $121.12M (+$19.54M, +19.2%)** on a balance of $8.05B → $8.08B.

## CECL methodology used to stamp the booked results

Lifetime, undiscounted expected credit loss with annual steps:

```
ECL_i = Σ_s w_s Σ_y  S_{i,s,y-1} · (1 − (1 − PD_{i,s,y})^{frac_y}) · LGD_{i,s,y} · EAD_{i,y}
PD_{i,s,y}  = logistic(scorecard_logit_i + Σ_v β_{portfolio,v} · deviation_{s,v,y})
LGD_{i,s,y} = clip(base_LGD_i + κ_portfolio · (−collateral_price_deviation_{s,y}))   (Mortgage: HPI, CRE: CRE prices)
EAD_{i,y}   = exposure profile (amortising / bullet / revolving with CCF) × (1 − prepay)^y
Allowance   = Σ_i ECL_i + management overlays
```

Macro paths: 8-quarter reasonable & supportable forecast, then reversion to the long-run anchor by quarter 12. Defaulted loans: ECL = LGD × EAD. Card lines are unconditionally cancellable, so their undrawn amount carries no allowance (CCF 0); commercial revolvers use CCF 0.5. Scorecard and LGD parameters are in `config/models.yaml`, keyed by model id.

## Intended use and limits

- Built to evaluate explanation and tool-calling systems for allowance analysis. It is not calibrated to any real portfolio and must not be used to estimate real credit losses.
- One designed story. Real allowance movements have more noise, data-quality issues and interacting causes.
- `ground_truth.json` is published so results can be checked, but the agent and tools must not read it (enforced by `tests/test_tools.py`).
