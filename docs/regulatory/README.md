# Regulatory reference library — CECL & IFRS 9

Source corpus for the planned **regulatory agent** (RAG with citations) and for the pitch.
Public-domain / freely redistributable documents are fetched into `docs/regulatory/sources/`
with `scripts/fetch_regulatory_docs.sh`. Copyrighted standards (FASB ASC, IFRS Foundation,
Big-4 guides) are **linked only, never committed**.

Last reviewed: 2026-10-17.

## 1. Accounting standards (link only — copyrighted)

| Ref | What it governs | RiskLens relevance | Link |
|---|---|---|---|
| FASB ASC 326 (ASU 2016-13) | CECL: lifetime ECL at day 1; past events + current conditions + reasonable & supportable forecasts; no prescribed method; pool when risk characteristics are similar | Our PD×LGD×EAD engine is one permitted method; the disclosure of *changes in factors influencing the estimate and reasons* is the legal hook for "explain every dollar" | https://asc.fasb.org/326 |
| ASU 2022-02 | Removes TDR accounting; gross write-offs by vintage disclosure | Vintage view for Portfolio Agent | https://www.fasb.org |
| ASU 2025-05 (Jul 2025) | Practical expedient for current AR / contract assets (ASC 606); eff. FY beginning after 15 Dec 2025 | Out of scope for loans | https://www.fasb.org |
| ASU 2025-08 (Nov 2025) | "Purchased seasoned loans" use PCD gross-up | Future scenario: acquired book | https://www.fasb.org |
| IFRS 9 §5.5 | 3-stage ECL (12m vs lifetime), SICR (30 dpd rebuttable), probability-weighted scenarios, EIR discounting, POCI | `stage` + IFRS 9 mode of `src/cecl/engine.py` | https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/ |
| IASB PIR of IFRS 9 impairment (Jul 2024) | Works as intended; more guidance needed on SICR, overlays, forward-looking info; IFRS 7 credit-risk disclosure project | Overlay transparency is a known gap | https://www.ifrs.org/projects/completed-projects/2024/post-implementation-review-of-ifrs-9-impairment/ |

## 2. US supervisory (public domain — fetched)

| File in `sources/` | Issuer / date | Key points | RiskLens relevance |
|---|---|---|---|
| `us_interagency_policy_statement_ACL_rev2023.pdf` | Fed, FDIC, NCUA, OCC — 2020, revised Apr 2023 | Design, documentation and validation expectations for ACL; board/management responsibilities; Q-factors | Reconciliation + provenance logs map onto documentation expectations |
| `us_interagency_ACL_and_credit_risk_review_2020_release.pdf` | Same agencies — May 2020 | Release of the ACL statement + credit risk review systems guidance | Independent review of agent findings |
| `us_capital_rule_CECL_transition_2019.pdf` | Fed, FDIC, OCC — Feb 2019 | Optional 3-year phase-in of day-1 CECL capital impact (CET1) | Background |
| `us_capital_rule_CECL_revised_transition_2020.pdf` | Same — Sep 2020 | 2-year delay + 3-year phase-in, 25% scaling factor | Background (fully phased in) |
| `us_SR26-2_model_risk_guidance_2026_FDIC.pdf` | Fed SR 26-2 / OCC 2026-13 / FDIC — 17 Apr 2026 | **Supersedes SR 11-7 and SR 21-8**; risk-based, materiality-driven MRM; vendor models remain bank's responsibility | Model Risk Agent; the agents themselves are in scope → cite this, not SR 11-7 |
| `us_SR11-7_model_risk_2011_superseded.pdf` | Fed — 2011 | Historical: effective challenge, validation | Superseded; keep for comparison |
| `us_SEC_SAB119.pdf` | SEC staff — 2019 | Documentation & validation of CECL allowance for registrants | Audit trail requirements |

## 3. International supervisory (redistribution permitted with attribution — fetched)

| File in `sources/` | Issuer / date | Key points | RiskLens relevance |
|---|---|---|---|
| `intl_BCBS_d350_credit_risk_and_ECL_accounting_2015.pdf` | Basel Committee — Dec 2015 | 11 principles for ECL governance, methodology, validation, disclosure | Principle-level checklist for the regulatory agent |
| `eu_ECB_IFRS9_overlays_novel_risks_2024.pdf` | ECB Banking Supervision — Jul 2024 | Overlays for novel risks (climate, geopolitics, energy, supply chain); GDP-only treatment = bad practice; prefer in-model | Overlays must be attributed as an explicit step |
| `eu_ESRB_wp92_cyclicality_SICR_IFRS9.pdf` | ESRB WP 92 | SICR pro-cyclicality; PD-ratio (>3×) staging test | SICR rule for IFRS 9 mode |

Also relevant (not fetched): EBA/GL/2017/06 on credit-risk management and ECL accounting; ECB supervisory priorities 2026–2028 (novel risks); RBI ECL directions (27 Apr 2026, India).

## 4. How the agents should use this

1. Regulatory agent answers *"what does the rule say?"* with a citation (file + page); it never computes numbers.
2. Disclosure-style explanations (ASC 326-20-50) are built only from tool outputs.
3. Overlay changes are reconciled separately (ASC 326 / ECB 2024 / IASB PIR all single them out).
4. Agent decisions are logged with tool provenance to meet SR 26-2 and interagency documentation expectations.
