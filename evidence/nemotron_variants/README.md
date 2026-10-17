# Nemotron on a held-out story variant (`property_shock`)

Same model, server and 15 questions as [`../nemotron_baseline/`](../nemotron_baseline/README.md), but on
the `property_shock` story: CRE prices and rates worsen while unemployment stays flat, so the correct
"which macro variable?" answer is CRE prices, not unemployment. This story is held out of all training data.

| | |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, vLLM on one A100 80 GB, 2026-10-17 |
| Command | `run_gym_eval("vllm", split="example", variant="property_shock")` |
| Story | CECL $97.49M → $114.98M (+$17.49M); macro +$15.87M, of which CRE prices +$12.02M |

**Mean reward 0.959** (base story: 0.977), scored with the current verifier.

| Component | property_shock | base story |
|---|---:|---:|
| correctness | 0.978 | 0.978 |
| grounding | 0.920 | 0.974 |
| tool_selection | 0.933 | 1.000 |
| call_validity | 0.933 | 0.944 |
| efficiency | 1.000 | 1.000 |

| Question | Reward | Lost points on |
|---|---:|---|
| q14_cre | 0.769 | did not name a deteriorated CRE borrower; grounding 0.76 |
| q11_macro_variable | 0.882 | correct answer (CRE prices, $12.0M) taken from `calculate_macro_attribution`; the task requires `calculate_variable_attribution` (tool_selection 0) |
| q07_top_borrowers, q08_pd | 0.93 | first call errored, then retried successfully |
| q01, q03 | 0.96–0.98 | grounding: e.g. $97,492,788.84 written as "$97,492,788" (truncated, not rounded) |

## Verifier fix found by this run

The rollout file keeps the reward computed at collection time, **0.940**. Nemotron writes
"CRE price growth" with narrow no-break spaces (U+202F) and non-breaking hyphens (U+2011); the
"mentions" checks used plain spaces, so a correct q11 answer failed "names cre_price_growth"
(0.607 → 0.882 after the fix). `src/evaluation/verifier.py` now matches through Unicode spacing and
hyphens. Re-scoring the base-story run with the fix leaves it at 0.977 (no answer there was affected).
`rollouts_to_sft(..., rescore=True)` applies the current verifier to rollouts collected before the fix.
