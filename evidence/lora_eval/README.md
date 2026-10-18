# Base vs LoRA on the held-out story

Both models served from one vLLM (`deploy/brev_serve_nemotron.sh` with `LORA=`), same NeMo Gym environment,
same 15 questions on `property_shock` (never in training), 4 attempts each, scored by the same verifier.
`python -m scripts.eval_adapter` (thinking off) and `--thinking` (thinking on).

## Thinking off (the format the adapter is trained on) — [`property_shock_nothink.md`](property_shock_nothink.md)

| | base | LoRA |
|---|---:|---:|
| **mean reward** | **0.751** | **0.939** |
| correctness | 0.781 | 0.944 |
| grounding | 0.790 | 0.931 |
| call_validity | 0.758 | 0.982 |
| answers given with no tool call | 14 / 60 | 0 / 60 |
| output tokens per episode | 418 | 567 |

What changed:
- **Tool discipline.** Without reasoning, base Nemotron answered 14 of 60 questions without calling a
  tool, with figures it made up (e.g. "the July allowance was $45.3M"; the engine says $97.49M). The
  adapter always called a tool first. q01–q04 (the allowance figures) go from ~0.5 / 0.06 to ~0.99.
- q11 (which macro variable): 0.30 → 0.90.

What did not:
- **q14 (CRE drill-down), the hint-distillation target, did not transfer**: 0.754 → 0.732. On the held-out
  story the adapter still answers at portfolio level and never calls `identify_top_contributors` for CRE.
  40 hinted rows (10% of the data) over 2 epochs were not enough.
- q05–q07 dip (−0.10 to −0.16): the adapter sometimes answers from a broader tool (`compare_allowance`,
  `calculate_portfolio_attribution`) instead of the one the task requires.

Caveat: base with thinking **on** scored 0.959 on this story (1 attempt each, `../nemotron_variants/`).
The thinking-on comparison below is the like-for-like check of whether the adapter beats base at its best.

## Thinking on

Pending: `python -m scripts.eval_adapter --thinking`.

Raw rollouts: `eval_{base,lora}_property_shock_nothink_example_rollouts.jsonl`.
