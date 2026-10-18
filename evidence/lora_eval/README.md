# Base vs LoRA on the held-out story

Both models served from one vLLM (`deploy/brev_serve_nemotron.sh` with `LORA=`), same NeMo Gym environment,
same 15 questions on `property_shock` (never in training), 4 attempts each (60 episodes per row), same verifier.
`python -m scripts.eval_adapter` (thinking off) and `python -m scripts.eval_adapter --thinking`.

| | Mean reward | Correctness | Grounding | Answers with no tool call | Output tokens / episode |
|---|---:|---:|---:|---:|---:|
| base, thinking on | 0.964 | 0.978 | 0.929 | 0 / 60 | 970 |
| LoRA, thinking on | 0.959 | 0.978 | 0.930 | 0 / 60 | 1,042 |
| base, thinking off | 0.751 | 0.781 | 0.790 | 14 / 60 | 418 |
| **LoRA, thinking off** | **0.939** | **0.944** | **0.931** | **0 / 60** | **567** |

**What the adapter does:** it makes Nemotron's no-reasoning mode nearly as accurate as its reasoning mode
(0.939 vs 0.964) with about 40% fewer output tokens per episode (567 vs 970). Without it, the no-reasoning
mode scores 0.751 and answers 14 of 60 questions without calling a tool, with figures it made up
(e.g. "the July allowance was $45.3M"; the engine says $97.49M). The adapter always calls a tool first.

**What it does not do:** improve the reasoning mode (0.959 vs 0.964, within run-to-run noise), or teach the
CRE drill-down it was given hint-distilled examples for: q14 stays at 0.73–0.79 in every configuration,
and the adapter never calls `identify_top_contributors` for CRE on the held-out story. 40 hinted rows
(10% of the data) over 2 epochs were not enough.

Per question, thinking off ([`property_shock_nothink.md`](property_shock_nothink.md)): q01–q04 (allowance
figures) ~0.5 / 0.06 → ~0.99; q11 (macro variable) 0.30 → 0.90; q05–q07 −0.10 to −0.16, where the adapter
sometimes answers from a broader tool than the task requires. Thinking on: [`property_shock_think.md`](property_shock_think.md).

Output tokens include reasoning tokens when thinking is on; latency was not measured.
Raw rollouts: `eval_{base,lora}_property_shock_{think,nothink}_example_rollouts.jsonl`.
