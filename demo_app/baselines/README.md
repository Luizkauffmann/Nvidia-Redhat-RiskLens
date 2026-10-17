# Model run baselines

Fixed benchmark for comparing models on the demo: the same five evaluation-bank questions
(`demo_app/runlog.py: BENCH_TASKS` = q02, q05, q07, q13, q15), one at a time, single agent, reasoning off.
Each run records timing (total, model, tools), turns, tokens, tool calls, grounding, and the
Gym verifier reward (`src/evaluation/verifier.py`).

| Folder | What | Setup |
|---|---|---|
| `2026-10-17_frontier-claude/` | Frontier reference (Claude Opus 5.5, hand-driven, same prompt/tools/verifier) on all 15 questions: 0.997 (1.0 on the bench five) | Chat session via `scripts/frontier_validate.py`; compare with `python -m scripts.compare_to_frontier` |
| `2026-10-17_stock-4b/` | **Invalid, kept for the record.** Stock `nemotron-3-nano:4b` with Ollama's default 4,096-token context, which silently truncated every prompt to ~2k tokens (the agent's prompt is ~11.8k). The model never saw its full instructions or the question. | Ollama 0.30.11 (Q4_K_M), MacBook M4 16 GB |
| `2026-10-17_stock-4b-32k/` | Same model with a 32k context (`deploy/ollama/Modelfile.nemotron-nano-32k`): the valid baseline | Ollama 0.30.11 (Q4_K_M), num_ctx 32768, MacBook M4 16 GB, nothing else heavy running |

## Reproduce / add a model

```bash
python -m demo_app.server &          # not required; bench runs in-process
python -m demo_app.bench --label "trained 4B" --provider vllm \
    --base-url http://localhost:11434/v1 --model <ollama-model-name>
python -m demo_app.bench --summary   # compare with everything in data/outputs/demo_app_runs.jsonl
python -m demo_app.bench --export demo_app/baselines/<date>_<name>
```

Keep the setup identical when comparing (same machine, Ollama build, quantization, reasoning off, single
agent), so differences come from the model, not the environment.

## What the valid baseline (32k context) shows

| | Truncated 4k (invalid) | **32k context** |
|---|---|---|
| Mean quality (0-1) | 0.39 | **0.77** |
| Questions fully right | 0 / 5 | 3 / 5 (q02, q07, q15 at 1.0; q05 0.86) |
| Failed runs | 0 | 1 (q13: gave up after repeated malformed tool calls, scored 0) |
| Median time / question | 116 s | 115 s (almost all model time) |
| Tool calls / question | 0.8 | 1.8 |

With its full prompt the stock 4B picks the right tools (`get_allowance_summary`, `compare_allowance` +
`calculate_portfolio_attribution`, `identify_top_contributors`, `reconcile_allowance_change`). It still failed
"Could this increase just be because we changed models?" by writing tool calls as text until the agent gave up.
Speed is the other gap: ~2 minutes per question on an M4, mostly reading the ~12.7k-token prompt.

**Scoring note:** a run that gives up ("[RiskLens could not complete ...]") now scores 0. Before this fix the q13
run scored 0.988, because the verifier's "says no model version changed" pattern (a negation near "model")
matched the give-up message itself. `effective_quality()` in `demo_app/runlog.py` applies the rule to logged runs;
the pattern in `src/evaluation/tasks.py` could be tightened too.

## Context window: check it first

Ollama's default context is 4,096 tokens and it truncates longer prompts **silently**: a run log where every
model turn shows the same ~2,050 prompt tokens is the tell. The single agent needs ~11.8k tokens before any tool
result and each specialist 3.5-5k, so use `ollama create nemotron-3-nano-4b-32k -f deploy/ollama/Modelfile.nemotron-nano-32k`.

## What the first (truncated) 2026-10-17 run showed

- **Time is almost all model time**: median 116 s per question, of which 114 s waiting on the model
  (about 10 tokens/s on the M4). The deterministic tools take seconds or less.
- **Quality 0.39** vs 1.0 for the mock planner. The 4B model answered 2 of 5 questions without calling
  a tool, and twice called `build_attribution_workbook` (the whole Excel pack) to look up one figure.
- **No unsupported numbers**: all answers passed the grounding check, but mostly because they contained no
  numbers at all (`NO_NUMBERS`), which the verifier's correctness checks then penalise.

For a trained model of the same size, per-token speed stays about the same (same size, same hardware);
it gets faster end to end by choosing the right tool first and writing shorter answers.
