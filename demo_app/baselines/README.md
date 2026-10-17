# Model run baselines

Fixed benchmark for comparing models on the demo: the same five evaluation-bank questions
(`demo_app/runlog.py: BENCH_TASKS` = q02, q05, q07, q13, q15), one at a time, single agent, reasoning off.
Each run records timing (total, model, tools), turns, tokens, tool calls, grounding, and the
Gym verifier reward (`src/evaluation/verifier.py`).

| Folder | What | Setup |
|---|---|---|
| `2026-10-17_stock-4b/` | Stock `nemotron-3-nano:4b` + the mock planner as reference | Ollama 0.30.11 (Q4_K_M), MacBook M4 16 GB, nothing else heavy running |

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

## What the 2026-10-17 baseline shows

- **Time is almost all model time**: median 116 s per question, of which 114 s waiting on the model
  (about 10 tokens/s on the M4). The deterministic tools take seconds or less.
- **Quality 0.39** vs 1.0 for the mock planner. The 4B model answered 2 of 5 questions without calling
  a tool, and twice called `build_attribution_workbook` (the whole Excel pack) to look up one figure.
- **No unsupported numbers**: all answers passed the grounding check, but mostly because they contained no
  numbers at all (`NO_NUMBERS`), which the verifier's correctness checks then penalise.

For a trained model of the same size, per-token speed stays about the same (same size, same hardware);
it gets faster end to end by choosing the right tool first and writing shorter answers.
