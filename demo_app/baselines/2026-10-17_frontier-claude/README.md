# Frontier reference: Claude Opus 5.5 on the evaluation bank

Upper-bound reference for the small models: a frontier model answering all 15 evaluation-bank questions
(q01–q15) with exactly the inputs the small models get.

- Same system prompt (`ORCHESTRATOR_PROMPT`), same 33 tools through `call_tool`, same seeded data (seed 20260831).
- Ground truth read only after each answer was submitted; scored by `src/evaluation/verifier.py` (`config/gym.yaml` weights).
- Driven by hand from a chat session with `scripts/frontier_validate.py`, so there is no meaningful model time;
  compare quality, tool choice and call counts, not speed.

| | Mean reward |
|---|---|
| All 15 questions | **0.997** |
| 5 bench questions (q02, q05, q07, q13, q15) | **1.000** |

Points lost: q08 (0.992) cited "50%" from a field name, not a tool value (grounding); q14 (0.967) one malformed
argument (`top_n` instead of `n`), recovered from the tool hint.

## Files
- `frontier_reference.json` — per question: reward, components, checks, tools used, grounding and full answer.
- `runs.jsonl`, `SUMMARY.md` — the raw run-log records (same format as the other baselines).

## Compare your model
```bash
python -m demo_app.bench --label "trained 4B" --provider vllm --base-url http://localhost:11434/v1 --model <name>
python -m scripts.compare_to_frontier --label "trained 4B" --markdown evidence/vs_frontier.md
python -m scripts.compare_to_frontier --runs demo_app/baselines/<folder>/runs.jsonl   # any exported baseline
```
Runs are matched by `task_id` (paraphrases count toward their base question); means use only the questions
both sides answered, and "% of frontier" = your mean / frontier mean on those questions.
