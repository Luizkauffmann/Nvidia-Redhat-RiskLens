# Nemotron baseline (base story, 15 demo questions)

First live run of the RiskLens agent environment against Nemotron, through real NeMo Gym servers.

| | |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, served by vLLM 0.27.1 on one A100 80 GB (Brev, 2026-10-17) |
| Environment | `resources_servers/risklens_cecl`, Gym's `simple_agent`, split `example` (15 questions), 1 attempt each |
| Story | base July → August 2026 (CECL $99.06M → $118.30M) |
| Command | `run_gym_eval("vllm", split="example")` (`src/evaluation/gym_runner.py`) |

**Mean reward 0.977** (mock planner on the same split: 0.988).

| Component (weight) | Mean |
|---|---:|
| correctness (0.55) | 0.978 |
| grounding (0.20) | 0.974 |
| tool_selection (0.10) | 1.000 |
| call_validity (0.10) | 0.944 |
| efficiency (0.05) | 1.000 |

23 tool calls over 15 questions. The 2 errors (`identify_top_contributors`, `calculate_lgd_migration`) were each followed by a successful retry of the same tool.

| Question | Reward | Lost points on (grounding = share of the answer's figures found in this episode's tool outputs) |
|---|---:|---|
| q14_cre | 0.801 | did not name a deteriorated CRE borrower (correctness 0.67); grounding 0.92 |
| q07_top_borrowers | 0.910 | first call errored (call_validity 0.5); grounding 0.80 |
| q09_lgd | 0.967 | first call errored (call_validity 0.67) |
| q05_drivers | 0.985 | grounding 0.92 |
| q02_current_allowance | 0.994 | grounding 0.97 |
| the other 10 | 1.000 | |

Reading: on the story it was evaluated on, base Nemotron already answers almost everything correctly
with grounded numbers, so this split leaves little headroom to demonstrate fine-tuning. That is why
the story variants (`../story_variants/`) exist: same questions, different correct answers.

Files: `vllm_example_rollouts.jsonl` (every rollout: messages, tool calls, answer, reward components,
per-check results), `vllm_example_rollouts_aggregate_metrics.json` and `quality_summary.json` (Gym's own
aggregates). Load with `src.evaluation.gym_runner.load_rollouts(path)`.
