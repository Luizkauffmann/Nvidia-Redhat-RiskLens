# Evidence

Results we cite in the submission, committed with how they were produced. Large regenerable data
(`data/`) stays out of git; what is here is small enough to read and reproduces from the commands given.

| Folder | Shows | Produced by |
|---|---|---|
| [`nemotron_baseline/`](nemotron_baseline/README.md) | Base Nemotron 3.5 Lightning in the RiskLens NeMo Gym environment: **0.977** mean reward on the 15 demo questions (mock planner 0.988) | `run_gym_eval("vllm", split="example")` on an A100 |
| [`nemotron_variants/`](nemotron_variants/README.md) | Base Nemotron on the held-out `property_shock` story: **0.959** (base story 0.977). Also documents a verifier fix (Unicode spacing in answers) this run exposed | `run_gym_eval("vllm", split="example", variant="property_shock")` |
| [`lora_smoke/`](lora_smoke/README.md) | LoRA SFT of the 30B model fits one A100 80 GB: peak 68.7 GiB, ~6.3 s per example | `python -m scripts.finetune_lora --smoke` |
| [`lora_full/`](lora_full/README.md) | Full LoRA SFT run: 401 rows from Nemotron's own passing rollouts (+ hint distillation), 102 steps, 67 min, peak 73.5 GiB | `python -m scripts.build_sft_dataset --max-per-task 40`, `python -m scripts.finetune_lora ... --max-len 20480` |
| [`lora_eval/`](lora_eval/README.md) | Held-out story: LoRA lifts the no-reasoning mode from **0.751** to **0.939** (base with reasoning: 0.964) at ~40% fewer output tokens than reasoning; no-tool answers 14/60 → 0/60. No gain with reasoning on (0.959); CRE drill-down did not transfer | `python -m scripts.eval_adapter [--thinking]` |
| [`story_variants/`](story_variants/summary.md) | Five extra July → August stories with different correct answers; 14/14 design claims hold in the engine's attribution; mock planner scores 0.988 on every variant through Gym, same as the base story (`mock_baseline.json`) | `python -m scripts.build_story_variants`, then `run_gym_eval("mock", split="example", variant=...)` |

Every figure in these files comes from the deterministic CECL engine or the Gym verifier, never from a model.
