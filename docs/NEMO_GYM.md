# RiskLens in NeMo Gym

*How the CECL investigation environment is built, how to run it locally, and what the reward means.*

## Why this is a good Gym environment

We generate the bank ourselves from a fixed seed, so every question has a known right answer (`data/ground_truth/ground_truth.json`). The actions are real, deterministic risk tools, so a model can only score by calling them well. That makes the reward **verifiable**: no LLM judge, no human labels.

## Components

| NeMo Gym concept | RiskLens file | Notes |
|---|---|---|
| Dataset | `resources_servers/risklens_cecl/data/{example,train,validation}.jsonl` | Built by `python -m scripts.build_gym_dataset` from `src/evaluation/tasks.py`. `example` = the 15 demo questions (committed); train/validation add paraphrases (regenerated) |
| Resources server | `resources_servers/risklens_cecl/app.py` | `SimpleResourcesServer`; every tool in `src/tools` registered as `POST /<tool_name>`; `/verify` returns `reward` + `reward_components` (`BaseMultiRewardVerifyResponse`) |
| Agent | Gym's built-in `simple_agent` (`max_steps: 10`) | Model ↔ tool loop; tool errors go back to the model as normal outputs |
| Model | Gym's `vllm_model` | Any OpenAI-compatible chat endpoint: local/Brev vLLM serving Nemotron 3.5 Lightning, the NVIDIA API catalog, or `scripts/serve_policy.py` (mock planner) |
| Config | `resources_servers/risklens_cecl/configs/risklens_cecl.yaml` | Found automatically when `gym` runs from the repo root |

Each dataset row:

```json
{"id": "q07_top_borrowers", "agent": "orchestrator",
 "responses_create_params": {"input": [{"role": "system", "content": "You are RiskLens..."},
                                       {"role": "user", "content": "Which borrowers contributed most...?"}],
                             "tools": [{"type": "function", "name": "identify_top_contributors", ...}, ...],
                             "parallel_tool_calls": false},
 "verifier_metadata": {"task_id": "...", "checks": [{"type": "mentions", "any_of": ["\\bCP_040408\\b"]}],
                       "required_tools": [["identify_top_contributors"]], "call_budget": 4}}
```

The expected facts are written into `verifier_metadata` when the dataset is built. The server never opens `data/ground_truth`, the same rule the tools follow.

## Reward

`src/evaluation/verifier.py`, weights in `config/gym.yaml`:

| Component | Weight | Meaning |
|---|---:|---|
| correctness | 0.55 | Share of the task's checks passed: named facts (primary driver, top portfolio, macro variable, largest borrower, model/weight change, reconciliation status) and key figures (within 1.5%, or 6% where sequential and Shapley attributions legitimately differ) |
| grounding | 0.20 | Share of numbers in the answer that appear in this episode's tool outputs (same matcher as the agent's grounding check: rounding and unit changes allowed, new arithmetic not) |
| tool_selection | 0.10 | Required tool groups called (e.g. a borrower question must call `identify_top_contributors`) |
| call_validity | 0.10 | Share of tool calls that executed without error |
| efficiency | 0.05 | 1 within the task's call budget, budget ÷ calls beyond it |

Two gates: an answer with **no successful tool call** is multiplied by 0.25, because every episode replays the same seeded story and a policy could otherwise learn to recite it; an **empty answer** scores 0.

The same `score()` runs in three places, so numbers are comparable: Gym's `/verify`, the in-process evaluator (`src/evaluation/local_eval.py`) and the agent notebook (`ask()` shows the reward per demo question). The notebook checks parity between Gym and in-process runs.

## Run it locally

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate     # nemo-gym needs Python >= 3.13.14
uv pip install -r requirements-gym.txt
python -m scripts.generate_demo_data && python -m scripts.precompute_analytics
python -m scripts.build_gym_dataset
```

**Offline (no GPU, no key)**, with the mock planner behind an OpenAI-compatible endpoint:

```bash
python -m scripts.serve_policy --provider mock --port 8011 &
gym env start --resources-server risklens_cecl --model-type vllm_model \
    --model mock-planner --model-url http://127.0.0.1:8011/v1 --model-api-key none
# second terminal
gym eval run --no-serve --agent risklens_cecl_simple_agent \
    --input resources_servers/risklens_cecl/data/train.jsonl --output data/outputs/gym/mock_train_rollouts.jsonl
```

**Nemotron 3.5 Lightning** on vLLM (`bash deploy/brev_serve_nemotron.sh` on the GPU box):

```bash
gym env start --resources-server risklens_cecl --model-type vllm_model \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --model-url http://<host>:8000/v1 --model-api-key none
gym eval run --no-serve --agent risklens_cecl_simple_agent --num-repeats 4 \
    --input resources_servers/risklens_cecl/data/example.jsonl --output data/outputs/gym/vllm_example_rollouts.jsonl
```

From Python or a notebook, `src.evaluation.gym_runner.run_gym_eval("mock" | "vllm" | "nvidia", split=...)` does start → collect → stop in one call and `load_rollouts()` gives a DataFrame. The first `gym env start` builds a venv per server (a few minutes); later starts take seconds. Gym writes those venvs and Hydra logs under the repo root (`environment_servers/`, `responses_api_*/`, `outputs/`); they are git-ignored.

## Results so far

| Policy | Split | Rollouts | Mean reward | Notes |
|---|---|---:|---:|---|
| mock planner (keyword script) | train | 21 | 0.965 | Floor. Fails a paraphrase it has no keyword for ("Which segment drove…") and never says "increased" for PD |
| mock planner | example, each of the 5 story variants | 15 each | 0.988 each | Same as the base story; only miss is the PD template (`evidence/story_variants/mock_baseline.json`) |
| Nemotron 3.5 Lightning (BF16, vLLM on A100) | example | 15 | 0.977 | Lost points only on q14 (no CRE borrower named), two recovered tool errors and a few ungrounded figures. [Evidence](../evidence/nemotron_baseline/README.md) |

## Story variants

The base story always has the same answer (macro, CRE, unemployment), so a policy could score by reciting it. Story variants keep the bank, dates and questions and change what happened in August, so the correct answers change too:

| Variant | August story | Correct primary driver |
|---|---|---|
| `credit_led` | macro and weights flat; 60 CRE borrowers deteriorate sharply | credit deterioration |
| `growth_led` | macro and weights flat; ~5× the usual new business | portfolio growth |
| `property_shock` | CRE prices and rates worsen, unemployment flat | macroeconomic, via CRE prices |
| `weights_shift` | paths flat; weight moved from baseline to severe | model / methodology & judgment |
| `consumer_stress` | unemployment shock, property flat | macroeconomic, via unemployment |

Each is `config/story_variants/<name>.yaml` merged over `demo_story.yaml` and states its intended outcome under `design:`. `python -m scripts.build_story_variants` generates each variant into `data/variants/<name>/` (data, official runs, caches, ground truth), builds its Gym rows in `resources_servers/risklens_cecl/data/variants/<name>/` (ids prefixed `<name>__`), checks every `design` claim against the engine's attribution and writes [`evidence/story_variants/summary.md`](../evidence/story_variants/summary.md). The base story is untouched.

The questions are the same; the checks follow the variant's ground truth. Where the macro paths are flat, "which macro variable mattered most?" expects "none" (every variable's impact is exactly $0), and "did the weights change?" expects "no".

```bash
python -m scripts.build_story_variants                      # all five, ~25 s each
python -c 'from src.evaluation.gym_runner import run_gym_eval
run_gym_eval("vllm", split="train", variant="credit_led", num_repeats=4)'
```

`run_gym_eval(..., variant=...)` points the Gym servers at the variant through `RISKLENS_*` env vars and reads that variant's rows.

## From rollouts to training

`python -m scripts.build_sft_dataset` assembles the SFT set: every `vllm_*_train` / `*_hint` rollout outside the held-out story (`HELD_OUT` in `src/evaluation/story_variants.py`, currently `property_shock`), re-scored with the current verifier, kept at reward ≥ 0.95.

**Hint distillation** (`src/evaluation/hint_distill.py`, `python -m scripts.build_hint_rows`): base Nemotron never drills down to borrowers on "what happened in CRE?" (0 of 16 rollouts named a deteriorated CRE borrower), so its own rollouts hold nothing to learn from. `hint.jsonl` re-asks those questions with one extra sentence in the system prompt; passing rollouts are exported with the original prompt (`verifier_metadata.distill_system`), so the tuned model learns the drill-down unprompted.

`src/evaluation/trajectories.py` keeps rollouts with reward ≥ 0.8 and converts them to OpenAI chat format with tool calls (`data/outputs/gym/sft_trajectories.jsonl`), ready for LoRA SFT. For RL, NeMo RL's GRPO recipes take a Gym resources server as the reward source; full RL on Lightning needs a multi-node GPU budget, so that is post-hackathon.

## Next

- Specialist datasets: `python -m scripts.build_gym_dataset --agent credit_agent` (15 tools instead of 30) to score each Phase 4 specialist.
- Recovery tasks: rows that start with a malformed call already in the context, scored on whether the model reads the hint and retries.
