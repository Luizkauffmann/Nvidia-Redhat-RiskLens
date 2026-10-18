# Full LoRA SFT run: Nemotron 3.5 Lightning on its own passing RiskLens rollouts

| | |
|---|---|
| Base | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Hardware | 1 × A100 80 GB (Brev / Hyperstack), 2026-10-17 |
| Command | `python -m scripts.finetune_lora --data data/outputs/sft/vllm_sft.jsonl --epochs 2 --max-len 20480` |
| LoRA | rank 16, alpha 32, attention q/k/v/o + Mamba `in_proj` + shared experts (`adapter_config.json`) |
| Result | 102 steps, 67 min, peak 73.5 of 79.3 GiB, mean train loss 0.286 (step 1: 0.674) — `run_summary.json` |

## Data (`sft_dataset_summary.json`)
- 540 Nemotron rollouts collected through NeMo Gym on the base story and four story variants
  (`credit_led`, `growth_led`, `weights_shift`, `consumer_stress`): 420 on the train split, 120 hint rollouts.
- Kept: re-scored with the current verifier, reward ≥ 0.95, at most 40 per question → **401 rows**.
- Hint distillation for "what happened in CRE?" (base Nemotron never named the deteriorated borrowers):
  40 rows answered with a one-sentence hint, exported with the original prompt.
- `property_shock` is held out of all training data.
- Trained in thinking-off format (`<think></think>`); only assistant turns are trained (~540 tokens per row).

Evaluation: [`../lora_eval/`](../lora_eval/).
