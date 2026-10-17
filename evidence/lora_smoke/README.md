# LoRA SFT smoke run: Nemotron 3.5 Lightning on one A100 80 GB

Feasibility check before the real fine-tune: can the 30B hybrid Mamba/MoE model be LoRA-tuned on a
single GPU, with our ~13k-token agent trajectories? Yes.

| | |
|---|---|
| Base | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Hardware | 1 × A100 80 GB (Brev / Hyperstack), 2026-10-17 |
| Stack | transformers 5 + PEFT + Hugging Face Hub Mamba kernels, torch built for CUDA 12.8 (`requirements-train.txt`) |
| Command | `python -m scripts.finetune_lora --smoke` (10 steps, batch 1) |
| Data | 14 mock-planner trajectories (`data/outputs/sft/mock_sft.jsonl`); 1 of 15 skipped as > 16,384 tokens |
| LoRA | rank 16, alpha 32; attention q/k/v/o, Mamba `in_proj`, shared experts. Routed experts frozen |

| Result | |
|---|---:|
| Peak GPU memory | 68.7 of 79.3 GiB |
| Time | 63 s for 10 steps (~6.3 s per ~13k-token example) |
| Mean training loss | 1.20 |
| Adapter saved | yes |

`run_summary.json` is the file the script wrote on the GPU box.

What it means for the full run: 500 rows × 2 epochs ≈ 1.75 h on the same GPU. Most of each row is the
system prompt and 30 tool schemas; only the assistant turns (~190 tokens per row) are trained, and the
script computes logits only from the first trained token on, which is what keeps memory under 80 GB.

Not shown by this run: that the adapter improves anything. The smoke data is 14 scripted examples;
the improvement claim needs the full run on Nemotron's own passing rollouts and a Gym re-score.
