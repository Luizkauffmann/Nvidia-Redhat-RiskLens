#!/usr/bin/env bash
# Serve Nemotron 3.5 Lightning with vLLM on a Brev GPU instance (OpenAI-compatible API on :8000).
#
#   export HF_TOKEN=hf_...            # accept the model license on Hugging Face first
#   bash deploy/brev_serve_nemotron.sh
#
# Then point RiskLens at it:
#   RISKLENS_LLM_PROVIDER=vllm RISKLENS_LLM_BASE_URL=http://<brev-host>:8000/v1 python -m scripts.check_llm --provider vllm
#
# Flags follow NVIDIA's model card / vLLM day-0 recipe (vLLM >= 0.27.1):
#   qwen3_coder tool parser + auto tool choice  -> structured tool_calls
#   nemotron_v3 reasoning parser                -> thinking kept out of the answer text
# Checkpoints: NVFP4 (~4-bit, default; Blackwell native, Hopper/Ampere via W4A16 kernels)
#              BF16 (~60 GB weights; one 80 GB A100/H100)
# Untested on the event's Brev instance types; check `nvidia-smi` and adjust MAX_LEN / GPU_UTIL.
set -euo pipefail
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
MAX_LEN="${MAX_LEN:-65536}"          # the model supports 1M tokens; 64k is plenty for RiskLens and saves KV cache
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1}"
: "${HF_TOKEN:?set HF_TOKEN (Hugging Face token with the Nemotron license accepted)}"

docker run --rm --gpus all --ipc=host -p "${PORT}:8000" \
  -e HF_TOKEN="${HF_TOKEN}" -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "${IMAGE}" \
  --model "${MODEL}" --served-model-name "${MODEL}" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "${MAX_LEN}" --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-prefix-caching --max-num-seqs 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
