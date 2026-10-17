#!/usr/bin/env bash
# Serve NVIDIA Nemotron 3.5 Lightning (30B total / ~3B active, MoE) with vLLM on an NVIDIA Brev GPU
# instance (1x H100 or similar). Exposes an OpenAI-compatible API with tool calling on :8000.
# Red Hat AI Inference Server ships the same vLLM engine and accepts the same flags.
# Flags follow the vLLM day-0 recipe for this model (reasoning parser nemotron_v3, tool parser qwen3_coder).
#
#   bash deploy/brev_serve_nemotron.sh
#   # from the app machine:
#   export RISKLENS_LLM_PROVIDER=vllm RISKLENS_LLM_BASE_URL=http://<brev-host>:8000/v1
#   python -m scripts.check_llm --provider vllm
set -euo pipefail
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"   # or ...-NVFP4 (needs --trust-remote-code)
SERVED="${SERVED:-nemotron-3.5-lightning}"
PORT="${PORT:-8000}"
pip install -q "vllm>=0.27"
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "${MAX_LEN:-65536}" \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
