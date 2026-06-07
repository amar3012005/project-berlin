#!/usr/bin/env bash
# One-shot environment bring-up on a fresh ThunderCompute GPU instance.
# Usage on the instance:
#   git clone <repo> && cd berlin && bash setup_env.sh && source .venv/bin/activate
# Then: huggingface-cli login   (Pharia-1 is gated)
#       ./start.sh train configs/pharia_7b.yaml   (single GPU)
#   or  ./start.sh train_multi configs/pharia_7b.yaml   (8 GPU FSDP)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.11}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python "$PY" .venv
source .venv/bin/activate

# CUDA 12.x PyTorch wheels (ThunderCompute images ship CUDA 12.x drivers)
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
uv pip install transformers accelerate datasets pyyaml huggingface_hub sentencepiece

# FlashAttention — optional, big speedup on Ampere/Hopper. Skip if it fails (slow build).
echo "[setup] attempting flash-attn (optional; FA3 on H100)..."
uv pip install flash-attn --no-build-isolation 2>/dev/null \
  && echo "[setup] flash-attn OK — set attn_implementation: flash_attention_2 in config" \
  || echo "[setup] flash-attn skipped (use sdpa) — non-fatal"

echo "[setup] GPU check:"
python -c "import torch; print(' cuda', torch.cuda.is_available(), \
'| gpus', torch.cuda.device_count(), \
'|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"

echo "[setup] DONE. Next: source .venv/bin/activate && huggingface-cli login"
