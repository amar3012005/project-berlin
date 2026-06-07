#!/usr/bin/env bash
# Project Berlin — DLM surgery pipeline entrypoint.
# Toy proofs run on laptop CPU/MPS; train scales to GPU with ZERO code change.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$PWD/src/diffusion:$PWD/src/data:${PYTHONPATH:-}"

cmd="${1:-help}"
cfg="${2:-configs/toy_gpt2.yaml}"
case "$cmd" in
  # --- milestone selftests (proofs) ---
  m0|load)      python src/diffusion/load_model.py ;;
  m1|rewire)    python src/diffusion/attention_surgery.py ;;
  m2|mask)      python src/diffusion/diffusion.py ;;
  m3|sample)    python src/diffusion/generate.py ;;
  m4|overfit)   python src/diffusion/train_overfit.py ;;
  selftest)     for s in load_model attention_surgery diffusion generate train_overfit; do
                  echo "=== $s ==="; python "src/diffusion/$s.py"; done ;;

  # --- real training ---
  train)        python src/diffusion/train.py "$cfg" ;;                 # single device
  train_multi)  accelerate launch --config_file configs/accelerate_ddp.yaml \
                  src/diffusion/train.py "$cfg" ;;                       # multi-GPU DDP
  monitor)      python src/monitor/server.py --metrics "${2:-checkpoints/eurollm_cluster/metrics.jsonl}" --port "${3:-8888}" ;;

  *) cat <<EOF
usage: ./start.sh <cmd> [config.yaml]
  m0|m1|m2|m3|m4   run individual milestone proof
  selftest         run all proofs
  train  [cfg]     train on one device   (default cfg: configs/toy_gpt2.yaml)
  train_multi[cfg] train on 8-GPU FSDP   (use configs/pharia_7b.yaml)
EOF
     exit 1 ;;
esac
