#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${1:-calibration}"

if [[ "$PHASE" == "calibration" ]]; then
  "$PYTHON" scripts/e003_bpe_train.py \
    --arms relative_fixed16 --seeds 101 \
    --output runs/E003_calibration/lr1e-3 \
    --target-tokens 5000000 --schedule-tokens 100000000 \
    --lr 0.001 --checkpoint-every 38
  "$PYTHON" scripts/e003_bpe_train.py \
    --arms relative_fixed16 --seeds 101 \
    --output runs/E003_calibration/lr2e-3 \
    --target-tokens 5000000 --schedule-tokens 100000000 \
    --lr 0.002 --checkpoint-every 38
elif [[ "$PHASE" == "confirm" ]]; then
  : "${E003_LR:?Set E003_LR to the LR selected by registered calibration}"
  "$PYTHON" scripts/e003_bpe_train.py \
    --arms relative_fixed16 combined_random8_16_aux --seeds 23 47 \
    --output runs/E003 --target-tokens 25000000 --schedule-tokens 100000000 \
    --lr "$E003_LR" --checkpoint-every 192
else
  echo "Usage: $0 {calibration|confirm}" >&2
  exit 2
fi
