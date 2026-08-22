#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

exec "$PYTHON" scripts/e006_depth_range.py \
  --arms combined_random8_24_final combined_random8_24_aux \
  --seeds 23 47 \
  --output runs/E006 \
  --target-tokens 25000000 \
  --schedule-tokens 100000000 \
  --lr 0.002 \
  --checkpoint-every 192
