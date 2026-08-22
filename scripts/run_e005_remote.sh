#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_DIR="runs/E005_v2"
mkdir -p "$RUN_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/remote_started_at.txt"

set +e
"$PYTHON" scripts/e005_order_interaction.py \
  --arms relative_fixed16_mlp_first combined_random8_16_aux_mlp_first \
  --seeds 23 47 \
  --output "$RUN_DIR" \
  --target-tokens 25000000 --schedule-tokens 100000000 \
  --lr 0.002 --checkpoint-every 192 \
  > "$RUN_DIR/remote_training.log" 2>&1
status=$?
set -e

printf '%s\n' "$status" > "$RUN_DIR/remote_exit_code.txt
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/remote_finished_at.txt"
exit "$status"
