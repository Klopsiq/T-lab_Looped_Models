#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
WINDOWS="reports/E008_locked_test_windows.npz"
OUT="runs/E008"
mkdir -p "$OUT"

evaluate() {
  local checkpoint="$1"
  local output="$2"
  "$PYTHON" scripts/eval_bpe_checkpoint.py \
    --checkpoint "$checkpoint" --windows "$WINDOWS" --output "$OUT/$output" \
    --depths 12 16 24 32 --batch 8 --device cuda:0
}

evaluate runs/E003/combined_random8_16_aux_s23/last.pt old_aux_s23.json
evaluate runs/E003/combined_random8_16_aux_s47/last.pt old_aux_s47.json
evaluate runs/E006/combined_random8_24_final_s23/last.pt new_final_s23.json
evaluate runs/E006/combined_random8_24_final_s47/last.pt new_final_s47.json
evaluate runs/E007/combined_random8_24_final_s71/last.pt new_final_s71.json
evaluate runs/E007/combined_random8_24_final_s89/last.pt new_final_s89.json
