#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mkdir -p runs/E002

finish() {
  status=$?
  printf '%s\n' "$status" > runs/E002/remote_exit_code.txt
  date -u +'%Y-%m-%dT%H:%M:%SZ' > runs/E002/remote_finished_at.txt
  exit "$status"
}
trap finish EXIT

date -u +'%Y-%m-%dT%H:%M:%SZ' > runs/E002/remote_started_at.txt
.venv/bin/python scripts/e002_screen.py \
  --device cuda:0 \
  --tokens 5000000 \
  --batch 64 \
  --seed 17 \
  --arms \
    relative_fixed16 \
    combined_fixed16 \
    combined_random8_16 \
    combined_random8_16_aux \
  2>&1 | tee -a runs/E002/remote_training.log
