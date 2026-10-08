#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${CHECKPOINT:?Set CHECKPOINT to the trained adapter directory}"
model="${MODEL:-Qwen/Qwen3.5-0.8B-Base}"
data_dir="${EVAL_DATA:-data/prepared}"
output="${EVAL_OUTPUT:-runs/evaluation-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir "$output"
python -m jev_alpha.predict --model "$model" --adapter "$CHECKPOINT" --evaluate \
  --input "$data_dir/calibration.jsonl" --output "$output/calibration.jsonl"
python -m jev_alpha.calibrate "$output/calibration.jsonl" "$CHECKPOINT"
python -m jev_alpha.predict --model "$model" --adapter "$CHECKPOINT" --evaluate \
  --input "$data_dir/test.jsonl" --output "$output/test.jsonl"
python -m jev_alpha.eval "$output/test.jsonl" "$CHECKPOINT/temperature.json" \
  "${REPORT:-reports/boolq-baseline.json}"
