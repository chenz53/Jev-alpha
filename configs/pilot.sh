#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
bash scripts/build_pilot.sh
JEV_TRAIN="$PWD/data/pilot/train.jsonl" bash configs/train.sh \
  --num_train_epochs 2 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 --output_dir runs/pilot --logging_steps 4
