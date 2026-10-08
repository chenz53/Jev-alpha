#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m jev_alpha.datasets.boolq
python -m jev_alpha.datasets.hard_cases
python -m jev_alpha.datasets.prepare data/boolq/train.jsonl data/boolq/validation.jsonl data/hard_cases.jsonl --output data/prepared
