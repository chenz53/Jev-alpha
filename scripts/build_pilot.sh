#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m jev_alpha.datasets.pilot
