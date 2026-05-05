#!/bin/bash
# Consolidate Unity (image, mask) pairs into data/raw/
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.dataset.prepare "$@"
