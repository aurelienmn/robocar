#!/bin/bash
# Predict mask + raycast for a single image. Usage:
#   ./scripts/predict.sh --image data/raw/images/pair_000172.png
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.predict "$@"
