#!/bin/bash
# Visual end-to-end evaluation: image → predicted mask → raycast.
# Saves a 2x2 comparison grid for ~8 samples to models/eval_<timestamp>/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.evaluate "$@"
