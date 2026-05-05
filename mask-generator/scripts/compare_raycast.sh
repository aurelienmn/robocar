#!/bin/bash
# Compare our pipeline's raycast vs the reference (GT mask) raycast on the test set.
# Produces stats + a verdict + plots in models/compare_<timestamp>/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.compare_raycast "$@"
