#!/bin/bash
# Train the U-Net on the prepared dataset. Saves best.pt + tensorboard logs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.model.train "$@"
