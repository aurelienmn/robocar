#!/bin/bash
# Export the trained model to fp32 + int8 ONNX (for Jetson Nano deployment).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.model.quantize "$@"
