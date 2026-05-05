#!/bin/bash
# Launch TensorBoard on the training logs. Open http://localhost:6006
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
tensorboard --logdir logs "$@"
