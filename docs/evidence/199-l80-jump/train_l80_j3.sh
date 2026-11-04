#!/bin/bash
# Positive completion rebalance only; supervisor starts this, no automatic extension.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/../../.."
exec .venv-sim/bin/python docs/evidence/199-l80-jump/launch_j3.py "$@"
