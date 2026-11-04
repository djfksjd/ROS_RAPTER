#!/bin/bash
# Evidence 199: fixed stage 5a, positive-only jump bonuses; never overwrite a run.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/../../.."
exec .venv-sim/bin/python docs/evidence/199-l80-jump/launch_j1.py "$@"
