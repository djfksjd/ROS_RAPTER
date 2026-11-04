#!/bin/bash
# Evidence 199: remove two anti-flight costs only during requested takeoff/flight.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/../../.."
exec .venv-sim/bin/python docs/evidence/199-l80-jump/launch_j2.py "$@"
