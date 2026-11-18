#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TAPTIC_ROOT:?Set TAPTIC_ROOT to the verified T-Rex-style LeRobot export}"
: "${TAPTIC_PROCESSOR:?Set TAPTIC_PROCESSOR to local Qwen3-VL processor/config files}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
"${TAPTIC_PYTHON:-python3}" -m pytest tests/taptic/test_adapter.py -q --tb=short "$@"
