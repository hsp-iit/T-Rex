#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TAPTIC_ROOT:?Set TAPTIC_ROOT to the verified T-Rex-style LeRobot export}"
: "${TAPTIC_PROCESSOR:?Set TAPTIC_PROCESSOR to local Qwen3-VL processor/config files}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
# Baseline adapter + multi-split, then the risk-conditioned adaptive T-Rex (OURS) suites.
"${TAPTIC_PYTHON:-python3}" -m pytest tests/taptic/test_adapter.py tests/taptic/test_multisplit.py \
    tests/taptic/test_risk.py tests/taptic/test_ours_model.py tests/taptic/test_ours_data.py -q --tb=short "$@"
