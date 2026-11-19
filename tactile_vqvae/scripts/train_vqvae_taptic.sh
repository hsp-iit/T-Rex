#!/usr/bin/env bash
# Tactile VQ-VAE on TAPTIC native (~76 Hz) SDK fingertip force.
#   window = 16 native samples (~210 ms), per-finger codes (5 per window),
#   train/val = dataset suggested_split, native episodes only (legacy 10 Hz excluded).
#
#   TAPTIC_ROOT=/path/to/taptic_trex_style_v2 OUTPUT_DIR=/path/to/out \
#     bash tactile_vqvae/scripts/train_vqvae_taptic.sh
# Smoke: SMOKE=1 (5 steps, few episodes, 1 process).
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${TAPTIC_ROOT:?set TAPTIC_ROOT to the TAPTIC LeRobot export}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
PYTHON=${PYTHON:-python}
N_GPUS=${N_GPUS:-1}
WINDOW=${WINDOW:-16}
STRIDE=${STRIDE:-2}
CODEBOOK=${CODEBOOK:-64}
EMBED=${EMBED:-128}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-256}
LR=${LR:-3e-4}
WEIGHT_TAU=${WEIGHT_TAU:-0}          # 0 = auto (p75 of train raw magnitudes)
RUN_NAME=${RUN_NAME:-vqvae_taptic_force_w${WINDOW}_k${CODEBOOK}_finger}
USE_WANDB=${USE_WANDB:-0}
SMOKE=${SMOKE:-0}
EXTRA=()
if [[ "$SMOKE" == 1 ]]; then EXTRA=(--smoke_test 1 --max_episodes 8 --num_workers 0); N_GPUS=1; fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$PYTHON" -m accelerate.commands.launch --num_processes "$N_GPUS" --num_machines 1 \
    --mixed_precision bf16 -m tactile_vqvae.train \
    --data_format taptic --lerobot_root "$TAPTIC_ROOT" \
    --output_dir "$OUTPUT_DIR" --run_name "$RUN_NAME" \
    --granularity finger --window "$WINDOW" --stride "$STRIDE" \
    --codebook_size "$CODEBOOK" --embed_dim "$EMBED" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" \
    --weight_tau "$WEIGHT_TAU" --val_every 2000 --use_wandb "$USE_WANDB" "${EXTRA[@]}"

echo ">>> checkpoint: $OUTPUT_DIR/$RUN_NAME/latest.pt"
