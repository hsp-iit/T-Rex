#!/bin/bash
# Stage 2 / midtrain on TAPTIC — T-Rex cascaded baseline (fixed split tau=0.4)
# with SDK fingertip force + embedded TAPTIC VQ-VAE (native ~76 Hz history).
#
# Pipeline:
#   1. VQ-VAE:  TAPTIC_ROOT=... OUTPUT_DIR=... bash tactile_vqvae/scripts/train_vqvae_taptic.sh
#   2. this script with VQVAE_CKPT=<run>/latest.pt
#
# Tactile expert input per sample: 5 VQ code tokens (16 native force samples,
# ~210 ms, per finger; with a residual-VQ checkpoint each token is the sum of
# its stage embeddings) + 5 current-force tokens. Legacy 10 Hz episodes (3 tasks)
# are kept for the action expert/FLARE but masked out of the tactile loss.
set -euo pipefail

: "${TAPTIC_ROOT:?set TAPTIC_ROOT to the TAPTIC LeRobot export (same revision as TAPTIC_STATS)}"
: "${MODEL_PATH:?set MODEL_PATH to Qwen3-VL-2B-Instruct}"
: "${VQVAE_CKPT:?set VQVAE_CKPT to the TAPTIC VQ-VAE latest.pt}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAPTIC_STATS=${TAPTIC_STATS:-${PROJECT_ROOT}/docs/taptic_train_stats.json}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}      # optional pretrain checkpoint (no VQ-VAE inside)
FLOW_TIME_PRECISION=${FLOW_TIME_PRECISION:-fp32}   # legacy_bf16 = original T-Rex arithmetic
NUM_GPUS=${NUM_GPUS:-8}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_BSZ=${TRAIN_BSZ:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}         # >0: probe run, stops early, no checkpoint
MAX_CKPTS=${MAX_CKPTS:-3}
VAL_RATIO=${VAL_RATIO:-0.02}                  # >0 enables validation on the dataset val split
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-10}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-trex_midtrain_taptic_force_vq}
RUN_NAME=${RUN_NAME:-${EXPERIMENT_NAME}_$(date +%m%d_%H%M)}

cd "${PROJECT_ROOT}/scripts"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export WANDB_MODE=${WANDB_MODE:-offline}

RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    RESUME_ARGS=(--resume_checkpoint "$RESUME_CHECKPOINT" --resume_source pretrain)
fi

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes $((NUM_MACHINES * NUM_GPUS)) \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --deepspeed_multinode_launcher standard \
    midtrain.py \
    --model_path "${MODEL_PATH}" \
    --data_format taptic \
    --lerobot_root "${TAPTIC_ROOT}" \
    --taptic_stats "${TAPTIC_STATS}" \
    --taptic_tactile_input sdk_force \
    --n_epochs "${EPOCHS}" \
    --save_freq 2 \
    --action_dim 21 \
    --action_chunk 16 \
    --train_bsz_per_gpu "${TRAIN_BSZ}" \
    --learning_rate "${LR}" \
    --min_lr_ratio 0 \
    --warmup_rates 0.03 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --max_ckpts "${MAX_CKPTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "${RUN_NAME}" \
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 0 \
    --use_tactile_vqvae 1 \
    --vqvae_ckpt "${VQVAE_CKPT}" \
    --tactile_f6_stats_ckpt "${VQVAE_CKPT}" \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --tactile_delay_offsets 0 \
    --cascaded_total_steps 10 \
    --cascaded_split_mode fixed \
    --cascaded_split_step 6 \
    --flow_time_precision "${FLOW_TIME_PRECISION}" \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --num_workers "${NUM_WORKERS}" \
    --val_ratio "${VAL_RATIO}" \
    --val_freq 500 \
    --max_val_batches 50 \
    "${RESUME_ARGS[@]}"

echo ">>> TAPTIC mid-training finished."
