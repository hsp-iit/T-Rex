#!/bin/bash
# T-REX midtraining on primitive-annotated TAPTIC data — Panda + RobotEra XHand.
#
# Identical to scripts/midtrain.sh (same DeepSpeed ZeRO-2 bf16 config, LR,
# schedule, cascaded 10/6, tactile dropout 0.1, FLARE 4x8 stride 4, delays
# 0/4/8/12, tactile MLP 1536, resume from the official pretrain checkpoint)
# except for the robot interface:
#   --action_dim 21 --tacf6_dim 3 --n_tactile_fingers 5
#   --use_tactile_deform 0 (no deformation maps)  --use_robot_state 0
#   --data_format taptic --taptic_dirs ...        (primitive samples)
#   --vqvae_ckpt <XHand VQ-VAE>                   (retrained, [16,5,3])
#
# Prerequisites (see docs/xhand/README.md):
#   python -m taptic.convert ...           -> $TAPTIC_DIR
#   bash tactile_vqvae/scripts/train_vqvae_xhand.sh  -> $VQVAE_CKPT
#   Qwen3-VL-2B-Instruct + miniFranka/T-Rex_pretrain_mecka22k_epoch1 downloaded.
set -e

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd ${PROJECT_ROOT}/scripts
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export WANDB_MODE=${WANDB_MODE:-offline}

ORIGIN_MODEL_PATH=${ORIGIN_MODEL_PATH:?path to Qwen3-VL-2B-Instruct}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?path to T-Rex_pretrain_mecka22k_epoch1/checkpoint-0-610000}
TAPTIC_DIRS=${TAPTIC_DIRS:?space-separated taptic.convert task dirs}
TAPTIC_STATS=${TAPTIC_STATS:-}          # required when TAPTIC_DIRS has several tasks (python -m taptic.stats)
VQVAE_CKPT=${VQVAE_CKPT:?XHand VQ-VAE latest.pt}
OUTPUT_DIR=${OUTPUT_DIR:?output dir}
TACTILE_NORM_INIT=${TACTILE_NORM_INIT:-zeros}   # zeros = official; see COMPATIBILITY.md §B1

EXPERIMENT_NAME=${EXPERIMENT_NAME:-trex_midtrain_xhand}
RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M)"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))

TRAIN_BSZ=${TRAIN_BSZ:-16}
LR=${LR:-1e-4}
N_EPOCHS=${N_EPOCHS:-10}

STATS_ARG=()
if [ -n "$TAPTIC_STATS" ]; then STATS_ARG=(--taptic_stats "$TAPTIC_STATS"); fi

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard \
    midtrain.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_format taptic \
    --taptic_dirs ${TAPTIC_DIRS} "${STATS_ARG[@]}" \
    --n_epochs ${N_EPOCHS} \
    --save_freq 2 \
    --action_dim 21 \
    --action_chunk 16 \
    --tacf6_dim 3 \
    --n_tactile_fingers 5 \
    --train_bsz_per_gpu ${TRAIN_BSZ} \
    --learning_rate ${LR} \
    --min_lr_ratio 0 \
    --warmup_rates 0.03 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_name ${RUN_NAME} \
    --use_robot_state 0 \
    --use_tactile_vec 1 \
    --use_tactile_deform 0 \
    --use_tactile_vqvae 1 \
    --vqvae_ckpt ${VQVAE_CKPT} \
    --tactile_intermediate_size 1536 \
    --tactile_norm_init ${TACTILE_NORM_INIT} \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --tactile_delay_offsets 0 4 8 12 \
    --cascaded_total_steps 10 \
    --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
    --resume_checkpoint ${RESUME_CHECKPOINT} \
    --resume_source pretrain \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --num_workers 4 \
    --val_ratio 0.1 \
    --val_freq 200 \
    --max_val_batches 50

echo ">>> XHand mid-training finished."
