#!/bin/bash
# Stage 2 / midtrain — tactile-reactive training (cascaded flow), resumes from pretrain.
#
# Default data: LeRobot v3.0 recordings of a single Panda arm + right XHand
# (TAPTIC layout), read directly by qwen_vla/lerobot_dataset.py:
#   action            [18]  ee.x,y,z + rotation vector + 12 XHand joint targets
#   observation.state [40]  7 Panda joints | 12 XHand joints | 15 fingertip
#                           forces (5 x xyz) | 6 EE pose
#   cameras           observation.images.left (head) + observation.images.wrist
# The loader builds the 21-D delta-base chunk [arm 9D | hand 12D] from the EE
# pose in observation.state and the action targets, and the F6 tokens as
# [Fx, Fy, Fz, 0, 0, 0] per fingertip (constant channels are left unnormalized).
#
# For the original T-Rex merged layout use DATA_FORMAT=json (see git history for
# prepare_midtrain_merged.py usage).
set -e

PROJECT_ROOT=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/T-Rex
cd ${PROJECT_ROOT}/scripts

source /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/bin/activate /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot
export PATH=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/code/miniconda3/envs/dex_mot/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ORIGIN_MODEL_PATH="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp"
RESUME_CHECKPOINT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/exp/trex_pretrain_flare/checkpoint-0-610000"

EXPERIMENT_NAME="trex_midtrain_flare"
RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M)"

# ── data source: "lerobot" (default) or "json" ──
DATA_FORMAT="lerobot"
LEROBOT_ROOT="/path/to/lerobot/split_cups"      # one or more dataset dirs, space-separated
MERGED_DATA_ROOT="/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged_inlab"

# ── embodiment: single Panda arm + right XHand ──
ACTION_DIM=21
HAND_DIM=12

# ── time constants, in dataset frames (TAPTIC records at 10 Hz) ──
# T-Rex used 30 Hz: chunk 16, flare stride 4, delays 0 4 8 12. At 10 Hz the same
# chunk covers 1.6 s; delays should match how many frames the slow pass lags at
# deployment. Keep ACTION_CHUNK equal to the pretraining run.
ACTION_CHUNK=16
FLARE_STRIDE=1
TACTILE_DELAYS="0 1 2 3"

if [ "${DATA_FORMAT}" = "lerobot" ]; then
    DATA_ARG="--data_format lerobot --lerobot_root ${LEROBOT_ROOT} \
        --lerobot_head_key observation.images.left \
        --lerobot_fast_keys observation.images.wrist \
        --lerobot_base state --lerobot_state_ee_index 34 --lerobot_state_hand_index 7 \
        --lerobot_tactile_key observation.state --lerobot_tactile_index 19 \
        --lerobot_tactile_fingers 5 --lerobot_tactile_axes 3 \
        --image_fit crop"
else
    DATA_ARG="--data_format json --data_root ${MERGED_DATA_ROOT}"
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * 8))

TRAIN_BSZ=16
LR=1e-4

# Tactile inputs: F6 tokens only. The released VQ-VAE is trained on 10 Sharpa
# fingertips x 6-axis wrenches and there are no deformation maps, so
# --use_tactile_vqvae / --use_tactile_deform stay off until an XHand VQ-VAE
# (tactile_vqvae/) and a deformation encoder exist.
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
    ${DATA_ARG} \
    --n_epochs 10 \
    --save_freq 2 \
    --action_dim ${ACTION_DIM} \
    --hand_dim ${HAND_DIM} \
    --action_chunk ${ACTION_CHUNK} \
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
    --use_tactile_vqvae 0 \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --tactile_delay_offsets ${TACTILE_DELAYS} \
    --cascaded_total_steps 10 \
    --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
    --resume_checkpoint ${RESUME_CHECKPOINT} \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride ${FLARE_STRIDE} \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --num_workers 4 \
    --val_ratio 0.02 \
    --val_freq 500 \
    --max_val_batches 50

echo ">>> Mid-training finished."
