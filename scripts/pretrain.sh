#!/bin/bash
# Stage 1 / pretrain — VLM-action alignment on a large, tactile-free corpus + FLARE.
#
# Submit with:   sbatch scripts/pretrain.sh      (or run it inside an salloc shell)
# The #SBATCH lines target one A100 node of the Franklin cluster. gpua(-longrun)
# has five 4-GPU nodes (anode01-05) and one 8-GPU node (afnode01); for the 8-GPU
# node use --gres=gpu:8 --cpus-per-task=96 --mem=800G.
#SBATCH --job-name=trex_pretrain
#SBATCH --partition=gpua-longrun
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/iit.local/lgori/trex-mod/trex_runs/slurm-%x-%j.out
set -e
# Settings marked ${VAR:-default} below can be overridden from the environment,
# e.g. a short smoke test:  N_EPOCHS=1 SAVE_STEPS=40 VAL_FREQ=30 sbatch scripts/pretrain.sh

PROJECT_ROOT=${PROJECT_ROOT:-/home/iit.local/lgori/trex-mod}
cd ${PROJECT_ROOT}/scripts

# Env: pixi environment defined in ${PROJECT_ROOT}/pixi.toml (torch 2.6 cu124 +
# transformers 4.57.3 + lerobot 0.4.4). Build it once with `pixi install`.
PIXI=$(command -v pixi || echo "${HOME}/.pixi/bin/pixi")
eval "$(${PIXI} shell-hook --manifest-path ${PROJECT_ROOT}/pixi.toml)"
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Slurm sets CUDA_VISIBLE_DEVICES to the allocated GPUs.
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}

ORIGIN_MODEL_PATH=${ORIGIN_MODEL_PATH:-"/home/iit.local/lgori/ckpts/Qwen3-VL-2B-Instruct"}
# egomix retargets (LeRobot v3.0, one dataset per source, 30 fps): single Panda
# arm + right XHand. `action` is [ee xyz, rotation vector, 12 XHand joints]; the
# loader builds 21-D = [arm 9D delta-base eef | hand 12D] on the fly. Each source
# has one camera, used as head; it is also passed as the single fast image.
LEROBOT_ROOT=${LEROBOT_ROOT:-"/home/iit.local/lgori/egomix_100gb/retarget"}   # one or more dirs, space-separated
OUTPUT_DIR=${OUTPUT_DIR:-"/home/iit.local/lgori/trex-mod/trex_runs"}
RESUME_CHECKPOINT=""

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"trex_pretrain_egomix_panda_xhand"}
RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M)"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_PROCESSES=$((NUM_MACHINES * NUM_GPUS))

TRAIN_BSZ=${TRAIN_BSZ:-16}
LR=1e-4
# egomix retarget is ~225k frames (~2 h of video): one epoch is ~3.4k steps on
# 4 GPUs (~1.7k on 8), so it needs many more epochs than the 22k-hour T-Rex run.
N_EPOCHS=${N_EPOCHS:-10}
SAVE_STEPS=${SAVE_STEPS:-2000}          # mid-epoch resumable checkpoints
MAX_CKPTS=${MAX_CKPTS:-3}              # full-state checkpoints are large (weights + optimizer)
# Each sample decodes 9 head frames from HEVC on the CPU: keep workers high.
NUM_WORKERS=${NUM_WORKERS:-8}            # per GPU (+ half as many persistent val workers)

mkdir -p ${OUTPUT_DIR}

accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --deepspeed_multinode_launcher standard \
    pretrain.py \
    --model_path ${ORIGIN_MODEL_PATH} \
    --data_format lerobot \
    --lerobot_root ${LEROBOT_ROOT} \
    --lerobot_fast_keys observation.images.wrist \
    --lerobot_base action \
    --lerobot_video_backend pyav \
    --image_fit crop \
    --n_epochs ${N_EPOCHS} \
    --save_freq 1 \
    --action_dim 21 \
    --hand_dim 12 \
    --action_chunk 16 \
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
    --use_robot_state 1 \
    --image_size 384 288 \
    --num_workers ${NUM_WORKERS} \
    --shuffle_frames 1 \
    --save_steps ${SAVE_STEPS} \
    --val_ratio 0.02 \
    --val_freq ${VAL_FREQ:-1000} \
    --max_val_batches ${MAX_VAL_BATCHES:-50} \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --save_full_state 1 \
    --max_ckpts ${MAX_CKPTS}

echo ">>> Pretraining finished."
