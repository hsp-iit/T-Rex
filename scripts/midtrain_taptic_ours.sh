#!/bin/bash
# TAPTIC midtrain — controlled variants of the risk-conditioned adaptive T-Rex.
#
#   VARIANT=vanilla     fixed tau=0.4, no risk tokens, tactile at S (== scripts/midtrain_taptic.sh)
#   VARIANT=multisplit  uniform tau in {0.2,0.4,0.6}, no risk tokens, delayed tactile F = S + {0,50,100,150} ms
#                       (also the checkpoint for "external risk gate" inference: config 3)
#   VARIANT=ours        multisplit + learned risk tokens r_S (Action) / r_F (Tactile) + same delays
#                       + future-risk head (Tactile), risk dropout, high-risk oversampling
#                       (multisplit and ours also train a DETACHED future-risk probe on the Action expert)
#
# Everything else — data revision/split, VQ-VAE, initialization, optimizer, epochs,
# effective batch, seed, FLARE — is identical across variants. risk_stats are
# loaded for multisplit too (logging only: r_S/r_F in wandb, no effect on the model).
# TACTILE_DELAY_MS overrides the delay set for multisplit/ours (e.g. "0" for the
# no-delay ablation); vanilla always uses 0.
set -euo pipefail

: "${VARIANT:?set VARIANT=vanilla|multisplit|ours}"
: "${TAPTIC_ROOT:?set TAPTIC_ROOT to the TAPTIC LeRobot export (same revision as TAPTIC_STATS)}"
: "${MODEL_PATH:?set MODEL_PATH to Qwen3-VL-2B-Instruct}"
: "${VQVAE_CKPT:?set VQVAE_CKPT to the TAPTIC VQ-VAE latest.pt}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAPTIC_STATS=${TAPTIC_STATS:-${PROJECT_ROOT}/docs/taptic_train_stats.json}
RISK_STATS=${RISK_STATS:-${PROJECT_ROOT}/docs/taptic_ours/risk_stats.json}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}      # optional pretrain checkpoint (same for every variant)
NUM_GPUS=${NUM_GPUS:-8}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_BSZ=${TRAIN_BSZ:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}         # >0: probe run, stops early, no checkpoint
MAX_CKPTS=${MAX_CKPTS:-3}
VAL_RATIO=${VAL_RATIO:-0.02}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-10}
SEED=${SEED:-42}
TACTILE_DELAY_MS=${TACTILE_DELAY_MS:-"0 50 100 150"}
# OURS training components (docs/taptic_ours/REPORT.md addendum): future-risk head on the Tactile expert,
# detached future-risk probe on the Action expert (also in multisplit: it never changes the model),
# risk dropout and compute-neutral high-risk oversampling.
RISK_FUTURE_HEAD_WEIGHT=${RISK_FUTURE_HEAD_WEIGHT:-0.1}
RISK_FUTURE_PROBE=${RISK_FUTURE_PROBE:-1}
RISK_DROPOUT=${RISK_DROPOUT:-0.1}
RISK_OVERSAMPLE=${RISK_OVERSAMPLE:-2}
# trex-touch components (any VARIANT; defaults = baseline; docs/TOUCH_MODULES.md): long tactile history in
# the per-finger vector tokens, self-handover Tactile-expert pairs, contact-weighted hand dims of the tactile
# loss, event-weighted tactile loss, dataset variant selection by source repo, touch encoder from another dataset.
TACTILE_HISTORY_SPAN_MS=${TACTILE_HISTORY_SPAN_MS:-0}
TACTILE_HISTORY_POINTS=${TACTILE_HISTORY_POINTS:-8}
TACTILE_SELF_HANDOVER=${TACTILE_SELF_HANDOVER:-0}
TACTILE_FORCE_CONTACT_WEIGHT=${TACTILE_FORCE_CONTACT_WEIGHT:-1}
TACTILE_EVENT_WEIGHT=${TACTILE_EVENT_WEIGHT:-1}
TACTILE_EVENT_TYPES=${TACTILE_EVENT_TYPES:-"grasp_onset resistance"}
TACTILE_EVENT_WINDOW_MS=${TACTILE_EVENT_WINDOW_MS:-"200 1000"}
TACTILE_EVENT_JUMP_QUANTILE=${TACTILE_EVENT_JUMP_QUANTILE:-0.97}
TACTILE_EVENT_NORMALIZE=${TACTILE_EVENT_NORMALIZE:-1}
TAPTIC_EXCLUDE_SOURCES=${TAPTIC_EXCLUDE_SOURCES:-}
TAPTIC_VQVAE_SOURCE_IDENTITY=${TAPTIC_VQVAE_SOURCE_IDENTITY:-}
# Asynchronous touch for the fixed-split (vanilla) variant (docs/TOUCH_MODULES.md): each sample's touch is newer
# than its image by one of these delays (ms), as when the tactile expert re-runs on the cached vision pass
# between images (T-Rex fast ticks). "0" = synchronous (default, the baseline's batch).
TACTILE_ASYNC_DELAYS_MS=${TACTILE_ASYNC_DELAYS_MS:-0}
TOUCH_ARGS=(--tactile_history_span_ms "${TACTILE_HISTORY_SPAN_MS}" --tactile_history_points "${TACTILE_HISTORY_POINTS}"
            --tactile_self_handover "${TACTILE_SELF_HANDOVER}"
            --tactile_force_contact_weight "${TACTILE_FORCE_CONTACT_WEIGHT}"
            --tactile_event_weight "${TACTILE_EVENT_WEIGHT}" --tactile_event_types ${TACTILE_EVENT_TYPES}
            --tactile_event_window_ms ${TACTILE_EVENT_WINDOW_MS}
            --tactile_event_jump_quantile "${TACTILE_EVENT_JUMP_QUANTILE}"
            --tactile_event_normalize "${TACTILE_EVENT_NORMALIZE}"
            --taptic_exclude_sources ${TAPTIC_EXCLUDE_SOURCES})
if [[ -n "${TAPTIC_VQVAE_SOURCE_IDENTITY}" ]]; then   # only when set: an empty argument can be lost on the way
  TOUCH_ARGS+=(--taptic_vqvae_source_identity "${TAPTIC_VQVAE_SOURCE_IDENTITY}")
fi
# Restartable training (docs/TOUCH_MODULES.md): save the full training state every N micro-steps; continue from it.
RESUME_STATE_EVERY=${RESUME_STATE_EVERY:-0}
RESUME_FROM_STATE=${RESUME_FROM_STATE:-}
if [[ "${RESUME_STATE_EVERY}" != "0" ]]; then TOUCH_ARGS+=(--resume_state_every "${RESUME_STATE_EVERY}"); fi
if [[ -n "${RESUME_FROM_STATE}" ]]; then TOUCH_ARGS+=(--resume_from_state "${RESUME_FROM_STATE}"); fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-trex_midtrain_taptic_${VARIANT}}
RUN_NAME=${RUN_NAME:-${EXPERIMENT_NAME}_$(date +%m%d_%H%M)}

case "$VARIANT" in
  vanilla)
    VARIANT_ARGS=(--cascaded_split_mode fixed --cascaded_split_step 6 --tactile_delay_ms ${TACTILE_ASYNC_DELAYS_MS})
    if [[ "${TACTILE_ASYNC_DELAYS_MS}" != "0" ]]; then VARIANT_ARGS+=(--tactile_async_touch 1); fi ;;
  multisplit)
    VARIANT_ARGS=(--cascaded_split_mode multi --cascaded_splits 0.2 0.4 0.6 --tactile_delay_ms ${TACTILE_DELAY_MS}
                  --risk_stats "${RISK_STATS}" --risk_future_probe ${RISK_FUTURE_PROBE}) ;;
  ours)
    VARIANT_ARGS=(--cascaded_split_mode multi --cascaded_splits 0.2 0.4 0.6 --tactile_delay_ms ${TACTILE_DELAY_MS}
                  --risk_stats "${RISK_STATS}" --use_risk_tokens 1
                  --risk_coefficients 0.25 0.50 0.25 --risk_rate_window_ms 100 --risk_velocity_window_ms 100
                  --risk_joint_source sidecar --risk_rate_mode vector
                  --risk_future_head_weight ${RISK_FUTURE_HEAD_WEIGHT} --risk_future_probe ${RISK_FUTURE_PROBE}
                  --risk_dropout ${RISK_DROPOUT} --risk_oversample ${RISK_OVERSAMPLE}) ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

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
    --variant "${VARIANT}" \
    --seed "${SEED}" \
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
    --flow_time_precision fp32 \
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
    --val_freq "${VAL_FREQ:-500}" \
    --max_val_batches "${MAX_VAL_BATCHES:-50}" \
    "${VARIANT_ARGS[@]}" \
    "${TOUCH_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

echo ">>> TAPTIC ${VARIANT} mid-training finished."
