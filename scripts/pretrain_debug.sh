#!/bin/bash
# Debug launcher for pretraining: runs scripts/pretrain.sh with debug-friendly
# defaults and makes rank 0 wait for the VS Code debugger (debugpy) to attach.
#
# 1. In a tmux session, get a GPU allocation and a shell on the node, e.g.
#      salloc -p gpua --gres=gpu:1 -c 8 --mem=128G -t 02:00:00
#      srun --pty bash
# 2. Run this script there:
#      bash scripts/pretrain_debug.sh
#    It prints "[debugpy] rank 0 listening on <node>:5678" and waits.
# 3. In VS Code, Run and Debug -> "Attach to T-Rex job (debugpy)", enter the node
#    name printed above (e.g. anode04). Execution starts when the debugger attaches.
#
# Every setting below can be overridden from the environment, e.g.
#   TRAIN_BSZ=4 DEBUG_PORT=5690 bash scripts/pretrain_debug.sh
# and all the pretrain.sh overrides (LEROBOT_ROOT, N_EPOCHS, ...) apply too.
#
# Notes
# * NUM_WORKERS=0 loads data in the main process, so breakpoints in
#   qwen_vla/lerobot_dataset.py (__getitem__, collate_fn) are hit. With workers > 0
#   only the training loop is debuggable.
# * With NUM_GPUS > 1 only DEBUG_RANKS listen (default rank 0, on DEBUG_PORT +
#   local rank); the other ranks block at the first collective while you are
#   paused. Pausing longer than the 60 min process-group timeout kills the run.
# * If VS Code cannot reach <node>:5678 directly, forward the port from the login
#   node while the job runs:  ssh -N -L 5678:localhost:5678 <node>
#   and attach to host "localhost".
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TREX_DEBUGPY_PORT=${DEBUG_PORT:-5678}
export TREX_DEBUGPY_RANKS=${DEBUG_RANKS:-0}
export TREX_DEBUGPY_WAIT=${DEBUG_WAIT:-1}

export NUM_GPUS=${NUM_GPUS:-1}
export LEROBOT_ROOT=${LEROBOT_ROOT:-/home/iit.local/lgori/egomix_100gb/retarget}
export TRAIN_BSZ=${TRAIN_BSZ:-2}
export NUM_WORKERS=${NUM_WORKERS:-0}
export N_EPOCHS=${N_EPOCHS:-1}
export SAVE_STEPS=${SAVE_STEPS:-0}          # no mid-epoch checkpoints while debugging
export MAX_CKPTS=${MAX_CKPTS:-1}
export VAL_FREQ=${VAL_FREQ:-0}              # no validation while debugging
export MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-2}
export OUTPUT_DIR=${OUTPUT_DIR:-/home/iit.local/lgori/trex-mod/trex_runs/debug}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-debug}
export MASTER_PORT=${MASTER_PORT:-29531}    # avoid clashing with a real run on the node
# Synchronous CUDA calls: errors point at the line that caused them (slower).
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

echo ">>> debug run on $(hostname): ${NUM_GPUS} GPU(s), batch ${TRAIN_BSZ}, workers ${NUM_WORKERS}, data ${LEROBOT_ROOT}"
echo ">>> attach VS Code to $(hostname):${TREX_DEBUGPY_PORT} once it says 'waiting for the debugger'"
exec bash "${SCRIPT_DIR}/pretrain.sh"
