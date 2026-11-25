#!/usr/bin/env bash
# TAPTIC pipeline: tactile VQ-VAE (1 GPU) -> T-Rex midtrain (NUM_GPUS GPUs).
# Each stage: preflight -> short probe on real data -> full run.
#
#   WORKDIR=/big/disk/trex NUM_GPUS=8 bash scripts/taptic/run_pipeline.sh all
#   WORKDIR=... bash scripts/taptic/run_pipeline.sh vq          # 1-GPU job
#   WORKDIR=... NUM_GPUS=8 bash scripts/taptic/run_pipeline.sh midtrain
#
# Env:
#   WORKDIR      (required) venv/, data/, models/ from setup_env.sh + download.sh; runs/ is written here
#   NUM_GPUS     GPUs for the midtrain (required for midtrain/all)
#   TRAIN_BSZ    per-GPU midtrain batch (default 16, as original T-Rex)
#                GRAD_ACCUM is derived so that NUM_GPUS*TRAIN_BSZ*GRAD_ACCUM = 128 (fixed)
#   VQVAE_CKPT   midtrain stage only: VQ-VAE checkpoint (default: runs/vqvae/LATEST_CKPT)
#   TAPTIC_STATS midtrain stage only: normalization statistics (default docs/taptic_train_stats.json;
#                docs/taptic_train_stats_trex_q.json = original T-Rex per-step q01/q99, clipped)
#   RESUME_CHECKPOINT  midtrain stage only: start from a pretrain checkpoint dir (--resume_source pretrain)
#   MIN_GPU_GIB  minimum memory per GPU for preflight (default 40)
#   NUM_WORKERS  dataloader workers per process (default 6)
#   MAX_DISK_PCT disk usage ceiling checked by preflight (default 90, the lab rule)
#   DRY_RUN=1    print the commands without running anything
set -euo pipefail
STAGE=${1:-all}
case "$STAGE" in all|vq|midtrain) ;; *) echo "usage: $0 [all|vq|midtrain]"; exit 2;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
: "${WORKDIR:?set WORKDIR}"
WORKDIR=$(cd "$WORKDIR" && pwd)
VENV=${VENV:-$WORKDIR/venv}
PY=$VENV/bin/python
TAPTIC_ROOT=${TAPTIC_ROOT:-$WORKDIR/data/taptic_trex_style_midtrain_v1}
MODEL_PATH=${MODEL_PATH:-$WORKDIR/models/Qwen3-VL-2B-Instruct}
RUNS=$WORKDIR/runs
TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EFFECTIVE_BSZ=128
TRAIN_BSZ=${TRAIN_BSZ:-16}
MIN_GPU_GIB=${MIN_GPU_GIB:-40}
NUM_WORKERS=${NUM_WORKERS:-6}
MAX_DISK_PCT=${MAX_DISK_PCT:-90}
TAPTIC_STATS=${TAPTIC_STATS:-$REPO/docs/taptic_train_stats.json}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
DRY_RUN=${DRY_RUN:-0}

[[ -x "$PY" ]] || { echo "missing $PY: run scripts/taptic/setup_env.sh"; exit 1; }
[[ -f "$TAPTIC_ROOT/meta/info.json" ]] || { echo "missing dataset $TAPTIC_ROOT: run download.sh"; exit 1; }
if [[ "$STAGE" != vq && ! -f "$MODEL_PATH/config.json" ]]; then echo "missing model $MODEL_PATH: run download.sh"; exit 1; fi

if [[ "$STAGE" != vq ]]; then
    : "${NUM_GPUS:?set NUM_GPUS (GPUs used by the midtrain)}"
    if (( EFFECTIVE_BSZ % (NUM_GPUS * TRAIN_BSZ) )); then
        echo "NUM_GPUS*TRAIN_BSZ=$((NUM_GPUS * TRAIN_BSZ)) must divide $EFFECTIVE_BSZ"; exit 2
    fi
    GRAD_ACCUM=$((EFFECTIVE_BSZ / (NUM_GPUS * TRAIN_BSZ)))
    PROBE_STEPS=$((2 * GRAD_ACCUM > 8 ? 2 * GRAD_ACCUM : 8))
fi

mkdir -p "$RUNS" "$WORKDIR/tmp" "$WORKDIR/cache"
LOG=$RUNS/pipeline-$STAGE-$TAG.log
exec > >(tee -a "$LOG") 2>&1

export PATH=$VENV/bin:$PATH PYTHONPATH=$REPO
export HF_HOME=${HF_HOME:-$WORKDIR/hf} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR=$WORKDIR/tmp XDG_CACHE_HOME=$WORKDIR/cache TRITON_CACHE_DIR=$WORKDIR/cache/triton
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} DS_ACCELERATOR=cuda
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$RUNS

run() { echo "+ $*"; [[ "$DRY_RUN" == 1 ]] || "$@"; }

info() {
    local f=$RUNS/RUN_INFO-$STAGE-$TAG.txt
    {
        echo "stage=$STAGE tag=$TAG host=$(hostname) date=$(date -Is)"
        echo "repo=$REPO"
        echo "git_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo n/a)"
        echo "git_branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
        echo "git_dirty_files:"; git -C "$REPO" status --porcelain 2>/dev/null || true
        echo "NUM_GPUS=${NUM_GPUS:-} TRAIN_BSZ=$TRAIN_BSZ GRAD_ACCUM=${GRAD_ACCUM:-} EFFECTIVE_BSZ=$EFFECTIVE_BSZ"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_ID=${SLURM_JOB_ID:-}"
        echo "TAPTIC_STATS=$TAPTIC_STATS sha256=$(sha256sum "$TAPTIC_STATS" 2>/dev/null | cut -c1-64)"
        echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-none (base Qwen3-VL)}"
        echo "--- nvidia-smi"; nvidia-smi 2>&1 || true
        echo "--- pip freeze"; "$PY" -m pip freeze 2>&1 || echo "(pip freeze failed)"
    } > "$f"
    echo "run info: $f"
}

stage_vq() {
    local out=$RUNS/vqvae run=run-$TAG
    local first=${CUDA_VISIBLE_DEVICES:-0}; first=${first%%,*}
    echo "=== VQ-VAE (GPU $first) ==="
    CUDA_VISIBLE_DEVICES=$first run "$PY" "$REPO/scripts/taptic/preflight.py" --n-gpus 1 \
        --min-mem-gib "$MIN_GPU_GIB" --storage "$out" --need-gib 20 --max-used-pct "$MAX_DISK_PCT"
    mkdir -p "$out"
    local args=(--data_format taptic --lerobot_root "$TAPTIC_ROOT" --granularity finger
                --window 16 --stride 2 --codebook_size 64 --embed_dim 128 --batch_size 1024
                --recon_target delta --n_quantizers 2
                --lr 1e-3 --weight_tau 0 --num_workers "$NUM_WORKERS" --val_every 2000
                --use_wandb 0 --mixed_precision bf16)
    cd "$REPO"
    # probe: dataset, model wiring and checkpoint I/O on 8 real episodes
    CUDA_VISIBLE_DEVICES=$first run "$PY" -m tactile_vqvae.train "${args[@]}" --epochs 1 \
        --smoke_test 1 --max_episodes 8 --output_dir "$out/probe-$TAG" --run_name probe
    run rm -rf "$out/probe-$TAG"
    CUDA_VISIBLE_DEVICES=$first run "$PY" -m tactile_vqvae.train "${args[@]}" --epochs 30 \
        --output_dir "$out" --run_name "$run"
    if [[ "$DRY_RUN" != 1 ]]; then
        ls -1 "$out/$run"/checkpoint_epoch*.pt 2>/dev/null | head -n -3 | xargs -r rm -f
        echo "$out/$run/latest.pt" > "$out/LATEST_CKPT"
    fi
    VQVAE_CKPT=$out/$run/latest.pt
    echo ">>> VQ-VAE done: $VQVAE_CKPT"
}

stage_midtrain() {
    local out=$RUNS/midtrain
    if [[ -z "${VQVAE_CKPT:-}" ]]; then
        [[ -s "$RUNS/vqvae/LATEST_CKPT" ]] || { echo "no VQ-VAE checkpoint: run the vq stage first"; exit 1; }
        VQVAE_CKPT=$(cat "$RUNS/vqvae/LATEST_CKPT")
    fi
    [[ "$DRY_RUN" == 1 || -f "$VQVAE_CKPT" ]] || { echo "VQ-VAE checkpoint not found: $VQVAE_CKPT"; exit 1; }
    [[ "$DRY_RUN" == 1 || -f "$TAPTIC_STATS" ]] || { echo "TAPTIC_STATS not found: $TAPTIC_STATS"; exit 1; }
    [[ "$DRY_RUN" == 1 || -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT/model.pt" ]] || {
        echo "RESUME_CHECKPOINT has no model.pt: $RESUME_CHECKPOINT"; exit 1; }
    echo "=== midtrain: $NUM_GPUS GPUs x batch $TRAIN_BSZ x accum $GRAD_ACCUM = $EFFECTIVE_BSZ; VQ $VQVAE_CKPT ==="
    echo "=== stats $TAPTIC_STATS; init ${RESUME_CHECKPOINT:-base Qwen3-VL} ==="
    run "$PY" "$REPO/scripts/taptic/preflight.py" --n-gpus "$NUM_GPUS" \
        --min-mem-gib "$MIN_GPU_GIB" --storage "$out" --need-gib 60 --max-used-pct "$MAX_DISK_PCT"
    mkdir -p "$out"
    local common=(TAPTIC_ROOT="$TAPTIC_ROOT" MODEL_PATH="$MODEL_PATH" VQVAE_CKPT="$VQVAE_CKPT"
                  TAPTIC_STATS="$TAPTIC_STATS" RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
                  NUM_GPUS="$NUM_GPUS" TRAIN_BSZ="$TRAIN_BSZ" GRAD_ACCUM="$GRAD_ACCUM"
                  LR=1e-4 EPOCHS=10 NUM_WORKERS="$NUM_WORKERS"
                  MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-$$} % 1000))}")
    # probe: 2 optimizer updates at the real batch; prints peak GPU memory; no checkpoint.
    # An OOM here stops everything: the batch is never changed silently.
    run env "${common[@]}" OUTPUT_DIR="$out/probe-$TAG" RUN_NAME=probe \
        MAX_TRAIN_STEPS="$PROBE_STEPS" VAL_RATIO=0 bash "$REPO/scripts/midtrain_taptic.sh"
    run rm -rf "$out/probe-$TAG"
    run env "${common[@]}" OUTPUT_DIR="$out/run-$TAG" RUN_NAME="run-$TAG" \
        MAX_TRAIN_STEPS=0 MAX_CKPTS=3 VAL_RATIO=0.02 bash "$REPO/scripts/midtrain_taptic.sh"
    echo ">>> midtrain done: $out/run-$TAG"
}

info
[[ "$STAGE" == vq || "$STAGE" == all ]] && stage_vq
[[ "$STAGE" == midtrain || "$STAGE" == all ]] && stage_midtrain
echo ">>> pipeline '$STAGE' finished ($LOG)"
