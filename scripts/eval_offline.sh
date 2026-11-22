#!/bin/bash
# Offline open-loop eval (scripts/eval_offline.py) of a pretraining checkpoint on the episodes its run held out.
#   sbatch scripts/eval_offline.sh                                  # checkpoint-0-12000 of run 1115_1059
#   CKPT=<checkpoint dir> sbatch scripts/eval_offline.sh            # another checkpoint of the same data
#   EVAL_ARGS="--max_samples 3000 --n_noise 4" sbatch scripts/eval_offline.sh
# Reads the same data as the training job (/fastwork/lgori/claude_scratch/pretrain_4gpu.sbatch): the egomix
# snapshot views, copied to node-local disk (BeeGFS fails some parquet reads). The val split is drawn from each
# view's episode count, so it is training's held-out set only while the views hold what they held then: do not
# refresh the snapshot (scripts/egomix_snapshot.sh) before evaluating. Both checks below stop the job otherwise.
#SBATCH --job-name=trex_eval_offline
#SBATCH --partition=gpua
#SBATCH --nodes=1
#SBATCH --gres=gpu:nv-a100-sxm4-80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/fastwork/lgori/trex_runs/pretrain_egomix/slurm-%x-%j.out
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_90
#SBATCH --mail-user=leonardo.gori@iit.it
set -eo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/lgori/trex-mod}
EXP_DIR=/fastwork/lgori/trex_runs/pretrain_egomix/trex_pretrain_egomix_panda_xhand
CKPT=${CKPT:-${EXP_DIR}/trex_pretrain_egomix_panda_xhand_1115_1059/checkpoint-0-12000}
TRAIN_STATS=${TRAIN_STATS:-${EXP_DIR}/lerobot_norm_stats.json}   # the training jobs' stats cache
SNAPSHOT=${SNAPSHOT:-/fastwork/lgori/egomix_snapshot/retarget}
OUT_DIR=${OUT_DIR:-${CKPT}/offline_eval}
NUM_WORKERS=${NUM_WORKERS:-10}

cd ${PROJECT_ROOT}
export PATH=/fastwork/lgori/envs/trex/bin:$PATH PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
echo ">>> $(date +%T) on $(hostname): ${CKPT}"

# Per-job dir: a training job on the same node uses /tmp/egomix_views_$USER and removes it when it ends.
L=/tmp/egomix_eval_${USER}_${SLURM_JOB_ID:-$$}
trap "rm -rf $L" EXIT
LEROBOT_ROOT=$(python -m utils.egomix_snapshot --local-copy ${SNAPSHOT} $L/views)
export HF_DATASETS_CACHE=$L/hf
mkdir -p ${OUT_DIR}

# The views must be the ones training read (same names, episode and frame counts). The training stats cache
# records them; a copy with the roots pointed at this job's views lets the loader reuse it instead of
# recomputing the stats over every frame (~15 min). The eval normalizes with the checkpoint's own stats anyway.
python - "${TRAIN_STATS}" "${LEROBOT_ROOT}" "${OUT_DIR}/lerobot_norm_stats.json" <<'EOF'
import json, os, sys
train_stats, views, out = sys.argv[1:]
with open(train_stats) as f:
    data = json.load(f)
settings = data[next(iter(data))]["settings"]
want = {os.path.basename(r["root"]): r for r in settings["roots"]}
have = {n for n in os.listdir(views) if os.path.isfile(os.path.join(views, n, "meta/info.json"))}
problems = [f"{n}: not in training" for n in sorted(have - set(want))]
problems += [f"{n}: in training, missing now" for n in sorted(set(want) - have)]
for n in sorted(have & set(want)):
    with open(os.path.join(views, n, "meta/info.json")) as f:
        info = json.load(f)
    r = want[n]
    if (info["total_episodes"], info["total_frames"]) != (r["total_episodes"], r["total_frames"]):
        problems.append(f"{n}: {info['total_episodes']} episodes/{info['total_frames']} frames, "
                        f"training had {r['total_episodes']}/{r['total_frames']}")
    r["root"] = os.path.realpath(os.path.join(views, n))
if problems:
    sys.exit("snapshot views differ from training's, the val split would not be its held-out set:\n  "
             + "\n  ".join(problems))
settings["roots"].sort(key=lambda r: r["root"])
with open(out, "w") as f:
    json.dump(data, f, indent=2)
print(f">>> {len(have)} views match training's; stats cache for this copy: {out}")
EOF

# Data settings default to scripts/pretrain.sh's (eval_offline.py); seed and val_ratio are training's.
python scripts/eval_offline.py \
    --checkpoint_path ${CKPT} \
    --out_dir ${OUT_DIR} \
    --cuda 0 \
    --lerobot_root ${LEROBOT_ROOT} \
    --lerobot_stats_path ${OUT_DIR}/lerobot_norm_stats.json \
    --seed 42 \
    --val_ratio 0.02 \
    --num_workers ${NUM_WORKERS} \
    ${EVAL_ARGS}

echo ">>> $(date +%T) done: ${OUT_DIR}"
