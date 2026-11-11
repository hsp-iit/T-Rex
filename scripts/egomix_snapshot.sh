#!/bin/bash
# Build or refresh the snapshot views of the egomix outputs (utils/egomix_snapshot.py) on a cpu node:
# only the episodes the conversion jobs have finished, as LeRobot v3 datasets for --lerobot_root.
#   sbatch scripts/egomix_snapshot.sh                                    # refresh (append what finished since)
#   sbatch --export=ALL,SNAPSHOT_ARGS=--dry-run scripts/egomix_snapshot.sh   # only the plan
#   sbatch --export=ALL,SNAPSHOT_ARGS=--verify scripts/egomix_snapshot.sh    # views still match their sources?
# Then: LEROBOT_ROOT=/fastwork/lgori/egomix_snapshot/retarget sbatch scripts/pretrain.sh
#SBATCH --job-name=egomix_snapshot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/fastwork/lgori/egomix_10tb_logs/%x-%j.out
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/lgori/trex-mod}
# Any python with pyarrow, pandas, numpy and pyyaml (the trex env has them); the world-up recheck
# runs egomix's own python, set in the config.
PYTHON=${PYTHON:-/fastwork/lgori/envs/trex/bin/python}
CONFIG=${CONFIG:-${PROJECT_ROOT}/config/egomix_snapshot.yaml}
cd ${PROJECT_ROOT}
echo "$(date) on $(hostname)"
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-} ${PYTHON} -m utils.egomix_snapshot ${CONFIG} ${SNAPSHOT_ARGS:-}
