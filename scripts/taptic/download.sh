#!/usr/bin/env bash
# Download the TAPTIC dataset and Qwen3-VL-2B-Instruct at pinned revisions.
#
#   WORKDIR=/big/disk/trex bash scripts/taptic/download.sh
#
# Both repos are public; an HF token (HF_TOKEN or `hf auth login`) only helps
# against rate limits. ~3 GB dataset + ~4 GB model. Safe to re-run (resumes).
set -euo pipefail
: "${WORKDIR:?set WORKDIR}"
PY=${PYTHON:-$WORKDIR/venv/bin/python}
mkdir -p "$WORKDIR/data" "$WORKDIR/models" "$WORKDIR/tmp" "$WORKDIR/cache"
export HF_HOME=${HF_HOME:-$WORKDIR/hf} TMPDIR=$WORKDIR/tmp XDG_CACHE_HOME=$WORKDIR/cache
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1

for attempt in 1 2 3 4 5 6; do
    if "$PY" - "$WORKDIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
w = sys.argv[1]
d = snapshot_download("mfincato/taptic_trex_style_midtrain_v1", repo_type="dataset",
                      revision="31d33a627a70cff8752c621833fa07be7b71e25a",
                      local_dir=f"{w}/data/taptic_trex_style_midtrain_v1", max_workers=4)
print("DATASET", d)
m = snapshot_download("Qwen/Qwen3-VL-2B-Instruct",
                      revision="89644892e4d85e24eaac8bacfd4f463576704203",
                      local_dir=f"{w}/models/Qwen3-VL-2B-Instruct", max_workers=4)
print("MODEL", m)
PY
    then
        break
    fi
    [[ $attempt == 6 ]] && { echo "download failed after $attempt attempts"; exit 1; }
    echo "download attempt $attempt failed (often HF rate limit); retrying in 5 min"
    sleep 300
done

"$PY" - "$WORKDIR" <<'PY'
import json, sys
w = sys.argv[1]
info = json.load(open(f"{w}/data/taptic_trex_style_midtrain_v1/meta/info.json"))
n = info.get("total_episodes")
print("episodes:", n, "fps:", info.get("fps"))
assert n == 868, f"expected 868 episodes, got {n}"
print("DOWNLOAD OK")
PY
