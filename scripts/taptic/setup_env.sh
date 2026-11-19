#!/usr/bin/env bash
# Create the pinned Python environment in $WORKDIR/venv.
#
#   WORKDIR=/big/disk/trex bash scripts/taptic/setup_env.sh
#
# Optional: PYTHON_BIN (default python3.10), CUDA_HOME (only if the deepspeed
# install asks for it). Works on a node without visible GPUs (login node).
set -euo pipefail
: "${WORKDIR:?set WORKDIR to a directory with ~80 GB free}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.10}
VENV=$WORKDIR/venv
mkdir -p "$WORKDIR/tmp" "$WORKDIR/cache"
export TMPDIR=$WORKDIR/tmp XDG_CACHE_HOME=$WORKDIR/cache PIP_CACHE_DIR=$WORKDIR/cache/pip

# The system Python is usable only if it has venv/ensurepip AND its C headers
# (Python.h): triton compiles a small C helper the first time it runs on a GPU.
# Otherwise (no sudo for python3.10-venv / python3.10-dev), uv provides a
# standalone CPython 3.10.12 that includes both.
system_python_ok() {
    "$PYTHON_BIN" - <<'PY' 2>/dev/null
import ensurepip, os, sys, sysconfig
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)
PY
}
if [[ ! -x "$VENV/bin/python" ]]; then
    if system_python_ok; then
        "$PYTHON_BIN" -c 'import sys; print("Python", sys.version.split()[0], "(system)")'
        "$PYTHON_BIN" -m venv "$VENV"
    else
        # uv: single user-space binary (https://docs.astral.sh/uv/); set UV=/path/to/uv if not on PATH
        UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
        [[ -x "$UV" ]] || { echo "$PYTHON_BIN lacks venv or headers and uv was not found: install uv (user space)"; exit 1; }
        echo "$PYTHON_BIN lacks venv or Python.h: using a uv-managed CPython 3.10.12"
        export UV_CACHE_DIR=$WORKDIR/cache/uv UV_PYTHON_INSTALL_DIR=$WORKDIR/python
        "$UV" python install 3.10.12
        "$UV" venv --seed --python-preference only-managed --python 3.10.12 "$VENV"
    fi
fi
"$VENV/bin/python" -c 'import sys; print("venv Python", sys.version.split()[0])'
PIP=("$VENV/bin/python" -m pip)
"${PIP[@]}" install -U pip
"${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple torch==2.6.0 torchvision==0.21.0
"${PIP[@]}" install -r "$REPO/scripts/taptic/requirements-taptic.txt"

# deepspeed: no ops are prebuilt (training uses torch AdamW + ZeRO-2).
# Its setup imports triton, which fails without a visible GPU driver: on such a
# node, hide triton during the build only (the real triton stays installed).
DS_ENV=(DS_BUILD_OPS=0)
SHIM=""
if ! "$VENV/bin/python" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    SHIM=$(mktemp -d "$WORKDIR/tmp/no_triton.XXXX")
    mkdir "$SHIM/triton"
    echo 'raise ImportError("triton hidden during deepspeed build")' > "$SHIM/triton/__init__.py"
    DS_ENV+=(PYTHONPATH="$SHIM" DS_ACCELERATOR=cpu)
    echo "No GPU visible: building deepspeed with triton hidden."
fi
env "${DS_ENV[@]}" "${PIP[@]}" install --no-build-isolation deepspeed==0.15.4
[[ -n "$SHIM" ]] && rm -rf "$SHIM"

"$VENV/bin/python" -c "import torch, transformers, accelerate, av, pandas, scipy; \
print('ENV OK torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, \
'accelerate', accelerate.__version__)"
"${PIP[@]}" freeze > "$WORKDIR/environment_versions.txt"
echo "Environment ready: $VENV (versions in $WORKDIR/environment_versions.txt)"
echo "Note: 'import deepspeed' only works where a GPU is visible; it is checked by preflight."
