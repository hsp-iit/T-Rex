# Running the TAPTIC T-Rex midtrain on a new cluster

This guide is written for whoever launches the training (a person or a coding agent) on a
machine we have not used before. Follow it top to bottom. Every command is
copy-pasteable. The experiment itself is fixed: **do not change hyperparameters, data or
code**. The only knob that depends on the hardware is the per-GPU batch, and section 5
explains exactly how to set it.


## 1. What runs

The pipeline has two stages. **The first one is already done: you only run the midtrain.**

| Stage | GPUs | What it does | Status / output |
|---|---|---|---|
| `vq` | 1 | Trains the tactile VQ-VAE on the native ~76 Hz SDK fingertip force: 16-sample windows (~210 ms), one code per finger, codebook 64, 30 epochs. Only the native episodes of the train split are used (512 train / 53 val). | **Done by the owner. Do not rerun.** You receive `latest.pt` (8 MB). |
| `midtrain` | N (1 node) | T-Rex cascaded midtrain (baseline: fixed split step 6/10, fp32 flow time, FLARE). The tactile expert gets 5 VQ code tokens + 5 current-force tokens. Legacy 10 Hz episodes are kept for the action expert and masked out of the tactile loss. Starts from Qwen3-VL-2B-Instruct. 10 epochs, effective batch 128, LR 1e-4. | **Your job.** `runs/midtrain/run-<tag>/checkpoint-*/` |

The VQ-VAE checkpoint you receive:

| | |
|---|---|
| File | `latest.pt` (run `run-20261119_131820`, 30 epochs, 1× A100, 9 min) |
| sha256 | `839f17358605580f95ac879c3ca99a069756563f1859154ef1b617fcea52a1c2` |
| Dataset identity inside | `1cb54e3dfd51811f0bdbebf64b0b2e8f532963a55c76b2a9355f0d80d71dc468` (same as `docs/taptic_train_stats.json`) |
| Final metrics | train recon 0.0059, val recon 0.0073, ~59/64 codes used per train batch |

It contains the weights, the force normalization stats and the dataset identity and
split. The midtrain refuses it if your dataset copy is a different revision or split.

Each stage runs a preflight check, then a short probe on the real data, then the full run.
If the probe fails, the full run does not start. Script: `scripts/taptic/run_pipeline.sh`.
The underlying launchers are `tactile_vqvae/train.py` and `scripts/midtrain_taptic.sh`;
details of the tactile pipeline are in `docs/taptic_force_vqvae.md`.

## 2. Rules

1. **Use the cluster's scheduler** (Slurm or whatever the site uses) for anything that uses a
   GPU, including the tests in step 4.4. Respect the site's rules about partitions, time
   limits, bookings and shared disks. Ask the machine owner if you don't know them.
2. **Never share a GPU with someone else's job.** `preflight.py` refuses busy GPUs. Do not
   bypass it.
3. **Do not change anything that defines the experiment**: code, hyperparameters, dataset
   revision, model revision, package versions. The only allowed change is the
   `TRAIN_BSZ` / `NUM_GPUS` pair of section 5, which keeps the effective batch at 128.
   If something else seems to need changing (for example a different torch build for a GPU
   that is not supported, see section 3), stop and ask.
4. **Keep disks ≤ 90% used.** Put `WORKDIR` on a disk with ≥ 80 GB free.
5. **The data stays on this machine.** Copy results only to where the owner says
   (section 8). Do not upload the checkpoints or the data to other services.
6. W&B stays offline (`WANDB_MODE=offline`), so nothing is sent to wandb.ai.

## 3. Requirements

- Linux, NVIDIA GPUs with **≥ 40 GB** each; ≥ 80 GB is recommended (A100 80GB, H100, H200).
  The midtrain uses 1 node with 1, 2, 4 or 8 GPUs.
- GPU architecture supported by torch 2.6.0+cu124: Ampere (sm_80/86), Ada (sm_89),
  Hopper (sm_90). **Blackwell GPUs (B100/B200/GB200, sm_100) are not supported** by this
  build: preflight stops with a clear message. In that case stop and ask the owner: it needs
  a different, untested environment.
- NVIDIA driver compatible with CUDA 12.4 (driver ≥ 550). Check with `nvidia-smi`.
- Python 3.10 (verified with 3.10.12). If `python3 -m venv` is unavailable (no `python3-venv`
  package), `setup_env.sh` uses [uv](https://docs.astral.sh/uv/) instead (user-space binary on `PATH`
  or in `~/.local/bin`). Another version can be passed with
  `PYTHON_BIN`, but then it's untested: record it in the report.
- Internet access from the machine where you run `download.sh` (Hugging Face, PyPI,
  download.pytorch.org).
- Disk in `WORKDIR`: ~10 GB venv, 3 GB dataset, 4 GB model, ~30 GB midtrain checkpoints
  (3 kept, ~8 GB each), small VQ-VAE files. Plan for ≥ 80 GB free.

## 4. Setup (once)

### 4.1 Get the code

```bash
git clone -b mfincato/risk_gated https://github.com/hsp-iit/T-Rex.git
cd T-Rex
git log -1 --format='%H %s'    # write this commit hash in the report
git status --short             # must be empty
```

### 4.2 Choose WORKDIR and create the environment

Everything big (env, data, model, outputs) goes under `WORKDIR`. It must be a directory that
the GPU nodes can read.

```bash
export WORKDIR=/path/on/a/big/disk/trex_taptic     # ≥ 80 GB free
mkdir -p "$WORKDIR"
bash scripts/taptic/setup_env.sh                   # CPU only; ~10 min
```

It prints `ENV OK torch 2.6.0+cu124 cuda 12.4 transformers 4.57.3 accelerate 1.8.1` and writes
`$WORKDIR/environment_versions.txt`. On a node without GPUs, `import deepspeed` fails. That's
expected: preflight checks it later on a GPU node.

If the deepspeed install complains about CUDA (`CUDA_HOME` / `nvcc`), re-run it with
`CUDA_HOME=/path/to/cuda-12.x bash scripts/taptic/setup_env.sh`.

### 4.3 Download data and model (pinned revisions)

```bash
bash scripts/taptic/download.sh     # network only; ~7 GB
```

- Dataset `mfincato/taptic_trex_style_midtrain_v1` @ `31d33a627a70cff8752c621833fa07be7b71e25a`
  (868 episodes, 10 FPS).
- Model `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`.

Both are public. On a Hugging Face rate limit the script waits and retries. A token helps:
set `HF_TOKEN` in your shell. Never write it into files or logs. The script ends with
`DOWNLOAD OK`. The dataset revision must match `docs/taptic_train_stats.json`. The loaders
check this and refuse a different one.

### 4.4 Tests (on 1 GPU, through the scheduler)

```bash
# inside a 1-GPU allocation, from the repo root:
TAPTIC_ROOT=$WORKDIR/data/taptic_trex_style_midtrain_v1 \
TAPTIC_PROCESSOR=$WORKDIR/models/Qwen3-VL-2B-Instruct \
PYTHONPATH=$PWD $WORKDIR/venv/bin/python -m pytest tests/taptic -q
```

Expected: **no failures**. One skip is normal: `TAPTIC_PANDA_URDF` is not set. If anything
fails, stop and report the full output.

## 5. Choose GPUs and batch

The effective batch is fixed at 128 = `NUM_GPUS × TRAIN_BSZ × GRAD_ACCUM`. The script derives
`GRAD_ACCUM` from the other two and refuses combinations that don't divide 128. So use 1, 2,
4 or 8 GPUs, not 3, 5, 6 or 7.

| GPU memory | Start with | If the probe runs out of memory (CUDA OOM) |
|---|---|---|
| ≥ 80 GB | `TRAIN_BSZ=16` (original T-Rex) | `TRAIN_BSZ=8`, then `4` |
| 40–48 GB | `TRAIN_BSZ=4` | `TRAIN_BSZ=2` |

Examples: 8 GPUs × 16 → accum 1; 4 × 16 → accum 2; 2 × 8 → accum 8.

More GPUs make it faster but don't change the result. With 80 GB GPUs, 8 GPUs at batch 16 is
the original configuration. The probe prints `[probe] ... peak CUDA memory X GiB on this rank`.
Write it in the report. Changing `TRAIN_BSZ` after an OOM is allowed and must be in the report.
Nothing else changes.

## 6. Launch

### 6.1 Put the VQ-VAE checkpoint in place and verify it

```bash
mkdir -p $WORKDIR/vqvae
cp /where/you/received/latest.pt $WORKDIR/vqvae/latest.pt
sha256sum $WORKDIR/vqvae/latest.pt
# must print: 839f17358605580f95ac879c3ca99a069756563f1859154ef1b617fcea52a1c2
```

If the hash differs, stop: the file is corrupted or is not the right one. Ask the owner
to send it again.

### 6.2 Launch the midtrain

```bash
WORKDIR=... NUM_GPUS=8 TRAIN_BSZ=16 VQVAE_CKPT=$WORKDIR/vqvae/latest.pt \
  bash scripts/taptic/run_pipeline.sh midtrain
```

Set `NUM_GPUS` and `TRAIN_BSZ` as in section 5. To check commands before using any GPU, add
`DRY_RUN=1`: it prints every command and runs nothing.

Slurm example. Adapt the partition, account and time to the site's rules. GPU count, CPUs
and memory are what the scripts expect.

```bash
#!/bin/bash
#SBATCH --job-name=taptic_midtrain
#SBATCH --nodes=1
#SBATCH --gres=gpu:8            # = NUM_GPUS
#SBATCH --cpus-per-task=64      # ~8 per GPU (dataloader workers)
#SBATCH --mem=256G              # ~32G per GPU
#SBATCH --time=3-00:00:00       # see section 7 for how to size it
#SBATCH --output=%x-%j.log
set -euo pipefail
cd /path/to/T-Rex
export WORKDIR=/path/on/a/big/disk/trex_taptic
NUM_GPUS=8 TRAIN_BSZ=16 VQVAE_CKPT=$WORKDIR/vqvae/latest.pt \
  bash scripts/taptic/run_pipeline.sh midtrain
```

Set `MASTER_PORT` only if the default (`29500 + job id % 1000`) clashes with something.

### 6.3 Only if the owner asks you to retrain the VQ-VAE

Don't do this by default. A new VQ-VAE changes the tactile codes, so it is a different
experiment. If asked: `WORKDIR=... bash scripts/taptic/run_pipeline.sh vq` in a 1-GPU job
(`--gres=gpu:1 --cpus-per-task=8 --mem=48G`, ~10 min on an A100). Then launch the
midtrain without `VQVAE_CKPT`: it reads `runs/vqvae/LATEST_CKPT`. `run_pipeline.sh all`
runs both stages in one job.

## 7. Monitoring and time limits

All output goes to `$WORKDIR/runs/pipeline-<stage>-<tag>.log` (and the scheduler log).
`RUN_INFO-<stage>-<tag>.txt` records the commit, settings, `nvidia-smi` and `pip freeze`.

- `PREFLIGHT OK` means the GPUs are the right ones, free and supported, and the disk is OK.
- Midtrain: the probe line `[probe] N steps done; peak CUDA memory ...` comes first. Then the
  full run logs `action`, `tactile` and `flare` losses, `tactile/valid_frac`, and validation
  every 500 steps. Checkpoints are written every 2 epochs (epochs 1, 3, 5, 7, 9); the last 3
  are kept.

**Time.** We don't have a measured duration on 80 GB GPUs yet. At the start of the full run
the log prints `steps_per_epoch=...`, and a progress bar `Epoch k: ... it/s` shows the speed
in micro-steps per second. Total time ≈ `steps_per_epoch × 10 / (it/s)`, plus validation and
checkpoint saving. Check it against the job's time limit. **There is no
exact resume:** checkpoints contain weights only, no optimizer or scheduler state. A job that
hits its time limit has to restart from scratch. So request enough time from the start. If
the site's maximum is too short, stop and report the estimate to the owner.

## 8. Results to hand back

When the midtrain ends (`>>> midtrain done`), these are the deliverables, all under
`$WORKDIR/runs/`:

- `midtrain/run-<tag>/checkpoint-9-*/` (final), plus the other kept checkpoints. Each has
  `model.pt`, `processor/`, `config.json` and `training_args.json`;
- `pipeline-*.log`, `RUN_INFO-*.txt`, the scheduler logs;
- `wandb/` (offline runs; the owner can sync them later).

Ask the owner where to copy them (for example via `rsync` to a lab machine). Don't put them
on public services.

## 9. Report to send to the owner

Copy this, fill it in, and send it back. Also send it when something fails: then include the
error message and the last ~50 log lines.

```
Commit: <git log -1 hash>          Branch: mfincato/risk_gated   Tree clean: yes/no
VQ-VAE checkpoint: received from owner, sha256 matches 839f1735…a52a1c2: yes/no
Machine: <hostname>, <N> × <GPU model> <memory> GB, driver <version>
Python <version>, torch/transformers/accelerate/deepspeed as in environment_versions.txt: yes/no
Tests (4.4): <passed>/<skipped>/<failed>
Midtrain: NUM_GPUS=<n> TRAIN_BSZ=<b> GRAD_ACCUM=<a> (effective 128)
          probe peak memory <x> GiB/GPU, speed <steps/s>, total time <h>
          final checkpoint <path>, final train/val action loss <x>/<y>, tactile <x>/<y>
Deviations from this guide: <none / what and why>
Results copied to: <where>
```

## 10. Troubleshooting

| Symptom | Cause / what to do |
|---|---|
| `PREFLIGHT FAIL: ... is sm_100 ...` | Blackwell GPU, not supported by this torch build. Stop and ask. |
| `PREFLIGHT FAIL: ... busy` | Someone else's process is on that GPU. Don't proceed. Fix the allocation (`CUDA_VISIBLE_DEVICES`). |
| `PREFLIGHT FAIL: N GPUs visible, expected M` | The allocation and `NUM_GPUS` disagree. |
| `NUM_GPUS*TRAIN_BSZ=... must divide 128` | Use 1, 2, 4 or 8 GPUs and a power-of-two `TRAIN_BSZ`. |
| `CUDA out of memory` in the probe | Halve `TRAIN_BSZ` (section 5) and relaunch. |
| deepspeed install fails on `triton` / CUDA | Run `setup_env.sh` again on the login node (it hides triton automatically when no GPU is visible); set `CUDA_HOME`. |
| `fatal error: Python.h: No such file or directory` (triton, at the first GPU use) | The venv was built from a system Python without headers. Delete `$WORKDIR/venv` and re-run `setup_env.sh`: when headers are missing it uses a uv-managed CPython 3.10.12. |
| `sha256sum` of `latest.pt` differs | Corrupted or wrong file. Ask the owner to send it again. |
| VQ-VAE source identity / train episodes mismatch | Your dataset copy is not revision `31d33a6…`. Re-run `download.sh`. |
| `Address already in use` (port) | Set `MASTER_PORT` to a free port. |
| HF `429` / rate limit | `download.sh` retries by itself; setting `HF_TOKEN` helps. |
| Dataset identity / split mismatch error | Wrong dataset revision. Re-run `download.sh`; don't edit the stats. |
