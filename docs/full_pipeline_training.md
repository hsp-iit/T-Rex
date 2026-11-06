# T-Rex training pipeline: pretrain → midtrain → post-train

This note describes the three-stage T-Rex training pipeline as implemented on the
`full-pipeline` branch, and compares it against the paper
([arXiv 2606.17055v1](https://arxiv.org/abs/2606.17055)).

**Sources and conventions**

- **Code** is `origin/full-pipeline`, unless a file is marked *(main)*. The
  post-training script exists on both branches and is almost identical; the
  differences are listed in [§9](#9-known-issues-and-gaps-in-the-branch).
- **Paper** is the arXiv HTML v1: Sec. 4.3 "Training Recipe", App. A (Table 4),
  App. B (delay augmentation), App. C (tactile encoders) and App. G (dataset).
- Legend used in the comparison tables: ✅ code matches paper · ⚠️ differs ·
  📄 paper only · 💻 code only.

---

## 0. Summary

| | Pretrain (stage 1) | Midtrain (stage 2) | Post-train (stage 3) |
|---|---|---|---|
| Script | `scripts/pretrain.py` / `.sh` | `scripts/midtrain.py` / `.sh` | `scripts/train.py` / `.sh` |
| Data | Egocentric **human** video, 22,889 h according to the paper ("following EgoScale"); code path `mecka_merged` | ~100 h of teleoperated **robot** data with tactile, 200+ objects, 22 primitives | ~100 demonstrations per task |
| Experts trained | latent + action | latent + action + **tactile** | latent + action + tactile |
| Cameras | head only | head + 2 wrist cameras | head + 2 wrist cameras |
| Tactile | none | F6 + deform maps + VQ-VAE codes, **delayed** by δ ∈ {0,4,8,12} | F6 + deform maps + VQ-VAE codes, δ = 0 |
| Proprioception | **yes** (62-D, with tracking-error noise) | no | no |
| Losses | `L_act + 0.5·L_future` | `L_act + 1.0·L_tac + 0.5·L_future` | same as midtrain |
| Epochs (scripts) | 1 | 10 (released checkpoint: epoch 6) | 100 |
| Initialized from | Qwen3-VL-2B-Instruct | pretrain checkpoint (`checkpoint-0-610000`) | midtrain checkpoint (`checkpoint-5-19464`) |
| Released checkpoint | `miniFranka/T-Rex_pretrain_mecka22k_epoch1` | `miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6` | — (you produce it) |

Two auxiliary models are trained outside the three stages: the **tactile VQ-VAE**
(code included) and the **deformation encoder** (only its weights path appears;
no training code).

---

## 1. Shared architecture and mechanics

These parts are the same in all three stages.

### 1.1 Model (`qwen_vla/modeling_vla.py`, `qwen_vla/modeling_qwen3vl_mot.py`)

- **Backbone:** Qwen3-VL-2B (28 layers, hidden size 2048), turned into a
  Mixture-of-Transformers with three experts. Each expert has its own Q/K/V/O,
  q/k norms, layer norms and MLP, and all experts share one joint attention:
  - **latent** expert: the original Qwen weights (vision + language tokens + flare queries);
  - **action** expert: initialized as a copy of the latent expert when there is no checkpoint to resume;
  - **tactile** expert: FFN intermediate size 1536 (`--tactile_intermediate_size 1536`).
- **Heads and embedders:**
  - `x_embedder` (noisy action → H), `t_embedder` (flow time τ → H);
  - `final_layer` (action velocity) and `final_layer_tactile` (tactile velocity);
  - `state_embedder` (62-D state → H, created only if `use_robot_state`);
  - `tacf6_embedder` (6-D wrench per finger → H);
  - `deform_encoder` (ResNet-18 stem + layers 1–3 + 3×3 convs to 128 channels; output 128×15×15) followed by `deform_proj` (28800 → H);
  - `tactile_code_embedder` (VQ code → H), plus an **embedded, frozen VQ-VAE** that encodes raw F6 history on the fly;
  - `flare_queries` (32 learnable tokens) and `flare_proj`.
- **Initialization** (`initialize_vla_weights`): Xavier for all new linear layers;
  `final_layer.fc2` is zero-initialized so `v_act` starts at 0. The tactile head
  is *not* zero-initialized.
- **Always frozen:** the Qwen ViT (`visual.*`), `deform_encoder.*` and `tactile_vqvae.*`.

### 1.2 Token layout and routing

```
[ head img + text | flare queries (32) ] [ "fast" imgs | state? | τ | noisy action ×16 ]   [ tactile obs | τ | x_τ ×16 ]
 └────────── latent expert ───────────┘ └──────────────── action expert ─────────────────┘ └───── tactile expert ─────┘
```

- The mask is causal. Latent tokens cannot see action tokens. The action expert
  sees the latent tokens but never tactile. The tactile expert sees everything
  through the cached `[latent | action@τ_split]` KV.
- "Fast" images (the wrist cameras) are placed after the text, so they are
  routed to the **action** expert, not the latent expert.
- Action and tactile tokens get 1-D M-RoPE positions that continue after the
  largest latent position.

### 1.3 Action space

- **62-D** = `[L-arm 9 | L-hand 22 | R-arm 9 | R-hand 22]`, predicted as a chunk of **16**.
- Arm 9-D: `R0ᵀ(t − t0)` (3) + first two columns of `R0ᵀR` (6D rotation). Both are
  relative to the end-effector pose at the **start of the chunk** ("delta-base").
- Hand: 22 absolute joint angles of the Sharpa Wave hand.
- Normalization: `clip(2·(x − q01)/(q99 − q01) − 1, −1, 1)`. Pretraining uses
  per-dimension statistics; midtrain and post-train use per-(chunk step, dimension)
  statistics with shape `[16, 62]`, because delta-base actions grow with the chunk
  index.

### 1.4 Flow matching (action expert, all stages)

```
τ ~ Beta(1.5, 1)·0.999 + 0.001         (τ = 1 is pure noise)
x_τ = τ·ε + (1−τ)·a
L_act = MSE( final_layer(h_action), ε − a )      # trained on the full τ ∈ (0, 1]
```

### 1.5 Cascaded tactile flow (midtrain and post-train)

1. **No gradient:** `forward_flow_action_partial` runs the action expert for
   6 of 10 Euler steps (Δτ = −0.1), from noise down to τ_split = 0.4. It then
   re-encodes the partially denoised `x_split` into the KV cache, so the cache
   holds `[latent | action @ τ_split]`.
2. Sample `τ_tac = τ̃ · 0.4` with `τ̃ ~ Beta(1.5, 1)`, and set `x_τ = (1−τ_tac)·a + τ_tac·ε`.
3. `tactile_flow_train_step` feeds `[tactile obs | τ_tac | x_τ]` to the tactile
   expert, which attends to the cached KV:
   `L_tac = MSE(final_layer_tactile(h), ε − a)`. This is the **same target** as
   the action expert, not a residual.
4. **Tactile dropout:** with probability 0.1 per batch, every tactile input is
   zeroed (values, not tensor shapes).
5. Because the KV cache is computed under `torch.no_grad()`, `L_tac` updates only
   the tactile expert and its embedders, plus the shared `x_embedder`/`t_embedder`.
   It does **not** update the latent or action experts.

Tactile observation tokens are `[VQ code tokens (10: one per finger per hand) | F6
tokens (10) | deform tokens (10)]` (`_embed_tactile_observations`).

**At inference** (`scripts/test.py`): a "slow" request runs the 6 action-expert
steps once per chunk and caches the KV. Each "fast" request clones that cache and
runs the remaining 4 steps with the tactile expert on fresh tactile data (the
README describes this as ~5 Hz slow and ~20 Hz fast).

### 1.6 Future-frame prediction ("flare" / `L_future`, all stages)

- 8 future head frames are used, at offsets `t + 4·(k+1)`, k = 0…7, so they cover
  32 frames (~1.07 s at 30 fps).
- The 32 query tokens (8 frames × 4 tokens) sit in the latent stream. Their
  last-layer hidden states (`flare_layer_index = −1`) go through `flare_proj`.
- Targets: frozen ViT features of each future frame, adaptively average-pooled to 4 tokens per frame.
- `L_future = mean(1 − cos(pred, target))`, weight 0.5.

---

## 2. Stage 1: pretraining (`scripts/pretrain.py`, `pretrain.sh`)

**Goal:** align the VLM with action prediction on large-scale egocentric human data.

### Data format (`EgoDexPretrainFlareDataset`)

```
DATA_ROOT/<batch>/pretrain_manifest.json       {"episodes":[{"episode_dir", "num_frames"}],
                                                 "statistics":{action, state q01/q99, tracking_error mean/std}}
DATA_ROOT/<batch>/<ep>/pretrain.hdf5            states [N,62], action_chunks [N,16,62], attrs["language"]
DATA_ROOT/<batch>/<ep>/ego_view.mp4             head video (falls back to *head*.mp4)
```

- `pretrain.sh` sets `DATA_ROOT=.../data/mecka_merged`. The docstring says
  "EgoDex large-scale pretraining".
- **The script that produces `pretrain.hdf5` from raw human video is not in the
  repo.** `midtrain.py` refers to it as `data/midtrain/scripts/gen_pretrain_mecka_parallel.py`.
  This missing step is where hand-pose estimation, wrist chunk-deltas and Sharpa retargeting happen.
- `utils/convert_egodex_to_lerobot.py` converts this format to LeRobot v3
  (`include_wrist=False`, `include_tactile=False`), and `--data_format lerobot` is accepted.

### Inputs per sample

- **Image:** the head frame resized to 384×288. It is passed **twice**, as the
  slow image (latent expert) and as the fast image (action expert), because the
  human data has no wrist cameras.
- **Text:** `attrs["language"]`.
- **State:** the 62-D human "state" plus Gaussian **tracking-error noise**. The
  noise is sampled in an axis-angle representation for the arm and per joint for
  the hand, with mean and std taken from the manifest statistics.
- **Flare targets:** 8 future head frames, decoded from the same mp4.

### What is trained

- latent expert, action expert, `x_embedder`/`t_embedder`, `state_embedder`,
  `final_layer`, `flare_queries`/`flare_proj`;
- **frozen:** the ViT and all tactile parameters (`*_tactile*`). The model is
  built with `use_tactile_deform=False`.
- The action expert is copied from the latent expert at start.

**Loss:** `L_act + 0.5·L_future`.

### Hyperparameters (`pretrain.sh`)

| | value |
|---|---|
| GPUs | 8 per node × `NUM_MACHINES` |
| Batch size | 16 per GPU, gradient accumulation 1 |
| Learning rate | 1e-4, cosine decay to 0, **warmup 3%** |
| Weight decay | **0.01** (not applied to biases or norms) |
| Gradient clipping | 1.0 |
| Precision / optimizer | bf16, AdamW |
| Epochs | **1** |
| Checkpointing | every 5000 steps, with full optimizer, scheduler and RNG state (`--save_full_state 1`) |
| Validation | 2% of **episodes** held out, evaluated every 1000 steps |

### Data loading

- `EpisodeGroupedSampler` shuffles the order of episodes but keeps each
  episode's frames contiguous, so a worker decodes each episode only once. It
  shards indices per rank itself.
- For that reason the training dataloader is deliberately **not** passed to
  `accelerator.prepare`. Doing so would shard a second time, so each rank would
  see 1/N² of the data.
- On resume, already-seen batches can be skipped cheaply at the sampler
  (`--resume_skip_data`).

---

## 3. Auxiliary models: tactile VQ-VAE and deformation encoder

### Tactile VQ-VAE (`tactile_vqvae/`, `tactile_vqvae/scripts/train_vqvae_f6.sh`)

- **Data:** the **merged midtrain root**, i.e. the robot tactile data. Each sample
  is a `[16, 5, 6]` F6 window from one hand; left and right hands are treated as
  separate samples. Windows are taken with stride 4.
- **Encoder:** a 1-D temporal conv with 2 strided blocks, mean-pooled over time
  to a 256-D embedding. With `granularity=finger`, the conv weights are shared
  across fingers and a learned finger-ID embedding is added, giving **one code per finger**.
- **Quantizer:** EMA codebook with **K = 64** (the script sets `CODEBOOK=64`;
  the dataclass default is 1024), decay 0.99, commitment weight 0.25, and
  dead-code revival every 200 steps.
- **Loss:** magnitude-weighted reconstruction MSE (`weight_alpha=2`, `weight_tau=4`),
  which penalizes errors on high-force frames more.
- **Training:** 30 epochs, batch size 256, learning rate 3e-4, 8 GPUs.
- The trained VQ-VAE is **embedded in the VLA and frozen**
  (`--use_tactile_vqvae 1 --vqvae_ckpt …`). It runs in fp32 and encodes the raw
  16-frame F6 history on the fly. `utils/merge_vqvae_into_ckpt.py` bakes it into
  a checkpoint; `extract_codes.py` (pre-computing codes offline) is the older path.

### Deformation encoder

- `qwen_vla/DeformAE.py` defines only the **encoder**; the weights are loaded
  from `sharpa_wave_deform_encoder.pth`.
- **There is no autoencoder training code** in either branch. Input is a
  240×240 single-channel deformation map per fingertip, 10 per sample.

---

## 4. Stage 2: midtraining (`scripts/midtrain.py`, `midtrain.sh`, `prepare_midtrain_merged.py`)

**Goal:** move from human data to the robot, with multi-view observations and
executable actions, and train the tactile expert.

### Data

- `prepare_midtrain_merged.py --merged_root R --source nv=/… --source bkl_play=/…`
  merges several robot sources into a single root of symlinks:
  ```
  R/<prefix>_<category>/pretrain_manifest.json      (+ injected tactile_f6 q01/q99)
  R/<prefix>_<category>/<demo>/pretrain.hdf5        states, action_chunks, [tactile_f6 N×10×6], tracking_error
  R/<prefix>_<category>/<demo>/raw.h5               original episode .h5 (tactile_deform N×5×240×240 per hand, per-hand F6)
  R/<prefix>_<category>/<demo>/ego_view.mp4, left_wrist.mp4, right_wrist.mp4
  ```
- The head video can be cropped at decode time, using a per-batch `head_crop_box` from the manifest.
- **Normalization:** per-(step, dimension) action statistics and per-dimension
  F6 statistics are recomputed from the raw episodes and cached in
  `midtrain_statistics_c16_d62.json`. If F6 statistics are missing, it falls back
  to the manifest and then to the VQ-VAE checkpoint's statistics.
- Only the in-repo robot sources are merged. **Nothing in the code mixes human
  data into midtraining.**

### Inputs per sample (`MidtrainTacFlareDataset`)

- **Images:** `[head (slow) | text | right wrist, left wrist (fast)]`, each resized to 384×288.
- **Tactile, current frame t:** F6 `[10,6]` and deformation maps `[10,240,240]`.
- **Tactile, delayed frame t+δ:** with **δ ~ Uniform{0, 4, 8, 12}**, clamped to the
  episode end. It provides F6, deformation maps and the 16-frame F6 history that
  ends at t+δ (left-padded with the first frame). **Only the delayed version is
  fed to the tactile expert.** This mimics inference, where a fast tick happens
  0–12 frames after the visual context was cached.
- **State:** off (`--use_robot_state 0`).

### What is trained

- Everything except the ViT, the deformation encoder and the VQ-VAE.
- On resume, `--resume_source` defaults to `pretrain`, so the **tactile expert is
  re-initialized** (Xavier for matrices, zeros for 1-D parameters; see §9).

**Loss:** `L_act + 1.0·L_tac + 0.5·L_future`, with tactile dropout 0.1, 10 Euler
steps in total and the split after 6.

### Hyperparameters (`midtrain.sh`)

Same as pretraining (8 GPUs × nodes, batch 16, LR 1e-4 with cosine decay, warmup
3%, weight decay 0.01, clipping 1.0), except:

- **10 epochs**, checkpoint every 2 epochs;
- 2% of episodes held out for validation, evaluated every 500 steps;
- uses the same episode-grouped sampler and does not prepare the dataloader.

---

## 5. Stage 3: post-training (`scripts/train.py`, `train.sh`; on both branches)

**Goal:** fine-tune on about 100 demonstrations of a single task while keeping
the tactile-reactive behavior.

### Data

- **JSON format (default):** the JSON is produced by
  `utils/gen_json_tac_deltabase_eef_bimanual_parallel.py`. Each record has
  `input_image_slow`, `input_image_fast`, `input_prompt`, `action` [16·62],
  `state_fast`, `tactile_f6`, `tactile_image_deform` and optionally
  `tactile_codes`. Normalization statistics are read from `<json>_statistics.json`.
- **LeRobot v3** (`--data_format lerobot`, `qwen_vla/lerobot_dataset.py`) is also supported.

### Inputs

- Same views as midtraining. Images are separate PNG files on disk, not decoded video.
- **Delay δ is always 0:** the "delayed" tactile input is the current reading.
- Flare frames are found by rewriting the frame index in the slow image's file
  path (`image{idx}_…`).
- State is off.

### What is trained

- Same as midtraining. With `--resume_source midtrain`, the tactile weights are
  **kept** from the checkpoint.

**Loss:** same as midtraining.

### Hyperparameters (`train.sh`)

| | value |
|---|---|
| Epochs | 100, checkpoint every 50 |
| Learning rate | 1e-4, cosine decay to 0, **no warmup** |
| Weight decay | **0** |
| Batch size | 16 per GPU |
| Validation | 5% of **frames** (random per sample, not per episode), every 500 steps |
| Sampler | plain shuffled dataloader, passed to `accelerator.prepare` |

### Outputs

Each checkpoint contains `model.pt`, `training_args.json`, `stats_data.json`
and `processor/`. `test.py` rebuilds the architecture from `training_args.json`
and denormalizes actions with `stats_data.json`.

---

## 6. Differences between the three stages

| Aspect | Pretrain | Midtrain | Post-train |
|---|---|---|---|
| Dataset class | `EgoDexPretrainFlareDataset` | `MidtrainTacFlareDataset` | `SftDataset` (JSON) / `TRexLeRobotDataset` |
| On-disk format | `pretrain.hdf5` + `ego_view.mp4` | `pretrain.hdf5` + `raw.h5` + 3 mp4 | JSON + PNG files (or LeRobot v3) |
| Embodiment | human | robot (Vega-1 + Sharpa) | robot |
| Slow image (latent expert) | head | head (optionally cropped) | head (cropped by the JSON generator) |
| Fast images (action expert) | **head again** (duplicated) | right + left wrist | right + left wrist |
| State token | ✅ 62-D + tracking-error noise | ❌ | ❌ |
| Tactile expert | frozen, unused | **re-initialized**, then trained | continued from midtrain |
| Tactile delay δ | — | {0, 4, 8, 12} | 0 |
| Tactile dropout | — | 0.1 | 0.1 |
| Losses | act + future | act + tac + future | act + tac + future |
| Action normalization | per dimension (from manifest) | per (step, dimension), recomputed | per (step, dimension), from JSON statistics |
| Epochs | 1 | 10 | 100 |
| Warmup / weight decay | 3% / 0.01 | 3% / 0.01 | 0 / 0 |
| Validation split | 2% of episodes | 2% of episodes | 5% of **frames** |
| Sampler | episode-grouped, manual sharding | episode-grouped, manual sharding | random, via accelerate |
| Resume | full state (optimizer, scheduler, RNG) | weights only | weights only |

All three stages share: Qwen3-VL-2B MoT, 62-D × 16 actions, 384×288 images,
Beta(1.5, 1) flow times, learning rate 1e-4 with cosine decay to 0, batch 16 per
GPU, bf16, gradient clipping 1.0, and the flare settings (8 steps × 4 tokens,
stride 4, last layer, weight 0.5).

---

## 7. Paper vs. code

### 7.1 Data

| Item | Paper | Code | |
|---|---|---|---|
| Pretraining data | 22,889 h egocentric human video, "following EgoScale"; arm and hand motion retargeted into a unified action space | `DATA_ROOT=…/mecka_merged`, EgoDex-format loader; checkpoint name `mecka22k`; **no data and no preprocessing code released** | ⚠️ |
| Pretraining views | head view for the latent expert | head frame also duplicated as the action expert's "fast" image | 💻 |
| Midtraining data | 100 h teleoperation, 7,755 episodes, 207 objects × 22 primitives (502 pairs) | merges robot sources (`nv`, `bkl_play` in the example); checkpoint name `mecka23k_ucb100`; mixture not documented | ⚠️ unclear |
| Released dataset | "open-source 100-hour" dataset | README: ~50 h subset on HF | ⚠️ |
| Post-training data | ~100 demonstrations per task (ablation: 10–200) | any JSON or LeRobot set | ✅ |
| Image resolution | captured at 640×360 | resized to 384×288 for training | 💻 (training resolution not stated in paper) |

### 7.2 Model

| Item | Paper | Code | |
|---|---|---|---|
| Backbone | Qwen3-VL-2B, 28 layers, hidden size 2048 | same | ✅ |
| Tactile expert FFN size | 1536 (0.62B parameters) | `--tactile_intermediate_size 1536` | ✅ |
| Attention kernel | Flash Attention 2 | `F.scaled_dot_product_attention` with an additive float mask (PyTorch picks the backend) | ⚠️ |
| Tactile token fusion (Eq. 2) | [VQ force embedding; projected current force; projected deformation] | `[codes | tacf6_embedder | deform_proj]` | ✅ |
| Force VQ-VAE | T = 16, 2 strided conv blocks, mean-pool to 256-D, K = 64, EMA, dead-code reseeding, magnitude-weighted MSE, shared conv with finger-ID embedding, one token per finger | same (`granularity=finger`) | ✅ |
| VQ-VAE training data | not stated | merged midtrain robot data | 💻 |
| Deformation encoder | ResNet-18 first 3 stages + 3×3 conv to 128 channels; self-supervised autoencoder pretraining; frozen | encoder only; frozen; **no training code** | ⚠️ |
| Proprioception input | not mentioned | pretraining **uses** state (with tracking-error noise); mid/post do not | 💻 |

### 7.3 Objectives and flow

| Item | Paper | Code | |
|---|---|---|---|
| Flow target | `v* = ε − A`, `x_τ = (1−τ)A + τε` | same | ✅ |
| τ for action expert | Beta(1.5, 1) on (0, 1] | `Beta(1.5,1)·0.999 + 0.001` | ✅ |
| τ for tactile expert | `τ_split · τ̃`, τ̃ ~ Beta(1.5, 1), KV cache from a detached slow pass | same | ✅ |
| Loss weights | λ_tac = 1.0, λ_future = 0.5 | same | ✅ |
| Future prediction | "future visual latent prediction"; frames, stride and layer not given | 8 frames × 4 tokens, stride 4, last layer, cosine loss on pooled ViT features | 💻 (details) |
| Euler steps | N = 10; 6 slow + 4 fast; τ_split = 0.4 | `cascaded_total_steps 10`, `split_step 6` | ✅ |
| Delay augmentation | δ ~ U{0, 4, 8, 12} (App. B) | midtraining only; post-training uses δ = 0 | ⚠️ (stage not stated in paper) |
| Tactile dropout | not mentioned | p = 0.1 per batch | 💻 |
| Tracking-error state noise | not mentioned | pretraining only (state off later) | 💻 |
| Image augmentation | not mentioned for T-Rex | none in code | ✅ |

### 7.4 Optimization and compute

| Item | Paper (Table 4, "SFT") | Code | |
|---|---|---|---|
| Optimizer / LR | AdamW, 1e-4, cosine decay to 0 | same in all stages | ✅ |
| Weight decay / warmup | 0 / 0 | post-train 0 / 0 ✅; pre/mid **0.01 / 3%** | ⚠️ (pre/mid not in paper) |
| Batch size | 16 per device, gradient accumulation 1 | same | ✅ |
| Hardware | 24× H100 | 8 GPUs per node × `NUM_MACHINES` (default 1) | ⚠️ (24 = 3 nodes) |
| DeepSpeed | ZeRO **stage 1** | `config/sft_qwen.yaml`: `zero_stage: 2` | ⚠️ |
| Epochs / steps | not given for any stage | pretrain 1 epoch (midtrain resumes from step 610,000); midtrain 10 (released: epoch 6, step 19,464); post 100 | 💻 |

---

## 8. What you need to reproduce each stage

| Stage | You must provide | Missing from the repo |
|---|---|---|
| Pretrain | `pretrain.hdf5` (`states [N,62]`, `action_chunks [N,16,62]`, `language`), `ego_view.mp4`, and `pretrain_manifest.json` with q01/q99 (+ tracking-error) statistics | the human-video data, and the hand-pose → delta-base → Sharpa retargeting pipeline (`gen_pretrain_mecka_parallel.py`) |
| VQ-VAE | the merged midtrain root with F6 | — |
| Deformation encoder | the `.pth` weights | autoencoder training code |
| Midtrain | robot episodes in the layout from §4, plus the VQ-VAE and deformation encoder checkpoints | the midtrain data (only ~50 h of the T-Rex dataset are public) and the robot-episode → `pretrain.hdf5` converter |
| Post-train | JSON from `gen_json_tac_deltabase_eef_bimanual_parallel.py`, or LeRobot v3 | — |

---

## 9. Known issues and gaps in the branch

1. **`full-pipeline` does not import as committed.** Its
   `qwen_vla/modeling_vla.py` has `from diffusion import …` and
   `from models import DeformEncoder`, but `qwen_vla/diffusion.py` and
   `qwen_vla/DeformAE.py` exist only on `main`. Copy them over and restore the
   `main` imports (`from .diffusion …`, `from .DeformAE …`).
2. **Tactile RMSNorm gains start at 0 in midtraining.** The tactile
   re-initialization in `midtrain.py` (and in `full-pipeline:train.py`) sets every
   1-D `_tactile` parameter to 0, RMSNorm weights included. That disables every
   tactile norm branch at the start: zero q/k/v and an all-zero final norm. The
   branch can still learn, because gradients reach the gains, but it starts
   badly. `main:train.py` fixes this (`elif "norm" in name: ones_`) with the
   comment "zero would kill its branch". The released midtrain checkpoint may
   have been trained before or after this fix.
3. **The pretrained state pathway is thrown away.** Pretraining learns
   `state_embedder` and a token layout that includes a state token. Midtraining
   and post-training run with `use_robot_state 0`, so those weights are ignored
   and the action-expert sequence loses a token.
4. **In pretraining the action expert sees a copy of the head image** as its
   "fast" view, while midtraining switches it to the wrist cameras.
5. **Post-training validation is split by frame**, so neighbouring frames of the
   same episode land in both train and validation, and the validation loss is
   optimistic. Pretraining and midtraining split by episode.
6. **Delay δ differs between midtraining (random) and post-training (0).** This
   is intentional according to a code comment, but at deployment fast ticks do
   arrive 0–12 frames late.
7. **Small issues:**
   - `--tactile_loss_weight` is parsed but never used (`--cascaded_loss_weight` is the one applied);
   - validation never computes the flare loss;
   - there is a leftover `torch.set_printoptions(profile="full")` in the post-train loop;
   - midtraining resumes with weights only (no optimizer or scheduler state), unlike pretraining.
8. **Paper/code mismatches to keep in mind:** ZeRO stage 1 vs. 2, Flash Attention 2
   vs. SDPA, and a 100 h vs. ~50 h released dataset.
