# TAPTIC: SDK force at native rate + tactile VQ-VAE

Alternative tactile input for the TAPTIC midtrain (`--taptic_tactile_input sdk_force`).
The raw-taxel baseline (`taxels`, default) is unchanged.

## Why

303 of 868 episodes (fruit_in_the_basket, insert_game_object_cylinder,
unscrew_the_bottle) have **no recorded taxels**, only the 15-D SDK `calc_force`
at 10 Hz. Taxels cannot be recovered from 15 force values, and upsampling 10 Hz
force would fabricate exactly the contact transients the native stream captures.
So:

- tactile signal = recorded SDK fingertip force `[5 fingers x (Fx,Fy,Fz)]`, raw counts;
- only **native ~76.4 Hz** samples are used, never interpolated;
- legacy 10 Hz episodes stay in the midtrain (action expert, FLARE) but are
  **masked out of the tactile-expert loss**.

## Native force history (`taptic/force_history.py`)

Shared by VQ-VAE training and the midtrain loader, so the codebook sees the same
windows the policy encodes.

- Window = 16 consecutive native samples (~210 ms), ending at the latest sample
  with `read_end_ns <= tick` and `available_ns <= tick` (causal).
- Segments break at read gaps > `--tactile_gap_ms` (50), reset/error/drop
  changes and sequence jumps. Pre-roll before the first frame, samples after the
  last frame and samples inside recorder pauses (> 150 ms) are excluded; the
  latest causal sample at a resuming frame starts the new segment.
- Short histories (episode start, after a reset/pause) are left-padded with the
  segment's first sample, as the original T-Rex loader did at episode start.
- A history whose newest sample is older than `--tactile_stale_ms` (50) is invalid.

## VQ-VAE (`tactile_vqvae`, `--data_format taptic`)

- Input `[16, 5, 3]`, `granularity=finger` -> 5 codes per window,
  `per_finger_dim=3`, `n_hands=1`, `input_signal=taptic_sdk_force`.
- Train/val = dataset `suggested_split`, native episodes only.
- Normalization: train-only per-channel q01/q99 min-max to [-1, 1] (clipped),
  stored in the checkpoint with dataset identity and episode lists.
- Recon weighting scale `weight_tau`: auto = p75 of train raw window magnitudes.

```bash
TAPTIC_ROOT=... OUTPUT_DIR=... bash tactile_vqvae/scripts/train_vqvae_taptic.sh
# defaults: window 16, stride 2, codebook 64, embed 128, 30 epochs; SMOKE=1 for a quick check
```

Check in the log: recon decreasing, perplexity stable, `active` close to the
codebook size, `revived` going to ~0.

## Midtrain

```bash
TAPTIC_ROOT=... MODEL_PATH=.../Qwen3-VL-2B-Instruct VQVAE_CKPT=<run>/latest.pt \
OUTPUT_DIR=... [RESUME_CHECKPOINT=<pretrain ckpt>] bash scripts/midtrain_taptic.sh
```

Tactile expert tokens per sample: 5 VQ codes (embedded frozen VQ-VAE, encoded
on the fly from the raw history) + 5 current-force tokens (`tacf6_dim=3`,
normalized with the VQ-VAE stats). Batch adds `tactile_f6_history [B,16,5,3]`
(raw), `tactile_loss_mask [B]`, `force_history_n_real [B]`; W&B logs
`tactile/valid_frac`. The loader rejects a VQ-VAE trained on another dataset,
on non-training episodes, with a different window, or the original F6 VQ-VAE.

Split/precision are the fixed baseline (`--cascaded_split_step 6`, tau 0.4,
`--flow_time_precision fp32`; `legacy_bf16` for original T-Rex arithmetic).

## Tests

`tests/taptic/test_force_vqvae.py` (CPU, synthetic root with the real schema,
`tests/taptic/synthetic_root.py`): segment/causality/padding rules, native-only
train-only stats, VQ-VAE shapes and smoke training, VLA encoder vs VQ-VAE
equivalence (original 60-D F6 path unchanged), loader history/mask, config
guards, masked loss, code+force tactile tokens. A tiny-MoT end-to-end test runs
only with the pinned transformers 4.57.
