# trex-touch: touch modules, dataset variants, jobs and evaluation

Worktree `trex_taptic/worktrees/trex-touch`, branch `trex-touch` (uncommitted changes on `1a16caf`; every training
run archives the changed files in `runs/touch/CODE-<name>-<tag>.tgz`). This page lists every option we added to
make the Tactile expert use touch, what it does, why, how to turn it on, and how it is tested. The running record
of results is the midtrain report (`trex_taptic/REPORT-midtrain-20261120_010751.txt`) and the shareable page
(`trex_taptic/eval/TAPTIC_FINDINGS.html`).

## Principles

- **Modular.** Each module has its own flag (trainer) and environment variable (launcher). They combine freely.
- **Off by default.** With every module at its default, training is bitwise identical to `1a16caf`
  (`test_defaults_are_bitwise_identical_to_tip`).
- **Recorded.** Every value goes into the checkpoint's `training_args.json`; the offline evaluation reads the
  dataset selection and the event definition from there.
- **Policy inputs unchanged.** No module feeds `observation.haptics.*` (the operator's glove) to the policy.
- **Evaluation only on the inputs the robot has.** Event flags and contact flags are training targets/weights,
  never model inputs.

## Training modules

| module | trainer flags (launcher variable) | default | what it does |
|---|---|---|---|
| self-handover | `--tactile_self_handover p` (`TACTILE_SELF_HANDOVER`) | 0 | For a fraction p of samples the Tactile expert's training pair starts at the Action expert's own output x_split and follows the straight path to the demo, as at inference. |
| contact weighting | `--tactile_force_contact_weight w`, `--tactile_force_tasks`, `--force_contact_threshold` (`TACTILE_FORCE_CONTACT_WEIGHT`) | 1 | Hand dims (9:21) of force-task samples with fingertip \|F\| > 2 units at the fast tick count w times in the tactile loss. |
| long force history | `--tactile_history_span_ms`, `--tactile_history_points` (`TACTILE_HISTORY_SPAN_MS`, `TACTILE_HISTORY_POINTS`) | 0, 8 | Each finger's tactile vector token carries N force samples spread over the span (last = current). Changes the input size (tacf6_dim = 3 N). |
| touch-event weighting | `--tactile_event_weight w`, `--tactile_event_types`, `--tactile_event_window_ms PRE POST`, `--tactile_event_tasks`, `--tactile_event_labels`, `--tactile_event_jump_quantile`, `--tactile_event_normalize` (`TACTILE_EVENT_WEIGHT`, `TACTILE_EVENT_TYPES`, `TACTILE_EVENT_WINDOW_MS`, `TACTILE_EVENT_JUMP_QUANTILE`, `TACTILE_EVENT_NORMALIZE`) | 1, grasp_onset resistance, 200 1000, 5 force tasks, insert squeeze unscrew push, 0.97, 1 | Samples with a touch event in the window count w times in the tactile loss on every action dim (arm and fingers). Weights rescaled to mean 1 per batch (normalize = 1). Combines multiplicatively with contact weighting. |
| asynchronous touch | `--tactile_delay_ms D ...` for the fixed-split variant (`TACTILE_ASYNC_DELAYS_MS`; job components `fast20`, `fast40`) | 0 | Each training sample's touch is newer than its image by one of the delays D (uniform), as when the tactile expert re-runs on the cached vision pass between images (T-Rex fast ticks). Targets stay the recorded 10 Hz chunk; chunk steps already elapsed at the touch time are not supervised for the tactile expert (slot mask). `fast20` = 0 50 100 150 ms (vision 5 Hz, touch 20 Hz, T-Rex README rates; the multisplit/ours runs already used these delays); `fast40` = 0 25 50 75 ms (vision 10 Hz, touch 40 Hz). |
| dataset selection | `--taptic_exclude_sources REPO ...` (`TAPTIC_EXCLUDE_SOURCES`) | none | Drops every episode recorded in the listed source repos (see dataset variants). |
| touch encoder from another dataset | `--taptic_vqvae_source_identity HEX` (`TAPTIC_VQVAE_SOURCE_IDENTITY`; passed only when set) | none | Accepts a VQ-VAE (and its force statistics) trained on another dataset, named by >= 12 leading hex chars of that dataset's identity; refused for a wrong name or the dataset's own identity. The overlap check of the encoder's training episodes is skipped, so the two datasets must share no recordings. Used by `DATA=off` with the owner's encoder (trained on v1). The contact-risk statistics (`--risk_stats`, loaded by every offline evaluation for the risk levels and the gated condition) follow the same rule: accepted when fitted on this dataset, on unchanged parent episodes of a derived dataset, on the named dataset, or in a cross-dataset evaluation. |
| cross-dataset evaluation | `taptic_eval_other_dataset` (dataset config; set by `offline_eval.py --root` on another dataset) | 0 | Evaluation splits only: the model's own normalization and encoder on a dataset it was not trained on; the checks tying stats and encoder to this dataset's training episodes are skipped. Not a trainer flag. |

Why each one (evidence in the report and the page):

- **Self-handover.** The Tactile expert was trained on noisy points of the recorded answer but used on the Action
  expert's guess, so it never learned to correct; it was slightly worse than vision alone in all four models.
- **Contact weighting.** First attempt at focusing the loss on touch. Not selective: about half of the card
  frames are in contact.
- **Long history.** The card cue needs about 1 s of force (probe: 0.25 s explains 1.1 to 1.4%, 1 s 2.3 to 4%).
- **Touch-event weighting** (user 24 Nov). Touch matters in a few moments of each episode and those moments
  vanish in the average loss. Existing models: the tactile expert ties vision at touch events and loses elsewhere.

### Asynchronous touch and the T-Rex paper (user question 25 Nov)

T-Rex (arXiv 2606.17055, Niu, Liu, Wang et al., v2 18 Jun 2026), read 25 Nov:
- Data at 30 Hz, chunk T_a = 16. The slow stream (vision, latent and action experts; action expert tau 1 -> 0.4 in
  6 Euler steps) runs once per action chunk; the fast stream (tactile expert, tau 0.4 -> 0 in 4 steps) re-runs at
  intra-chunk offsets {0, 4, 8, 12} frames on a clone of the cached KV (no vision pass), and each fast tick
  executes the updated chunk: 4 tactile updates per vision update. The README quotes ~5 Hz action / ~20 Hz tactile.
- Training: delay augmentation, delta ~ U{0, 4, 8, 12} frames between the tactile and the visual-language tokens;
  both experts regress the same target; the tactile expert's pair is built from the demonstration (tau_tac =
  tau_split * Beta(1.5, 1)) on the KV of a detached slow pass (no self-handover, our addition).
- Their ablation (Table 2, robot success, 6 tasks after post-training): full 65, w/o async (synchronous tactile
  refinement) 60, w/o tactile 42, MLP force + deformation 58, deformation only 54, MLP force + VQ-VAE force 59.
  Most of the touch benefit is there without asynchrony; asynchrony adds ~5 points.
- Differences from our setting: robot success after per-task post-training vs our offline error against the
  demonstrations; their fingertips give deformation images + 6-axis wrench, ours 3-axis force; 22-DoF hands.

Our runs so far train and evaluate synchronous touch (delay 0), so the measured touch effect (~0.1 mm) is for
that case only.

Probe, 25 Nov (no training): the multisplit run, trained with touch 0/50/100/150 ms newer than the image (T-Rex
style: tactile pairs from the demonstration, no self-handover), evaluated with touch 50 and 100 ms newer (jobs
5805/5806; `eval/event_windows/event_windows_multisplit_delays_from2.md`, errors over steps 2..15). The touch input
changes in 25% of samples (the rest: free hand at zero force or 10 Hz recordings), but the deployed output moves by
~0.1 mm on average and every error measure is unchanged to 0.01 mm (deployed delay 50 / 100 minus delay 0: +0.00
[-0.00, +0.00] mm on native samples; touch effect at touch moments -0.01 mm at every delay). This model does not use
fresher touch; its tactile expert also ignores touch at delay 0, so the probe cannot say whether a tactile expert
that refines the plan (self-handover) would.

The fixed-split variant's contract (`qwen_vla/contract.py`) refuses touch delays unless `--tactile_async_touch 1`
(set by the launcher only when `TACTILE_ASYNC_DELAYS_MS` is not 0), so the baseline cannot drift by accident. `offline_eval.py --delay_ms D` evaluates with touch D ms newer than the image;
`event_window.py --from-step K` averages errors over chunk steps K..15 (steps elapsed at the touch time excluded:
K = 2 covers delays up to 150 ms).

### Touch events (`taptic/touch_events.py`)

One definition for training, evaluation and analysis. From the 10 fps fingertip force (strongest finger):

- `grasp_onset`: crossing 2 units after at least 0.5 s with every finger below it, reaching 5 within 0.3 s.
- `resistance`: while holding (at least 0.5 s of contact), a rise of at least the task threshold within 0.2 s.
  The threshold is the 0.97 quantile of such rises during contact on the task's native TRAINING episodes. Re-triggers
  within 1 s are suppressed.
- `release`: dropping below 2 units after at least 0.5 s of contact.
- `label`: start of a labelled step (insert, squeeze, unscrew, push). The re-collected unscrew has no labels.

A sample at frame t is an event sample when an event lies in [t − 1 s, t + 0.2 s]. Touch can only inform the
actions after the event (operator reaction 0.2 to 0.6 s). The flag is also required to have valid native touch
at the tick and to belong to `--tactile_event_tasks`.

Measured share with the defaults (v1 held-out episodes, `eval/event_windows/event_windows_tight.md`): 11% of the
force-task samples (pipette 9%, card 13%, glass 8%, unscrew 15%, cylinder 4%). Resistance thresholds in units per
0.2 s: pipette 16.3, card 41.9, glass 17.8 (cylinder, unscrew and fruit have no native training episodes in v1, so
only grasp onsets there; the derived dataset gives unscrew a threshold). A first, broader definition (any contact
change, any jump above the 95th percentile, any label change, 2.6 s window) flagged 43 to 69% and is not used.

With w = 5 and 11% event samples, normalized weights are about 3.5 for event samples and 0.7 for the others, so
event samples carry about 38% of the tactile loss of force tasks instead of 11%.

## Restartable training (user request 25 Nov, after C1 crashed at 8.7 of 15 h)

Checkpoints hold weights only, so a crash used to mean a restart from scratch. Now:

- **Trainer** (`scripts/midtrain.py`, off by default): `--resume_state_every N` saves the full training state every N
  micro-steps (rounded up to whole optimizer steps) and at each epoch end to `<output_dir>/resume_state`: model and
  optimizer (the DeepSpeed engine with its ZeRO partitions, via `accelerator.save_state`), RNG states, the LR
  scheduler, the step counters and the position in the epoch (`progress.json`). Written to `resume_state.tmp` and
  swapped in, so a crash during a save keeps the previous state (~30 GB per run, one copy). Deleted when the run
  completes. `--resume_from_state auto` continues from it (fresh start if there is none); the saved run must have
  the same world size, batch, accumulation, epochs, steps and seed. The data order per epoch is fixed (seed +
  epoch), and the sampler skips the samples of the interrupted epoch already trained on
  (`EpisodeGroupedSampler.skip_samples`). Launcher: `RESUME_STATE_EVERY`, `RESUME_FROM_STATE` (passed only when set).
- **Job** (`jobs/midtrain_touch.sbatch`): `RESUME_STATE_EVERY=1250` by default (~45 min); runs from a frozen code
  snapshot `<run>/code` taken at the first start (committed tree + uncommitted changes, so a restart uses identical
  code even if the worktree changes); records the run in `jobs/state/run-<job id>.env`; on a training failure it
  requeues itself with `scontrol requeue` (same job id, so the evaluations waiting on it with afterok stay queued;
  `--requeue`, `--open-mode=append`) up to `MAX_RESTARTS=2` times and continues with `RESUME_FROM_STATE=auto`.
  Self-requeue checked on this cluster (cpu test: same id, restart count 1, the afterok dependent ran after).
- Not exact to the bit: data-loader worker RNG (random touch delays) and dropout restart from new states.
- C2 (5492) and the feedback-OFF run (5756) started with the old code: no saved state, no automatic restart.

### Data-loading hang (C1, job 5491)

C1 died at step 14107 of 25150: a data-loader worker on rank 0 froze inside a threaded pyarrow dataset scan
(`taptic/data.py` `frames()`, per-episode filter; py-spy dumps `jobs/diag/pyspy-5491-*.txt`), and rank 1 timed out
in the gradient all-reduce after 10 min. Fix (approved): each process reads the frame rows once, without Arrow's
thread pool, and slices episodes from that table (`Source._frame_rows`); identical rows (test). Applies to jobs
started after 25 Nov 13:00.

## Dataset variants

| preset | root | selection | stats file | content |
|---|---|---|---|---|
| `v1` | `data/taptic_trex_style_midtrain_v1` | none | `docs/taptic_train_stats_trex_q.json` | HF `mfincato/taptic_trex_style_midtrain_v1` @ `31d33a6`; 868 episodes; unscrew = 47 old recordings with 10 Hz synthetic touch. |
| `v1u30` | `data/taptic_trex_style_midtrain_v1_plus_unscrew30` | `--taptic_exclude_sources TAPTIC/unscrew_the_bottle` | `docs/taptic_train_stats_trex_q_v1u30_fresh.json` | v1 with the old unscrew replaced by the 50 re-collected recordings (glove full scale 30, native ~80 Hz touch). 785 training episodes. |
| `v1both` | same derived root | none | `docs/taptic_train_stats_trex_q_v1both_fresh.json` | Both unscrews. 827 training episodes. |
| `off` | `data/nohaptic_trex_style_midtrain_v1` | none | `docs/taptic_train_stats_trex_q_off.json` | Feedback-OFF recordings (HF `NOHAPTIC/nohaptic_trex_style_midtrain_v1` @ `635711a7`), 887 episodes, 798 training; native touch on all 9 tasks; owner's touch encoder via `TAPTIC_VQVAE_SOURCE_IDENTITY=1cb54e3dfd51811f` (v1). Never mixed with ON data (user rule 24 Nov). |

#### Feedback-OFF preset: choices and checks (`jobs/nohaptic/off_prep.py`, `eval/nohaptic/off_prep.md`, jobs 5743/5744)

- **Own action bounds.** Without feedback the operators close the fingers further: the haptic v1 / v1u30 bounds
  would clip 3.8% / 2.8% of all OFF training-chunk values and up to 45-46% of one finger joint (pipette, card,
  split cups); the OFF train bounds clip 1.0%. So `off` uses its own train-only stats (like every dataset).
- **Owner's touch encoder, unchanged** (the run guide forbids retraining it; the robot sensor is the same, so both
  runs differ only in the demonstrations). Caveat: OFF operators squeeze harder, so 27% of OFF contact frames have
  a finger channel above the top of the encoder's force range (ON 16%; card 18% vs 4%, split cups 19% vs 3%,
  pipette 27% vs 12%, unscrew 41%): above it the model sees "at least this hard".
- **Touch moments are more frequent without feedback** (glass 19% vs 9%, split cups 20% vs 11%, clearing 19% vs
  12%; resistance thresholds higher): event weighting spreads over a larger share of OFF samples (normalized to
  mean 1 per batch, so the loss scale is unchanged).
- **Remaining asymmetries:** 19 OFF pipette recordings vs 50; cylinder and fruit have native touch only in OFF (in
  ON they are legacy and excluded from the tactile loss), so OFF supervises the tactile expert on more samples;
  separate sessions/days; subtask labels from a VLM pass (not used by the event types in use).

### The derived dataset (`taptic_trex_style_midtrain_v1_plus_unscrew30`)

Built by `scripts/taptic/append_recordings.py` (job `jobs/data/build_unscrew30.sbatch`, 5484) from v1 plus
HF `TAPTIC/unscrew_the_bottle_fs65` @ `4aa2d9aa` (local copy `data/taptic_unscrew_fs65`). The name says fs65 but
the glove full scale is 30 units (user; matches the fit of the logged glove signal).

- Episodes 0..867: v1, unchanged. Verified: identical frame rows (action, state, glove, subtask labels, timestamps)
  and episode metadata.
- Episodes 868..917: the 50 re-collected unscrew recordings, one episode per recording, caption / primitive /
  object copied from the old unscrew. No primitive labels (`subtask_index` = −1, `subtasks` = []). Val = 5
  recordings (10%, numpy seed 42). Videos copied as new files (not re-encoded), sidecars copied unchanged.
- `meta/derived_from.json`: parent path and identity, unchanged episodes, source repo and revision, validation
  episodes, verification results. `README.md` has an appended section.
- The frozen VQ-VAE was trained on v1 (identity `1cb54e3d`). The loader accepts it on the derived dataset only
  through `derived_from.json`, and only if every episode it was trained on is unchanged (true: it never saw unscrew).
- Stats: the loader requires a stats file fitted on the same content and the same training episodes, so every
  (root, selection) pair has its own file, made by `scripts/taptic/make_stats.py`. Two bound choices exist:
  `_fresh` (bounds refitted) and the plain files (`--action-bounds-from` the v1 file, data change only).
  **Use `_fresh`** (job default `STATS_BOUNDS=fresh`): the new unscrew turns the wrist much more than the old one,
  and the v1 bounds would clip 17 to 41% of its wrist-rotation values (old unscrew 4 to 9%; fresh bounds 6 to 12%).

### Other data (not used for training)

- `data/nohaptic_trex_style_midtrain_v1`: HF `NOHAPTIC/nohaptic_trex_style_midtrain_v1` @ `635711a7`, the same 9
  tasks recorded by the same operators without glove feedback (after the haptic data). Analysis only; the user
  decided not to mix it with the haptic data (it averages two grip styles). Results: `eval/nohaptic/`.
- `data/taptic_unscrew_fs65`: the raw re-collected unscrew (source of the derived dataset). Glove analysis:
  `eval/unscrew_fs65/`.

## Jobs

`jobs/midtrain_touch.sbatch` (2 x H200, the vanilla final's settings):

```
sbatch -J taptic_lane_a --dependency=singleton --export=ALL,ARM=<components>,DATA=<preset> jobs/midtrain_touch.sbatch
```

- `ARM`: components joined by `+`: `selfhand` (0.5), `contact3`, `hist1s`, `event5`; nothing or `vanilla` = vanilla.
  Aliases: `selfw` = selfhand+contact3 (lane A, job 5298), `hist` = selfhand+contact3+hist1s (lane B, job 5299).
- `DATA`: `v1` (default), `v1u30`, `v1both`, `off`. `STATS_BOUNDS`: `fresh` (default) or `parent` (derived presets).
- `EXTRA_TOUCH="KEY=VAL ..."`: any launcher variable, e.g. `TACTILE_EVENT_WINDOW_MS="200 1500"` or
  `TACTILE_EVENT_TYPES="grasp_onset resistance label"`.
- Output: `runs/touch/<ARM>-<DATA>/run-<tag>`, log `runs/touch/pipeline-<ARM>-<DATA>-<tag>.log`, run info
  (data, selection, stats sha, code state) `RUN_INFO-...txt`, code archive `CODE-...tgz`.

Data tools: `jobs/data/build_unscrew30.sbatch` (derived dataset + stats), `jobs/data/fresh_stats.sbatch`
(fresh-bound stats + the clipping comparison `jobs/data/compare_bounds.py`), `jobs/nohaptic/off_prep.sbatch`
(OFF stats + bound / encoder-range / touch-moment checks).

## Evaluation

- `jobs/eval/offline_eval.py` with `TREX_CODE=<this worktree>`: reads the dataset root, the selection and the event
  definition from the checkpoint's `training_args.json` (old checkpoints: v1, none). Saves a per-sample
  `touch_event` flag in `per_sample.npz`. Defaults otherwise unchanged (`--root` overrides the root).
- Cross-dataset evaluation: `offline_eval.py --root <other dataset> [--exclude_sources ...]` scores a model on a
  dataset it was not trained on (model's own normalization and encoder); `meta.json` records root, selection and
  `other_dataset`. `jobs/offline_eval.sbatch` also takes `CKPT` as a quoted pattern (last match), so an evaluation
  can be queued before the training run's start time (its directory name) is known.
- `jobs/eval/event_window.py --code <this worktree>` (`jobs/eval/event_window_ab.sbatch TAG DIRS [ROOT]`): tactile
  expert minus vision alone at event samples vs the rest, per task, plus a run-time gate (tactile expert only if a
  finger touched in the last second); real minus zeroed touch at events vs the rest; with several eval dirs on the
  same samples, each model's deployed output minus the first dir's (paired).
- `jobs/eval/pred_behaviour.py` (`jobs/eval/pred_behaviour.sbatch TAG ROOT DIRS`): contact behaviour in the
  predictions at the first contact of a grasp (thumb/index/middle extra closing over 0.3 s and 1 s, squeeze past
  the measured finger position at 1 s), recorded vs deployed vs vision alone vs zeroed touch, and paired model
  differences on the same inputs. First run (vanilla/A/B on v1 val, `eval/behaviour/pred_behaviour_on_models.md`):
  only 63 contact samples (35 force); in force tasks all models squeeze ~0.17 rad less past contact than the
  operators, touch changes it by ~0.01 rad.
- Probes (`touch_analysis.py`, `window_probe.py`, `task_force_analysis.py`, `glass_check.py`, `insertion_probe.py`).
- Data-only comparisons (no models): `jobs/nohaptic/compare.py`, `onoff_behaviour.py`, `demo_metrics.py`,
  `unscrew_fs65.py`, `glove_curve.py`, `glove_fingers.py`.

## Tests

`tests/taptic/test_touch.py` (13 tests) plus the rest of `tests/taptic`:

```
TAPTIC_ROOT=<v1> TAPTIC_PROCESSOR=<Qwen3-VL-2B-Instruct> TAPTIC_DERIVED_ROOT=<derived> TAPTIC_VQVAE=<vqvae/latest.pt> \
PYTHONPATH=. python -m pytest tests/taptic -q
```

Job: `jobs/smoke/test_touch_cpu.sbatch` (cpu node; `test_vqvae_train_smoke` needs a GPU and is deselected).
Covers: history points and padding; event definitions and window direction; event weight math; defaults bitwise
identical to `1a16caf`; self-handover pair math and gradient routing; contact weighting; history model end to end;
dataset event flag; trainer event weighting (training only, logged unweighted value); derived dataset provenance,
VQ acceptance and stats/selection consistency; derived dataset loading with selection (wrong selection refused).
GPU path check through the real launcher: `jobs/smoke/gpu_check_touch.sbatch` (1 L40S, 12 steps per variant).

## Run ledger

| run | job | ARM | DATA | status |
|---|---|---|---|---|
| vanilla final | 4892 | vanilla | v1 | done (reference) |
| lane A | 5298 | selfhand+contact3 | v1 | done 25 Nov 03:10; eval 5707 (5493 lost to a node failure) |
| lane B | 5299 | selfhand+contact3+hist1s | v1 | done 25 Nov 03:15; eval 5494 (results below) |
| lane C1 | 5491 | selfhand+contact3+event5 | v1 | FAILED 25 Nov 11:54 at step 14107 (data-loader hang); rerun 5866 |
| lane C1 rerun | 5866 | selfhand+contact3+event5 | v1 | queued after the feedback-OFF run (~03:00 26 Nov): C1 vs A = event weighting |
| lane C2 | 5492 | selfhand+contact3+event5 | v1u30 (fresh stats) | running since 25 Nov 03:16 (~15 h): C2 vs C1 = re-collected unscrew |
| feedback OFF | 5756 | selfhand+contact3+event5 | off | running since 25 Nov 11:54 (took lane A when C1 failed): OFF vs C2 = demonstrations recorded without glove feedback |
| fast20 | 5840 | selfhand+contact3+event5+fast20 | v1u30 | held until the restart check 5897 passes (H200 after C2), then lane B: fast20 vs C2 = touch 20 Hz / vision 5 Hz |
| fast40 | 5867 | selfhand+contact3+event5+fast40 | v1u30 | lane A after the C1 rerun: fast40 vs C2 = touch 40 Hz / vision 10 Hz |

C1/C2 use lane A's recipe (no history) so that each comparison changes one thing; the 1 s history (lane B) can be
added once its result is known. Queued evaluations: lane A final 5493, lane B final 5494 (L40S, `--stride 4
--mean_seeds 6`, same as the vanilla final 4964), then the event-window comparison 5495
(`eval/event_windows/event_windows_laneAB.md`). 5493 hit NODE_FAIL (cluster) and 5495 was cancelled with it; rerun as
5707 (lane A eval) and 5709 (vanilla / A / B event windows, after 5707). Lane B vs vanilla alone: 5712
(`eval/event_windows/event_windows_laneB.md`).

Evaluation chain, updated 25 Nov afternoon: batch jobs `jobs/offline_eval_batch.sbatch` (several evaluations of one
checkpoint per job; the gpu-l40s QOS allows 10 submitted jobs per user, shared with PiHand): C2 own 5750, C2 batch
5852 (on the OFF held-out set + delays 25/50/75/100/150), OFF batch 5853, fast20 batch 5854, C1 batch 5868, fast40
batch 5869; comparisons 5856-5863 and 5870-5873 (see jobs/SUBMITTED_20261125.txt). Earlier chain, superseded:
C1 own 5749 -> C2 own 5750 -> C1 on the v1u30 held-out set 5751 -> C2 on the OFF held-out set 5758 -> OFF own 5759
-> OFF on the v1u30 held-out set 5760. Comparisons (cpu): C1 vs A 5752, C2 vs C1 on v1u30 5753; OFF vs C2 touch
moments 5761 (OFF held-out) / 5762 (v1u30 held-out), contact behaviour 5763 / 5764
(`eval/event_windows/event_windows_OFFvsC2_*.md`, `eval/behaviour/pred_behaviour_OFFvsC2_*.md`).
Feedback ON vs OFF is read on numbers inside each model (touch use: real vs zeroed touch at touch moments; tactile
expert vs its own vision expert) and on the same inputs (both models on both held-out sets; contact behaviour).

### Results: lane B (selfhand+contact3+hist1s) vs vanilla, held-out native-touch samples

| measure | vanilla | lane B |
|---|---|---|
| tactile expert − vision alone, pos mm (all native) | +0.21 ± 0.09 (worse) | −0.29 ± 0.20 |
| same, rotation deg | +0.034 ± 0.011 | −0.085 ± 0.022 |
| same at touch moments, pos mm [95% CI] | −0.03 [−0.65, +0.59] | −1.59 [−2.79, −0.47] |
| how far the tactile expert moves the plan (mm) | 4.7 | 12.0 |
| how far zeroing touch moves it (mm) | 0.42 | 1.27 |
| error change from zeroing touch, pos mm | +0.013 | +0.022 ± 0.021 |
| real − zeroed touch at touch moments, pos mm / fingers deg | +0.01 / +0.01 | −0.08 [−0.23, +0.05] / −0.05 [−0.10, −0.00] |
| deployed output (split 0.4) − vanilla's, pos mm | | −0.66 [−1.18, −0.14] |
| same, force tasks only | | −0.38 [−0.99, +0.26] (fingers −0.17 [−0.30, −0.06]) |

### Results: lane A (selfhand+contact3) and lane B against each other (eval 5707, windows 5709 / 5737)

| measure (native held-out) | vanilla | lane A | lane B |
|---|---|---|---|
| vision alone (action expert), pos mm | 26.66 | 26.93 | 26.50 |
| tactile expert − vision alone, pos mm | +0.21 ± 0.09 | −0.39 ± 0.18 | −0.29 ± 0.20 |
| same at touch moments [95% CI] | −0.03 [−0.63, +0.66] | −0.59 [−1.65, +0.53] | −1.59 [−2.66, −0.50] |
| prediction moved by zeroing touch, mm | 0.42 | 0.80 | 1.27 |
| error change from zeroing touch, mm | +0.013 | −0.002 ± 0.022 | +0.022 ± 0.021 |
| real − zeroed touch at touch moments, mm | +0.01 | −0.12 [−0.25, +0.01] | −0.08 [−0.24, +0.05] |
| deployed − vanilla deployed, mm | | −0.33 [−1.08, +0.37] | −0.66 [−1.18, −0.15] |
| same, force tasks | | +0.28 [−0.64, +1.22] | −0.38 [−1.04, +0.21] |

Lane B − lane A deployed: force tasks −0.66 [−1.30, −0.13], at force-task touch moments −1.35 [−2.99, −0.12]
(pipette, glass). That gap is already in the vision-only action expert (26.50 vs 26.93 mm; pipette 13.91 vs 15.52,
glass 23.74 vs 24.82), and the tactile stage adds the same in both lanes. The action expert never sees touch at
inference (the tactile tokens follow its pass in the sequence), so the gap is not use of the history: it is either
run-to-run variation or an indirect training effect through shared weights; one seed per run cannot tell which.
Run-to-run variation of the vision part (~0.4 mm here) is as large as the effects measured, so C1/C2 are judged on
the within-model numbers (tactile expert vs its own action expert; real vs zeroed touch), which cancel it.

Conclusions: self-handover fixes the tactile expert (from +0.2 mm worse to 0.3-0.4 mm better than vision) with or
without history; neither lane makes touch itself matter (real vs zeroed ~0.1 mm). History concentrates the tactile
stage's gain on touch moments (CIs overlap) but adds no touch effect -> C1/C2 keep lane A's recipe.

Reading: the tactile expert now improves on the action expert (self-handover), mostly at touch moments, and the
deployed output beats vanilla on native samples; but real touch versus zeroed touch is still worth only ~0.1 mm
(about 0.5% of the error), so the gain is refinement of the vision plan, not touch. Per task (deployed vs vanilla):
wiping −2.2, card −0.75, split cups −0.6, pipette and glass ~0; legacy-only tasks (old unscrew, fruit, cylinder: no
native touch in val) +0.3..+1.4. Fixed split 0.4 is the deployment setting: the risk gate (0.2/0.6) is +2.6 mm worse
for lane B because training only uses split 0.4. `event_window.py` (25 Nov) adds the cross-model table and the
real-vs-zeroed touch columns (backup `event_window.py.bak-20261125`).

GPU check (`jobs/smoke/gpu_check_touch.sbatch`, jobs 5489/5490): on both v1 and v1u30 the launcher, argument
parsing, dataset construction (event flags, selection, derived dataset) and the forward pass with the weighted
loss ran on the GPU with real data; the 48 GB L40S then ran out of memory at the first backward (full fine-tune
needs the H200), as expected.

## Taxel encoder (plan approved 25 Nov, steps 1-3; step 4 = training, needs a go)

Idea (user, 25 Nov): build a fingertip "deformation map" from the XHand taxels (120 per fingertip, 3-axis each) and
encode it with an autoencoder, as T-Rex does with its deformation images (their DeformAE: ResNet-18 stem..layer3,
self-supervised AE, frozen). Steps: 1 layout, 2 autoencoder on training episodes, 3 probe beyond force, 4 training run.
Step 1 DONE (job 5980): candidate layout from the public MuJoCo model github.com/tsingqingyun/xhand1 (vendor-style
T16/T30 coordinate files, copied to trex_taptic/docs_ext/xhand_taxel_layout/ with provenance) matches our data:
Spearman(distance, correlation) -0.17..-0.35 vs null 0.00 +- 0.03; 10 columns around the tip x 12 along the finger.
Official XHand manual: packet = combined force FX/FY/FZ (our SDK force) + 3x120 taxel forces + 20 temperatures; no
layout. Paper: Sharpa Wave fingertips (camera-based); "we record and use the estimated deformation depth and the
6-axis net wrench" (no reconstruction step described, so the map most likely comes from the hand's software).

## Changelog

- 24 Nov: self-handover, contact weighting, long history (lanes A/B).
- 24 Nov evening: touch-event weighting, `taptic/touch_events.py`, dataset selection by source repo, derived
  datasets (`append_recordings.py`, `make_stats.py`, `derived_from.json`), modular job presets, offline eval reads
  the selection from the checkpoint and saves event flags, event-window tool on the shared definition.
- 25 Nov afternoon (user: save what is needed to restart trainings): restartable training (trainer + job self-requeue
  + code snapshot), frame-table hang fix, batch evaluation jobs. Tests: 23 touch, full suite 100 passed (5886/5887);
  snapshot identical to the worktree (5890); GPU save-and-resume check 5897 on 1 H200 after C2 (the L40S runs out
  of memory at batch 1); fast20 is released only if it passes, otherwise the C1 rerun and fast40 are held too.
- 25 Nov midday (user: test 40 Hz touch / 10 Hz vision and T-Rex's 20 Hz / 5 Hz; read the paper): launcher
  `TACTILE_ASYNC_DELAYS_MS` for the fixed-split variant, job components `fast20` / `fast40`, `event_window.py
  --from-step`, `event_window_ab.sbatch EW_ARGS`; tests for both; probe of the multisplit model (trained with
  0/50/100/150 ms) evaluated at 50 / 100 ms (jobs 5805/5806, windows 5807/5816): no use of fresher touch.
  Contract opt-in `--tactile_async_touch` (the test suite caught the vanilla contract refusing delays).
  Tests: 19 touch, full suite 96 passed (5831/5832).
- 25 Nov afternoon (user: train on feedback-OFF data paired with C2): `--taptic_vqvae_source_identity`,
  `taptic_eval_other_dataset` + `offline_eval.py --exclude_sources`, `DATA=off`, OFF stats, `pred_behaviour.py`,
  CKPT patterns in `offline_eval.sbatch`; launcher passes the encoder flag only when set (the full suite caught an
  empty argument lost on the way to the parser). Tests: 16 touch tests, full suite 93 passed (jobs 5746, 5755).
  GPU smoke test of the cross-dataset evaluation (5768) failed on the contact-risk statistics (tied to v1; would
  also have stopped C2's own evaluation on the derived dataset) -> risk statistics follow the encoder rule;
  17 touch tests (5770), smoke rerun 5778, full suite 5779.
- 25 Nov: lane B results; event-window tool adds deployed-output-vs-reference and real-vs-zeroed-touch columns;
  `jobs/eval/event_window_ab.sbatch` takes `TAG` and `DIRS`; lane A results and lane B vs lane A.
