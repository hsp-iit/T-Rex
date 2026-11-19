# Native tactile sampling analysis, baseline fixes and multi-split training

Implemented on **`mfincato/risk_gated`**, starting from `b068f70` and preserving
all pre-existing uncommitted review artifacts. No branch switch/merge/cherry-pick;
`feat/TAPTIC` was not modified. No raw recording or learned checkpoint was
rewritten. No full training or risk/architecture extension was launched.

Order followed: native-frequency experiment and its initial results were
reported first, then timestamp/validity and flow-time fixes, then multi-split
integration and tests. This report consolidates all phases.

## 1. Sampling-frequency experiment

Source: the same pinned local TAPTIC export verified in the preceding review
against Hub revision `31d33a627a70cff8752c621833fa07be7b71e25a`. Seven native
recordings: **0, 48, 98, 197, 553, 554, 771**, covering wiping, pipette, glass,
cups, clearing and insertion. This comparison used **174.059 seconds** after
exclusions; it is not an exhaustive audit of all native recordings.

[Full numerical evidence](frequency_analysis.json) includes all ten grid phases
per frequency, episode diagnostics and source hashes. The original SDK `force`
vectors were used directly; they were not recomputed from taxels.

### Method and exclusions

- Remove synthetic rows explicitly marked `interpolated=True`; reject legacy
  episodes from this native-frequency comparison. Native all-zero measurements
  are retained: zero is not padding.
- Restrict to the episode's first/last real `observation_ns`, excluding all
  pre/post-roll. Exclude recorder pause interiors where adjacent RGB observations
  differ by >150 ms, even when tactile continued through that pause.
- Split valid segments at resets, >50 ms read gaps, nonconsecutive retained row
  indices, or read-error/drop counter changes. Do not compute rates/derivatives
  across these boundaries. Segments need at least four recorded samples.
- Build regular 76/70/65/60/50 Hz integer-nanosecond grids. Repeat each comparison
  at ten phases from 0 to 0.9 of a grid period. For each grid time g, select the
  latest recorded `read_end_ns <= g`. The inspected native files have no later
  `available_ns`; the adapter's general rule additionally checks availability.
- No interpolation, averaging, deduplication, learned filtering or output stream
  replacement is performed. Repeated payloads and repeated selection of the
  same software-read row are counted separately.
- Age is **time since the selected software read completed**. It is not true
  hardware-acquisition age: the SDK exposes no hardware timestamp/counter.

### Native read intervals and changed-value rates

Exact payload equality means all 15 SDK components, or all 1,800 raw taxel
components, are unchanged. Apparent change rate is observed unequal consecutive
payloads divided by valid segment duration. It must not be called hardware
sampling frequency.

| Episode | Read interval median / p99 / max (ms) | SDK repeats | Taxel repeats | SDK changes/s | Taxel changes/s |
|---:|---:|---:|---:|---:|---:|
| 0 | 13.076 / 15.195 / 21.023 | 82.77% | 80.40% | 13.12 | 14.93 |
| 48 | 13.084 / 14.467 / 17.714 | 71.13% | 71.30% | 22.00 | 21.87 |
| 98 | 13.074 / 14.039 / 17.123 | 87.22% | 87.61% | 9.75 | 9.45 |
| 197 | 13.082 / 13.950 / 16.375 | 72.84% | 72.97% | 20.69 | 20.60 |
| 553 | 13.101 / 13.970 / 15.793 | 63.78% | 64.05% | 27.57 | 27.36 |
| 554 | 13.088 / 14.034 / 29.167 | 57.84% | 60.28% | 32.07 | 30.22 |
| 771 | 13.087 / 14.886 / 16.926 | 78.38% | 64.85% | 16.47 | 26.79 |

The pooled SDK payload-change rate is **18.68 changes/s**. Native median read
spacing is ~13.1 ms (~76.4 reads/s), but a changing payload does not establish a
fresh independent hardware acquisition. Conversely, an unchanged payload does
not establish that the sensor failed to update. Among consecutive nonzero SDK
vectors, repeats range from **7.09% to 61.76%** across these episodes, so repetition
is not entirely explained by no-contact zeros. Some reads change taxels while
leaving SDK aggregates unchanged. Prior review also found changing joint values
on consecutive reads; that shows rows are not wholesale copied packets, but
cannot prove tactile firmware refresh on every read.

**Conclusion about polling:** the data demonstrate ~76 Hz successful software
reads with slower, task-dependent force-payload changes. They do **not** identify
the tactile hardware update rate, or prove that polling faster than that rate
causes the repeats. Static/quantized values and an unknown firmware refresh
cadence remain competing explanations. There is no evidence for choosing a
specific presumed hardware rate such as 70 Hz.

### Causal grid results

Values below are means over ten grid phases. The full JSON gives phase ranges.
“Same read” means the identical source row was selected twice; “SDK/taxel
repeat” means exact values match even if the source rows differ.

| Grid | SDK repeats | Taxel repeats | Same read | Read age median / p99 (ms) | Mean held-output age (ms) |
|---:|---:|---:|---:|---:|---:|
| 76 Hz | 75.62% | 71.93% | 1.002% | 6.56 / 13.15 | 13.15 |
| 70 Hz | 74.82% | 71.02% | 0.100% | 6.56 / 13.15 | 13.71 |
| 65 Hz | 74.16% | 70.28% | 0.024% | 6.56 / 13.16 | 14.26 |
| 60 Hz | 73.42% | 69.42% | 0.012% | 6.56 / 13.15 | 14.91 |
| 50 Hz | 71.48% | 67.27% | 0.006% | 6.56 / 13.15 | 16.57 |

Read age at calculation instants averages ~6.57 ms at every rate. For a score
held until the next grid tick, its mean read age gains half a period: this is
the “held-output age” column, not another measurement of sensor latency. Higher
frequencies can therefore improve output freshness despite similar sample age
at the calculation instants. Occasional read-age maxima are ~20–23 ms;
phase-specific maxima are in the evidence.

### Peaks, transient proxies and derivatives

The diagnostic envelope is the raw-count norm of the recorded SDK xyz vector
per finger. It is not calibrated physical force. No axis is assumed to be a
surface normal or tangential direction.

Short peak definition: >=5 count prominence and <=50 ms half-prominence span
using native held-sample times. Only **three** such peaks were found. Retention
means a causal grid observation within the event span reaches >=90% of its
native amplitude. Offline neighbor information defines the reference event
for evaluation only; it never constructs a policy observation.

A separate rise-event proxy uses >=5 count-norm increase in one native
increment; **97** events were found. Retention requires >=90% of that rise in
a crossing grid interval within 50 ms. These are operational signal diagnostics,
not contact, slip or risk labels; they do not cover every component/directional
transient.

Derivatives use `norm(F[k]-F[k-1])/dt` per finger, in counts/s. We compare grid
RMS against native consecutive-read RMS over matched source index ranges. Total
variation retention measures how much vector-force path variation survives;
missed rises/falls can cancel when several reads are skipped.

| Grid | Short peaks >=90% | Rise proxy >=90% | Vector-force variation retained | Derivative RMS / native RMS |
|---:|---:|---:|---:|---:|
| 76 Hz | 100.00% | 99.79% | 99.71% | 1.0212 |
| 70 Hz | 100.00% | 99.69% | 98.29% | 0.9916 |
| 65 Hz | 93.33% | 99.28% | 96.76% | 0.9570 |
| 60 Hz | 93.33% | 99.07% | 95.39% | 0.9259 |
| 50 Hz | 90.00% | 98.56% | 92.54% | 0.8567 |

76 Hz total-variation retention ranges **99.44–99.90%** across phases; 70 Hz
**97.95–98.61%**; 50 Hz **91.71–93.28%**. At 65/60/50 Hz, the short-peak result
can fall to 2/3 for some phases. With only three short peaks, this estimate is
weak and should not be generalized to all contact events.

Grid finite differences slightly inflate RMS at 76 Hz (~2.1%) and attenuate it
at lower rates (50 Hz ~14.3% lower). A rate close to native RMS by coincidence
is not evidence of better risk estimation: it can still omit variation. For
future risk features, compute derivatives from actual consecutive native read
times, within valid segments, then expose those causal summaries at the desired
calculation times. Repeated reads have a legitimate zero value increment; do
not divide by “time since the last changed value” as if it were the read period.

### Frequency recommendation

**Keep native-event processing (~76.4 Hz), or use 76 Hz if a regular contact-risk
calculation grid is needed.** It preserves the most measured variation and
transients of the tested grids. Reducing to 70 Hz barely changes SDK repetition
(75.62→74.82%) while discarding more signal variation. Lower repetition alone
is not a quality objective. If future hardware profiling demonstrates a compute
constraint, 70 Hz is a measurable trade-off, not an automatically selected rate.

This recommendation is provisional for causal force-feature computation;
there is no validated contact-risk definition or physical acquisition-rate
measurement yet. No risk scalar, gate or risk label was implemented. The loader
continues to retain every native row; no rate from this experiment was imposed
on recordings or on the neural fast-tick scheduler.

## 2. Timestamp alignment and quality handling implemented

`taptic/timing.py` provides reusable quality/segment metadata; `taptic/data.py`
and `qwen_vla/taptic_dataset.py` expose it without resampling raw streams.

- Frame-to-sidecar selection uses `read_end_ns <= observation_ns` and
  `available_ns <= observation_ns`, never the potentially incorrect frame
  `xhand_sequence`. Legacy episode 297 was tested against frame state using
  timestamps: all 310 recorded SDK force vectors match exactly.
- Native repeated values remain recorded observations. Separate flags identify
  exact repeated SDK/taxel payloads; equality does not make them stale/invalid.
- Native read freshness defaults to **50 ms**; legacy nominal-clock freshness
  defaults to **150 ms**. Native segment gaps default to **50 ms**, legacy to
  **150 ms**. These are explicit engineering limits, not calibration or proven
  hardware-age limits. Native freshness/gap and legacy freshness have CLI options.
- Stale current raw values remain accessible for auditing; validity is false and
  the model's tactile input is neutral zero. Missing raw taxels remain invalid
  for all legacy episodes. Legacy 10 Hz rows are retained, synthetic interpolation
  excluded, and nominal clock provenance stays explicit.
- Resets, native sequence discontinuities and read-error/drop counter changes
  start new segments. Legacy sequence increments after synthetic-row removal
  are not misclassified as native sample loss.
- `history_valid` restricts feature history to the trailing eligible segment.
  `derivative_valid` requires a consecutive previous returned sample in that
  segment; it is false on the first row and across breaks.
- A known recorder pause before the current tick leaves the raw interval intact
  but masks its policy history/derivatives. The current latest-read observation
  remains independently checked for freshness.
- At each re-cut episode's first frame, only one latest causal initial sample
  is returned; overlapping pre-roll is not treated as policy history. Subsequent
  returned intervals contain samples available strictly after the prior frame.

New per-current-observation fields in `batch['tactile_quality']`, shape `[B]`:
`source_index`, `age_ns`, `availability_age_ns`, `fresh`, `segment_id`,
`break_before`, `recording_gap_before_tick`.

New padded stream fields, shape `[B,N]`: `source_index`, `age_ns`,
`availability_age_ns`, `fresh_at_anchor`, `history_valid`, `derivative_valid`,
`break_before`, `segment_id`, `force_unchanged_from_previous_read`,
`taxels_unchanged_from_previous_read`. Existing raw `sdk_force [B,N,5,3]`,
`taxels [B,N,5,120,3]`, timestamps/counters and padding masks remain intact.
History samples can be old and still be valid recorded history; the separate
`fresh_at_anchor` mask must not be interpreted as “all history must be fresh”.

All ages are based on SDK read/availability times; true firmware acquisition
age remains unknown. Native and nominal legacy clocks must remain distinct.
No compact contact features or risk conditioning were introduced into forward.

## 3. Flow-time precision fix and baseline compatibility

New `qwen_vla/flow_matching.py` defines the integer-indexed FP32 grid:

```
time(step) = float32((total_steps - step) / total_steps)
Action, six Euler steps: 1.0 -> 0.9 -> 0.8 -> 0.7 -> 0.6 -> 0.5 -> 0.4
Tactile, four steps:     0.4 -> 0.3 -> 0.2 -> 0.1 -> 0.0
```

Decimal values are represented to normal FP32 tolerance; no BF16 time recurrence
is used. Euler state accumulation and Δτ arithmetic are FP32 internally. Actions
are cast to the existing network input dtype for embeddings, and returned chunks
retain the original BF16 API dtype. The existing TimestepEmbedder computes its
sinusoidal features in FP32 and casts its MLP input to the parameter dtype;
no embedding architecture or learned weights were changed.

`flow_time_precision='fp32'` is the revised default for partial, full-action and
tactile continuation paths. `legacy_bf16` retains the old recurrent arithmetic
for controlled comparisons. The original Action Expert training loss is unchanged.
Tactile training retains the original Beta(1.5,1) draw transformed as
`(.999*Beta + .001)*tau_split`, now scaled in FP32 in the revised mode. The
velocity target remains **epsilon_r - norm_actions**. TAPTIC stores the `time_r`
draw in FP32, or BF16 when explicitly requesting legacy arithmetic.

The Action cache is refreshed at the selected endpoint using the same returned
BF16 action state. Metadata stores its split, total/action step counts, precision,
length and action-context length. Tactile calls validate this metadata and reject
mismatched schedules or mutated snapshots. Training and continuation clone the
cache, so calls do not append tactile tokens to the caller's handover snapshot.
This is one coherent handover cache, **not** a cache ladder.

### Measured old-vs-revised comparison

The test loads original methods read-only from commit `b068f70` and runs them
with the same tiny model, inputs and random **untrained** nonzero Action head.
No production checkpoint is loaded or modified. It checks weights remain
identical before/after the comparison and constructor AST is unchanged.

| Result | Measured value |
|---|---:|
| Original τ after six steps | 0.39453125 |
| Revised returned τ | 0.4 |
| New legacy path vs original partial state and KV tensors | bit-for-bit equal |
| New legacy vs original final tactile continuation | bit-for-bit equal |
| Revised vs original partial state, max abs / RMS | 0.03125 / 0.00816049 |
| Revised vs original final chunk, max abs / RMS | 0.046875 / 0.01011330 |

Differences are normalized action units on a random small model, not physical
error or policy performance. They reflect corrected time embeddings **and**
FP32 integration rather than BF16 accumulation. Correcting the baseline is not
claimed to be bitwise equivalent; the explicit legacy path is the replay control.
Network constructor, parameter shapes and MoT backbone are unchanged. The full
Action-only solver was also executed: its legacy output equals the original
bit-for-bit, and its revised output has the expected shape and finite values.

## 4. Multi-split training implemented

| τ_split | Action Euler steps | Tactile Euler steps | Nominal Δτ |
|---:|---:|---:|---:|
| 0.2 | 8 | 2 | -0.1 |
| 0.4 | 6 | 4 | -0.1 |
| 0.6 | 4 | 6 | -0.1 |

`--cascaded_split_mode multi --cascaded_splits 0.2 0.4 0.6` selects one level
uniformly per batch. Rank 0 samples and broadcasts the choice when distributed
training is initialized. All examples/ranks in that batch use the same schedule.
There is no risk-dependent/adaptive gate. Fixed mode remains the default:
`--cascaded_split_mode fixed --cascaded_split_step 6`.

Both training and validation build a fresh Action cache at that selected
handover and scale tactile training times by the **same** split. The action
loss still covers the original full flow-matching range; tactile velocity target,
noise and loss formula are preserved. The trainer logs selected split/step
counts and saves precision/split/freshness settings in checkpoint metadata.

The endpoint cache refresh costs one extra Action forward but is not another
Euler step. Thus 10 Euler updates do not imply only 10 model calls. A physical
fast tick starts a fresh tactile continuation from the same cached handover;
it does not consume the next element of one persistent diffusion timeline.
Sensor sampling Hz and unitless Euler Δτ are not interchangeable.

## 5. Executed tests and numerical results

**20 passed in 15.95s**, using the updated `scripts/smoke_taptic.sh`. See
[full output](test_results.txt) and [parsed numerical evidence](test_evidence.json).

The multi-split tests use a real mixed native/legacy TAPTIC batch `[2,16,21]`,
with original Qwen/MoT classes at text H64/depth 1 and vision H32/depth 1, random
weights and CPU BF16 autocast. They perform three tiny optimizer updates per
split on transient random test models; no learned checkpoint is saved.

| Split | Action loss | Tactile loss | Handover KV length | Action-context length |
|---:|---:|---:|---:|---:|
| 0.2 | 1.38733327 | 1.55884647 | 180 | 89 |
| 0.4 | 1.38732862 | 1.55885828 | 180 | 89 |
| 0.6 | 1.38732064 | 1.55890441 | 180 | 89 |

These are finite normalized flow losses after the tiny warm-up, not trained
policy performance. The mixed batch's cache contains 91 latent and 89 action
context tokens; continuation/training leaves that 180-token snapshot intact.
All returned velocity/action tensors are `[2,16,21]`. The same-batch validation
Action loss was 1.44330609 at all three levels; tactile validation losses were
2.48865628 / 2.50960970 / 2.48085237 for 0.2 / 0.4 / 0.6.

Executed coverage:

- Original ten real-data/adapter checks, including normalization/inverse action
  conversion, independent nine-task Panda FK, native/legacy collation and the
  original 62D loader's synthetic-input regression.
- Integer-indexed FP32 time sequence and all three step allocations; fixed mode
  and multi-mode selection; unchanged original tactile velocity target and Beta
  distribution transformation.
- Actual two-process **Gloo CPU** test confirming shared split choices across
  ranks. This is not a distributed model-training test.
- Timestamp alignment on eight native/legacy episode IDs; exact legacy SDK/state
  equality by timestamp; no synthetic 76 Hz legacy rows; first-frame pre-roll
  bounds and actual recorder-pause history masking.
- Unit tests for repeated rows retained, stale observations invalidated, reset/
  gap/error segments, future payload changes not affecting past observations,
  and history/derivative masks.
- For all splits: mixed real-data forwards, finite Action/Tactile losses,
  correct time-embedding sequences, coherent KV lengths, immutable cache after
  training/inference, deterministic repeated continuation and mismatched schedule
  rejection. The native/legacy stream sample counts do not set Euler step counts.
- Backward checks: Tactile loss reaches tactile/shared observation/action/time
  embedders after warm-up, without nonzero Action-expert or visual gradients;
  Action loss reaches Action modules without nonzero Tactile-expert gradients.
  Visual encoder is frozen. The original tactile normalization zero-init is
  retained; deeper tactile-input gradients are tested on the third backward
  pass, after two tiny warm-up updates.
- The **actual trainer validation function** executes at all three splits on
  CPU, with only its CUDA-device query patched to CPU; one split per batch and
  finite returned losses are verified. Full Accelerate training is not claimed.
- Original-versus-legacy bitwise solver regression, revised numerical difference
  measurement, unchanged constructor and unchanged weights during comparison.

Early test failures were corrected transparently: unused routed modules can
receive allocated **zero** gradients rather than `grad=None`; tests now check
nonzero-gradient isolation. A long-lived CPU autocast context cached stale weight
casts across toy optimizer steps, so the multi-step CPU test disables autocast's
weight cache. This was a test-harness issue, not a network/initialization change.
A first check for deep tactile-input gradients was also too early for the
original zero-norm initialization; the tests retain that initialization and
verify propagation after warm-up rather than altering it to satisfy the test.
Final suite has no failures or skips in the executed configured environment.

## 6. Exact file changes in this task

| File | Change |
|---|---|
| `scripts/analyze_taptic_frequency.py` (new) | Read-only native analysis, causal grids, ten phases, repeat/age/peak/derivative diagnostics and source hashes. |
| `taptic/timing.py` (new) | Timestamp quality, freshness, segment, history and derivative-validity helpers. |
| `taptic/data.py` | Integrates timestamp quality with current and interval selection; retains native rows and raw values. |
| `qwen_vla/taptic_dataset.py` | Exposes metadata/masks, neutral stale model input, recorder-pause history masking, FP32 tactile training draw and configurable split validation. |
| `qwen_vla/flow_matching.py` (new) | Integer FP32 time grid, batch-wide fixed/multi handover selection, distributed broadcast and original tactile pair/target construction. |
| `qwen_vla/modeling_vla.py` | FP32 solver arithmetic/legacy option, embedding dtype boundary, cache schedule metadata/validation and immutable snapshots; no constructor/network changes. |
| `scripts/midtrain.py` | Training/validation schedule selection and coupling, precision/config options, logging and checkpoint metadata. Original action loss unchanged. |
| `tests/taptic/test_adapter.py` | Tightens fixed endpoint tolerance; updates allowed split guard and unchanged-source check now that solver methods intentionally changed. |
| `tests/taptic/test_multisplit.py` (new) | New numerical/data/distributed-selection/forward/backward/solver-regression/trainer-validation tests. |
| `scripts/smoke_taptic.sh` | Runs both suites, 20 tests. |
| `docs/taptic_multisplit/` (new) | This report, frequency JSON, test output/numerical evidence and unchanged-source hash checks. |

The prior `docs/taptic_review/`, `scripts/review_taptic_sidecars.py` and
`scripts/review_taptic_smoke.py` were pre-existing uncommitted work and remain
untouched. Original dataset loaders, shared official utilities, diffusion
embedding modules and `modeling_qwen3vl_mot.py` remain unmodified.

[Hash verification](source_unchanged.json) confirms all **20** inspected source
files still match their pre-change/reference hashes, including native/legacy
sidecars, frame files, main data and metadata. No replacement tactile export,
checkpoint file or force/risk labels were produced.

## 7. Exact commands

From the repository root:

```bash
export TAPTIC_PYTHON=/home/mfincato-iit.local/trex-venv/bin/python
export TAPTIC_ROOT=/home/mfincato-iit.local/midtrain_merge/derived/taptic_trex_style_v2
export TAPTIC_PROCESSOR=/home/mfincato-iit.local/midtrain_merge/derived/tiny_qwen3vl_test
export TAPTIC_PANDA_URDF=/home/mfincato-iit.local/lerobot_plugins/lerobot_robot_panda/lerobot_robot_panda/urdf/panda.urdf
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''

# Reproduce the read-only frequency analysis.
"$TAPTIC_PYTHON" scripts/analyze_taptic_frequency.py \
  --root "$TAPTIC_ROOT" --out /tmp/taptic_frequency_analysis.json

# Execute all 20 tests, including real data, gradients and two CPU ranks.
bash scripts/smoke_taptic.sh -s

# Confirm training CLI options without starting training.
"$TAPTIC_PYTHON" scripts/midtrain.py --help
```

Additional configuration for a future authorized run (not launched):

```text
--data_format taptic --lerobot_root <verified-root>
--taptic_stats docs/taptic_train_stats.json
--action_dim 21 --action_chunk 16
--use_tactile_vec 1 --use_tactile_deform 0
--use_tactile_code 0 --use_tactile_vqvae 0 --tactile_delay_offsets 0
--cascaded_total_steps 10 --flow_time_precision fp32
--cascaded_split_mode multi --cascaded_splits 0.2 0.4 0.6
```

For the corrected fixed baseline use `--cascaded_split_mode fixed
--cascaded_split_step 6 --flow_time_precision fp32`; for the historical solver
control replace precision with `legacy_bf16`. Existing model/resource/output
arguments remain required. Dataset sampling is not changed by these flags.

## 8. Remaining limitations

- No true sensor acquisition timestamp/update counter: sensor firmware refresh
  rate, physical acquisition age and whether any repeated payload is cached
  remain unknown. Native read-age freshness is not hardware freshness proof.
- Seven native episodes, only three defined short peaks and 97 defined rise
  proxies; no force calibration, normal/tangent frame proof, slip signal or risk
  labels. The recommended risk-computation frequency needs broader event and
  deployment validation before safety/performance claims.
- No actual risk scalar, gate, risk head, contact-aware Action Expert, adaptive
  selection policy, cache ladder or early replanning. Multi-split choice is
  uniform per batch. The Action Expert still has its original inputs/loss.
- The training dataset remains anchored at recorded 10 Hz frames. Native streams
  and masks are available, but a full independent physical fast-tick replay/
  scheduler, fast-tick target definition and real-time latency measurements are
  not implemented by this phase. Stored 10 Hz full robot targets do not become
  76 Hz labels. No higher-rate Panda state is invented.
- The local workstation has an 8 GB RTX 4060 Laptop GPU, but the configured real
  T-Rex paths contain checkpoint metadata, not actual pretrained weights. No
  full-size pretrained/GPU smoke test was attempted; no multi-GB download or
  remote heavy job was launched. CPU tests use random reduced models only.
- Full Accelerate/DeepSpeed training, distributed model/backward synchronization,
  full-size pretrained head migration/loading, checkpoint save/resume round trip,
  exact resumed split RNG trajectory, and FLARE prediction/loss remain unvalidated.
  The Gloo test covers split synchronization only. Split draws use the seeded
  PyTorch RNG; checkpoint RNG restoration was not added to the original trainer.
- The standalone bimanual inference server was not extended into a Panda/XHand
  deployment wrapper; its legacy-precision CLI and real robot integration remain
  outside this phase. Solver comparison uses the explicit model API precision
  argument and trainer CLI.
- Tests ran in Python 3.12 / torch 2.10 / transformers 4.57.1. The repository's
  pinned Python 3.10 / torch 2.6 environment and CUDA numerical behavior still
  require validation before a production training run.
