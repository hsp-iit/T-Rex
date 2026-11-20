# TAPTIC: risk-conditioned adaptive T-Rex (OURS): implementation report

Branch **`mfincato/trex-ours`** (first published as `mfincato/taptic-ours`), created from **`05c932853ac5984ac7128561de2583c34b568262`**
(the commit used to launch the vanilla TAPTIC midtrain), then fast-forwarded to `431adf4` (the
vanilla branch's single-threaded AV1 decoding fix in `taptic/video.py`; frames are bit-exact), so
OURS = vanilla tip + the OURS commits. Development happened in a separate
worktree (`/home/mfincato-iit.local/T-Rex-ours`). The original branch `mfincato/risk_gated` (renamed `mfincato/trex-vanilla`; its tip is `431adf4`), its
working tree, the vanilla launcher/configuration and every experiment output are untouched.

**Status.** The whole architecture is implemented as one coherent code path and validated on CPU
with reduced random models and real TAPTIC data: **69 tests passed, 0 failed, 0 skipped**
([test_results.txt](test_results.txt)). Real-data sweeps found **0 alignment violations in
31,292 delayed samples**. No full training was launched, no large checkpoint was downloaded,
no hardware was commanded, and **no claim about real-model performance is made**.

---

## 1. What OURS is

| Component | Baseline T-Rex (vanilla) | OURS |
|---|---|---|
| MoT backbone, causal rule, expert Q/K/V/MLPs, FLARE, flow targets | ✓ | unchanged |
| Action representation (21-D, chunk 16, delta-base, rot6d, 12 XHand q) | ✓ | unchanged |
| Tactile input: 5 VQ codes (16 native SDK-force samples, ~210 ms) + 5 force tokens | ✓ | unchanged, now read at the **fast tick F** |
| Contact-risk scalar r = clip(0.25 M + 0.5 R100 + 0.25 M V) | none | causal, train-normalized, explicit validity; R100 = vector force variation |
| Risk token, Action stream (r_S at slow tick S) | none | learned `risk_embedder_action` |
| Risk token, Tactile stream (r_F at fast tick F) | none | learned `risk_embedder_tactile` |
| Training split | fixed τ = 0.4 | uniform τ ∈ {0.2, 0.4, 0.6} per batch, risk-independent |
| Tactile timing in training | F = S | F = S + δ, δ ∈ {0, 50, 100, 150} ms; elapsed slots masked |
| Inference | one partial Action run + Tactile | risk-conditioned cache ladder + non-learned gate on r_F |

```mermaid
flowchart LR
  subgraph SENS[Native sensor stream ~76 Hz, CLOCK_MONOTONIC]
    F[SDK force 5x3 counts]
    Q[measured XHand joints 12, same read]
  end
  subgraph FEAT[taptic/risk.py: shared offline/online core]
    M[M: max_f norm |F_f|]
    R[R100: vector force variation, 100 ms]
    V[V: causal 100 ms secant of q]
    RS[r_S at S]
    RF[r_F at F]
  end
  F --> M & R
  Q --> V
  M & R & V --> RS & RF
  subgraph SLOW[Slow tick S, 10 Hz]
    CTX[images + caption at S -> latent + FLARE]
    AE[Action Expert x 8 Euler + 1 refresh]
    LAD[(immutable ladder<br/>tau 0.6 / 0.4 / 0.2<br/>x_tau + KV + positions)]
  end
  RS -->|risk_embedder_action| AE
  CTX --> AE --> LAD
  subgraph FAST[Fast tick F >= S]
    VQ[VQ-VAE: 16-sample history ending at latest read <= F]
    G[RiskGate: r_F -> tau]
    TE[Tactile Expert x 10*tau steps]
  end
  F --> VQ
  RF --> G
  RF -->|risk_embedder_tactile| TE
  VQ --> TE
  G -->|select snapshot, read only| LAD
  LAD --> TE --> OUT[action chunk 16x21]
```

### Token layout

```text
Action-routed  : [ latent (text+slow img+FLARE) | fast img | (state) | risk_S | t | x_t (16) ]
Tactile-routed : [ cached latent+action KV ] + [ VQ codes (5) | force (5) | risk_F | tau | x_tau (16) ]
```

Each risk token sits immediately **before its stream's flow-time token**, so the chunk tokens stay last:
prediction slices (`hidden[:, -16:]`), routing indices and M-RoPE extension work unchanged. The training
action slice becomes `L_latent + n_fast + n_state + n_risk + 1`. Position ids and cache lengths grow by
`n_risk` (1 or 0) automatically through `n_action_in_cache` and `_extend_position_ids`. Causality is
preserved: nothing before a risk token can see it. The tests check that latent/fast hidden states and KV
are **bitwise** independent of r_S and that the tactile observation tokens are independent of r_F.

**Risk embedder** (`qwen_vla/risk_tokens.py`), one per expert:

```text
phi(r, v) = [ sin(k*pi*r)*v , cos(k*pi*r)*v ]_{k=1..8} , r*v , v , 1-v      (v = validity bit)
token     = Linear(19 -> H) -> SiLU -> Linear(H -> H)
```

A missing measurement (v = 0) gets its own learned output through the `1-v` channel. It is never
encoded as genuine zero risk (r = 0, v = 1). Valid non-finite or out-of-range inputs raise an error.
A risk-conditioned model *requires* `(risk, risk_valid)` on every path and a risk-free model *rejects*
them, so a configuration mix-up cannot be silently ignored.

---

## 2. Contact-risk scalar

### Data facts verified first

* **Measured joint stream.** The native sidecar (`tactile_raw/episode_*_right.parquet`) has a
  `joints` column written by the same EtherCAT read as the force (`XHandDirectDriver.read_measured_state`).
  On native recordings it is a genuine ~76 Hz measured stream: the median read interval is 13.08 ms,
  **0 % of consecutive joint vectors repeat**, and frame samples equal `observation.state[7:19]`
  bit-exactly. Legacy recordings have only the 10 Hz frame samples (synthetic interpolated rows are
  excluded). This is tested in `test_real_sidecar_joint_stream_is_measured_native_rate`.
* **Encoder noise.** The encoder quantization is ≈0.5 mrad. At rest, the joint-speed norm from a 13 ms
  difference has a median of 0.41–0.52 rad/s, which is pure quantization noise. A 100 ms causal secant
  lowers this to 0.06–0.09 rad/s and leaves the motion tail (p99 ≈ 1.4–3 rad/s) almost unchanged.
  Hence the default velocity span `Wv = 100 ms`, which is configurable.

### Equations (`taptic/risk.py`)

At a physical tick t (int ns), let i\* be the latest usable sample with `read_ns ≤ t` and
`available_ns ≤ t`. Let m[i,f] = ‖F[i,f,:]‖₂ in SDK counts, with no newton calibration.

```text
M      = max_f clip( m[i*,f] / sM[f], 0, 1 )
u_f(t) = Σ_{i: t-W < read_i ≤ t, i ≤ i*, (i-1,i) same segment} ‖F[i,f,:] - F[i-1,f,:]‖₂ / W     [counts/s]
R100   = max_f clip( u_f(t) / sR[f], 0, 1 )                    W = 100 ms
qdot   = (q[i*] - q[k]) / (read_i* - read_k),   k = latest same-segment read ≤ read_i* - Wv   [rad/s]
V      = clip( ||qdot||_2 / sV, 0, 1 )                          Wv = 100 ms
r      = clip( 0.25 M + 0.50 R100 + 0.25 M V , 0, 1 )
```

**R100 is the vector force variation** (`rate_mode='vector'`, risk definition version 2, decided
after the first review). It counts force increases, **decreases** (release, loss of grip, the pipette
squeeze-release cycle) and **direction changes at constant |F|** (shear/slip candidates). By the
triangle inequality it is ≥ the |F| variation; on the training data the vector variation is 35–50 %
larger than the |F| variation, and decreases total 0.94–1.00× the increases in every task. The
previous positive-only definition, Σ max(m[i,f] − m[i−1,f], 0) / W, stays available as
`--risk_rate_mode positive` (statistics in [risk_stats_positive.json](risk_stats_positive.json)). The
positive (R_pos) and negative (R_neg) |F| variations are always computed with their own scales and
logged. Statistics files store `rate_mode` and are refused when it differs from the run's configuration;
version-1 files (positive-only R) are refused outright.

The coefficients are fixed experimental hyperparameters (`--risk_coefficients`). They are not learned
and r is not a probability. M, R100 and V, their validity flags, the read age, the joint age, the secant
span and the per-finger magnitudes/rates are all kept, logged (`risk/S_M_mean`, …) and exposed for
ablations.

**Validity** is explicit, and an invalid value is never treated as zero risk:
`M_valid` requires a usable sample younger than 50 ms. `R_valid` additionally requires the sample's
segment to cover the whole window (t−W, t]. `V_valid` requires a fresh joint sample and a same-segment
secant base. `risk_valid = M_valid ∧ R_valid ∧ V_valid`. Segments split at read gaps > 50 ms,
reset/error/drop counter changes and sequence jumps (the existing `taptic.timing` rules). Pre-roll,
post-episode and recorder-pause samples are excluded (the existing `build_force_track`). Legacy episodes
are always risk-invalid because they only have 10 Hz data on a nominal clock. A `state10hz` joint source
(LeRobot 10 Hz state, age exposed, no interpolation) is available as a fallback/ablation; it is not the
default, because the native stream exists.

### Train-only normalization ([risk_stats.json](risk_stats.json))

`python -m taptic.risk fit` evaluates the features at **every usable native read of the 512 native
training episodes** (925,343 ticks; the split comes from `docs/taptic_train_stats.json` and dataset
identity `1cb54e3d…` is verified):

| Scale (q99) | thumb | index | middle | ring | little |
|---|---:|---:|---:|---:|---:|
| sM magnitude [counts] | 154.79 | 88.69 | 51.48 | 49.77 | 63.40 |
| sR **vector** R100 (active) [counts/s] | 368.8 | 483.8 | 541.7 | 463.8 | 710.9 |
| sR positive R100 (R_pos, logged) [counts/s] | 220.22 | 314.02 | 253.97 | 282.22 | 430.18 |
| sR negative R100 (R_neg, logged) [counts/s] | 247.3 | 317.7 | 253.6 | 286.4 | 426.2 |

Rate scales are the q99 of the non-zero window values of each kind. `sV = 2.658 rad/s` (q99 of ‖q̇‖).
The magnitude and positive-rate scales reproduce the earlier, independently written offline analysis
(`feat/test1`: 154.79/88.69/51.48/49.81/63.40 and 220.23/314.04/253.97/282.32/430.29) to within 0.1 %,
and refitting with `--rate_mode positive` reproduces the previous statistics file exactly. Of all native training reads, 99.34 % are risk-valid (the remainder are the first 100 ms
of a segment). The training distribution of r is zero-inflated:
p50 0.0235 · p75 0.127 · p90 0.267 · p95 0.336 · p99 0.602 (positive-only R: p50 0.0094, p99 0.576).
With vector R the median of R itself is 0.02 instead of 0, because small decreases and direction
changes now count. `RiskStats.check_source` refuses statistics
from another dataset revision or from non-training/non-native episodes.

---

## 3. Training algorithm

```text
per batch:
  h  <- uniform{0.2, 0.4, 0.6} (rank-0 draw, broadcast), independent of risk     [--cascaded_split_mode multi]
  per sample (frame t, slow tick S = observation_ns[t]):
    δ  ~ U{0,50,100,150} ms (train), δ = delays[idx % 4] (val);   F = S + δ
    Action conditioning: images + caption at S, r_S from samples available by S
    Tactile conditioning: force + VQ history ending at the latest read <= F, r_F by F
  L_A : t ~ 0.999·Beta(1.5,1)+0.001, x_t = tε + (1-t)A;  v = ActionExpert([ctx | fast | r_S | t | x_t])
        L_A = MSE(v, ε − A) over all 16 slots                                   (unchanged target)
  cache (no grad): ε'; 10·(1−τ) Action Euler steps with r_S, Δτ = −0.1, FP32 integer grid
        flow_time(i) = (10−i)/10, then a refresh forward at (x_τ, τ)  →  KV(τ_split)
  L_T : τ = τ_split·u, x_τ = (1−τ)A + τε_r;  v_T = TactileExpert([KV | codes(F) | force(F) | r_F | τ | x_τ])
        L_T = Σ_{b,k} m_bk ‖v_T − (ε_r − A)‖²_k / Σ m_bk
        m_bk = tactile_valid_b(F) ∧ times[t+k] ≥ F                     (slot not yet elapsed at F)
  tactile dropout (p = 0.1): zero force/history values AND mark r_F missing (risk_valid = False)
  L = L_A + λ_T L_T + λ_F L_FLARE
```

| τ_split | Action updates | Tactile updates | No-grad cache forwards |
|---:|---:|---:|---:|
| 0.2 | 8 | 2 | 9 |
| 0.4 | 6 | 4 | 7 |
| 0.6 | 4 | 6 | 5 |

The mean no-grad cache cost under uniform multi-split (7 forwards) equals the vanilla cost.
`--cascaded_split_mode fixed --cascaded_split_step 6` keeps the fixed τ = 0.4 mode.

### How the masks interact with the noisy chunk

* **Action loss.** It does not depend on F. All 16 slots are noised (`x_t`) and supervised against their
  recorded targets, conditioned on context at S and r_S.
* **Tactile loss.** The noisy chunk `x_τ` is built for **all 16 slots** from their recorded targets
  (standard flow-matching interpolant), so the tactile sequence has exactly the inference layout and
  every slot token is attended as context. The velocity error of a slot counts only if the slot is
  **not yet elapsed at F** (`times[t+k] ≥ F`, recorded 10 Hz timestamps) and the sample has a valid
  native observation at F. Elapsed slots give no direct gradient.
* **Targets are never shifted or interpolated.** The chunk stays anchored at the measured pose at S
  (delta-base frame).
* With δ = 0, all slots are supervised. With δ = 50 ms, slot 0 is masked. With δ = 100 ms, slot 1 is also
  masked whenever the real frame clock put `times[t+1] < S + 100 ms` (**28.1 %** of real samples). With
  δ = 150 ms, slots 0 and 1 are masked. An all-invalid batch still yields a connected zero loss.
* `tactile_slot_mask = None` (delays [0]) uses the original loss formula, so it is bitwise identical.

**Rejection of invalid alignment.** F must lie inside the anchor's continuous 16-frame window (anchors
already exclude windows with recorder pauses > 150 ms, end padding and cross-episode spans). Samples
with stale tactile at F (> 50 ms), excluded reads, or no supervisable slot have `tactile_loss_mask = False`.
They still train the Action Expert and FLARE. Legacy samples are masked as before.

---

## 4. Cache ladder and inference

**Ladder** (`qwen_vla/cache_ladder.py`, `forward_flow_action_ladder`). The earlier `feat/test1` ladder
was reviewed and adapted rather than copied. Capture points and the BF16-API/FP32-accumulator
convention are the same; the new parts are the risk-token layout, r_S/slow-tick provenance, layout
checks, and byte-exact fingerprints. One Action trajectory per slow tick is conditioned on r_S:

* At i = 4 (τ = 0.6) and i = 6 (τ = 0.4), the regular forward already computes KV at (x_τ, τ). It is
  **cloned before the Euler update**, so no extra forward is needed.
* After the 8th update, one refresh forward at (x_0.2, 0.2) is captured. That makes 8 updates + 1 refresh
  = 9 forwards, versus 5 + 7 + 9 = 21 for three independent runs.
* Each snapshot owns x_τ (BF16 API boundary) plus the FP32 accumulator, the complete per-layer KV,
  latent/action position ids, token counts (`latent, fast, state, risk, time, chunk`) and the schedule
  metadata (`n_risk`, r_S, `slow_tick_ns`, FP32 time grid, step indices).
* Every snapshot is bitwise equal to the cache/state of a single-split partial run with the same noise
  and r_S. Continuing from the ladder is bitwise equal to partial + continue.
* Fast ticks only read a snapshot; the continuation clones before appending tactile KV. Fingerprints
  stay identical after repeated fast ticks at every split with changing r_F, and r_S is never rewritten.

**Fast tick** (`taptic/ours_policy.py`, `RiskAdaptivePolicy`): (1) the extractor reads the latest valid
native sample available by F; (2) it updates the 16-sample VQ history and r_F, using the same code as
training; (3) `RiskGate` selects τ; (4) the immutable snapshot is fetched; (5) the Tactile Expert runs
10·τ steps with codes, force tokens and r_F; (6) the 16×21 chunk is returned (normalized, and
denormalized when action stats are given) together with the gate decision, r_S, r_F, M/R/V, ages and
the first non-elapsed slot. Missing tactile gives zero inputs (the trained dropout condition), r_F is
marked missing and the gate falls back to 0.4.

**Gate** (`taptic/risk_gate.py`). It is monotone and non-learned:
`r_F < lo → 0.2`, `lo ≤ r_F < hi → 0.4`, `r_F ≥ hi → 0.6`. More risk hands more of the flow to the
Tactile Expert. Thresholds are initialized from **training** quantiles q50/q85 of r:
`lo = 0.0235`, `hi = 0.2011` (positive-only R: 0.0094 / 0.1675), floored at 1e-3 so exactly-zero risk
maps to 0.2. It supports:

* fixed-split mode (any of the three levels);
* additive hysteresis;
* minimum dwell time;
* immediate missing-data fallback to 0.4 (overrides dwell; recovery is immediate);
* per-tick logs to CSV/JSONL.

Because r is zero-inflated, hysteresis must stay below `lo`. Otherwise τ = 0.2 becomes unreachable;
this was found in the replay and is now a config error. **The mapping is an initialization, not an
optimized or safety-calibrated policy.**

---

## 5. Experimental configurations and checkpoint contract

| # | Configuration | Train variant | Inference |
|---|---|---|---|
| 1 | Vanilla T-Rex | `vanilla`: fixed τ = 0.4, no risk tokens, F = S | `GateConfig(mode='fixed', fixed_split=0.4)` |
| 2 | Uniform multi-split | `multisplit`: τ ∈ {.2,.4,.6}, delays 0–150 ms, no risk tokens | fixed gate (any τ) |
| 3 | Multi-split + external risk gate | same checkpoint as 2 | `GateConfig.from_risk_stats(...)`, no risk inputs to the model |
| 4 | **OURS** | `ours`: 2 + risk tokens r_S/r_F | adaptive gate + risk tokens |

`scripts/midtrain_taptic_ours.sh VARIANT=…` keeps the data revision/split, VQ-VAE, initialization,
optimizer, epochs, batch, seed and FLARE identical across variants. A test parses each launcher with
the real `midtrain.py` parser. **`VARIANT=vanilla` resolves to exactly the flags of the launched
`scripts/midtrain_taptic.sh`**, and every non-variant flag is equal across all three variants.
`--variant` is a guard: it refuses flags inconsistent with the named variant.

**`training_args.json`** now records:

* `architecture` (`model.architecture_contract()`: risk tokens, version, frequencies, stream layouts,
  dims, VQ/FLARE flags);
* `variant`;
* `tactile_delay_ms`;
* `risk_stats` with its SHA-256 and `risk_definition` (coefficients, windows, joint source, stale limit);
* `resume_load_report`.

**Loading** (`qwen_vla/contract.py`, used by the trainer's resume):

* a checkpoint with risk parameters loaded into a risk-free model is an **error**, never silently dropped;
* a risk-token model resumed from a risk-free checkpoint loads every shared weight bitwise, and the
  **new parameters are listed in the log** (they keep their Xavier/zero-bias init);
* a risk-parameter shape mismatch, or a declared-but-absent risk block, is an error;
* other shape mismatches/unexpected keys keep the pre-existing behaviour (skip, as for the pretrain
  tactile MLP), but every key is listed;
* pre-contract checkpoints are read as risk-free.

The baseline weight schema is unchanged: the meta-device `state_dict` of the risk-free constructor
equals the base commit's, and OURS adds exactly `risk_embedder_action.*` and `risk_embedder_tactile.*`.

---

## 6. Code modifications

| File | Change |
|---|---|
| `taptic/risk.py` (new) | `RiskConfig`, `RiskStats` (+ provenance checks), vectorized causal `risk_features`, train-only `fit_risk_stats` (+ gate quantiles, per-task occupancy), `OnlineRiskExtractor` (same core, availability-ordered ingest, late/duplicate handling, bounded buffer, VQ history), CLI `python -m taptic.risk fit` |
| `taptic/risk_gate.py` (new) | `GateConfig`, `RiskGate` (fixed/adaptive, hysteresis, dwell, fallback, logs) |
| `taptic/ours_policy.py` (new) | `RiskAdaptivePolicy`: slow tick → ladder(r_S); fast tick → extractor → gate → snapshot → Tactile Expert |
| `taptic/force_history.py` | the force track also carries the measured `joints` from the same rows (unchanged otherwise) |
| `qwen_vla/risk_tokens.py` (new) | `RiskEmbedder` |
| `qwen_vla/cache_ladder.py` (new) | `ActionHandoverSnapshot`, `ActionCacheLadder`, `handover_for_split` |
| `qwen_vla/contract.py` (new) | architecture extraction/checks, `load_with_contract`, `validate_variant` |
| `qwen_vla/modeling_vla.py` | `use_risk_tokens`; `risk_token()`; `architecture_contract()`; risk token in `forward_flow_action_full`, `forward_flow_action_partial` (→ shared `_action_trajectory`), `tactile_flow_continue`, `tactile_flow_train_step`; new `forward_flow_action_ladder`, `tactile_flow_continue_from_ladder`; `n_risk` in handover metadata and checks |
| `qwen_vla/taptic_dataset.py` | `--tactile_delay_ms`: S/F per sample, VQ history + force at F, `tactile_slot_mask`; r_S / r_F (+ M, R, V, validity, ages) in the batch; config guards |
| `scripts/midtrain.py` | `cascaded_losses()` is shared by the training loop and `run_validation` (same statement/RNG order; your study comments kept); r_S in the Action forward and cache, r_F in the tactile step, r_F missing under tactile dropout; slot-masked tactile loss; contract-checked resume; new flags; risk/delay logging; checkpoint metadata |
| `scripts/midtrain_taptic_ours.sh` (new) | variant launcher |
| `scripts/validate_taptic_ours.py` (new) | real-data alignment sweep, gate replay, model replay |
| `scripts/smoke_taptic.sh` | runs the new suites too |
| `scripts/build_taptic_gate_review.py` (new) | builds the human gate-review tool's data: side-by-side camera videos, per-finger force, M/R/V/r on a 50 ms grid, gate decisions of the vector and positive-only variants at every native read |
| `docs/taptic_ours/risk_stats_positive.json` (new) | train-only statistics of the previous positive-only R (ablation / comparison) |
| `tests/taptic/test_risk.py`, `test_ours_model.py`, `test_ours_data.py` (new); `synthetic_root.py` | +37 tests (incl. vector-R reference in both modes, release and direction-change cases, R-mode/version refusal); the synthetic sidecar gains a quantized `joints` column |

**Unchanged:** `modeling_qwen3vl_mot.py` (MoT, attention, causal mask), `flow_matching.py` (handover
selection, FP32 grid, training pair), `diffusion/`, the VQ-VAE model/data/training, `taptic/data.py`,
`taptic/timing.py`, `scripts/midtrain_taptic.sh`, `scripts/taptic/*`. `git diff 05c9328` lists every edit.

---

## 7. Validation

All runs are CPU only, using torch 2.10.0+cu128 and transformers 4.57.1, against the local export
`taptic_trex_style_v2` (identity `1cb54e3d…`, the same as `docs/taptic_train_stats.json`). Models are
reduced random models (hidden 64, 2 MoT layers, real 5+5 tactile token layout, random tiny vision
tower). Command: see §8.

| Requirement | Evidence (test, or artifact) |
|---|---|
| Risk features correct | Independent brute-force reference, exact to 1e-12 on irregular reads, gaps and excluded samples, for vector and positive R and for R_pos / R_neg (`test_features_match_independent_reference_and_formula`); a release and a constant-|F| rotation are seen by vector R and not by positive R (`test_vector_R_sees_decreases_and_direction_changes`) |
| Causal measured joint velocity | Irregular 9/17/13 ms reads, exact speed; age exposed; 10 Hz fallback age/staleness (`test_joint_velocity_uses_actual_timestamps_and_exposes_age`) |
| Normalization and validity | Train-only fit, provenance refusal, stale/segment/excluded/legacy invalid, missing ≠ zero (`test_fit_is_train_only…`, `test_validity_is_explicit…`, `test_missing_risk_is_not_zero…`) |
| No future leakage in features | Samples read or available after the tick never change features (`test_future_samples_never_change_features`) |
| Online = offline | 300 random ticks × 4 real episodes (3 tasks + val), plus synthetic reset/gap episodes (`test_online_extractor_matches_offline_*`) |
| Token order and attention visibility | Spy on every MoT call: risk token positions, routing, M-RoPE positions, cache lengths; bitwise causal-independence checks (`test_token_order…`, `test_causal_visibility…`) |
| Gradient propagation | L_A → `risk_embedder_action` only; L_T → `risk_embedder_tactile` + tactile path, nothing in `_action`, VQ frozen; real batches: every trainable block incl. FLARE (`test_risk_embedders_train…`, `test_actual_trainer_step_on_real_batches`) |
| Multi-split + FP32 endpoints | For τ ∈ {.2,.4,.6}: Action times `flow_time(0..a)`, Tactile `flow_time(a..9)`, FP32, a + t = 10, training τ ∈ (0, τ_split] (`test_split_schedules…`) |
| Ladder coherence + immutability | Bitwise equal to single-split runs; SHA-256 fingerprints unchanged after 12 fast ticks at mixed splits/r_F (`test_ladder_matches_single_split_runs_and_is_immutable`) |
| Slow/fast risk consistency | r_S constant across delays; r_S = r_F at δ = 0; perturbing reads in (S, F] leaves r_S unchanged (`test_delayed_alignment…`) |
| Delayed alignment, no leakage | See the sweep below and `test_delayed_alignment_and_no_future_leakage_on_real_episodes` |
| Inference at all splits | Engine with fixed 0.2/0.4/0.6 and adaptive, for both risk-token and risk-free models (`test_policy_slow_fast_ticks_all_splits`); real-context replay below |
| Fixed τ = 0.4 regression | Risk-free partial/continue/train-step/full **bitwise equal to the base commit** for 3 splits; refactored `run_validation` **bitwise equal to the base commit's** on real batches (fixed and multi) |
| Baseline compatibility | Schema identical to the base; contract load rules; real `save_checkpoint` round trip; vanilla launcher equivalence |
| Deterministic inference | Repeated ladder builds and continuations bitwise equal; replayed engine decisions and chunks identical |

### Real-data alignment sweep ([alignment_evidence.json](alignment_evidence.json))

The sweep covers all 86 validation episodes plus 24 random training episodes, every 3rd anchor, and
all 4 delays: **31,292 items, 0 violations**. Every native item re-derives r_S, r_F and the VQ history
on a copy of the sensor track *truncated at S or F*, and they match the loader exactly.

| δ (ms) | Native tactile valid at F | r_S valid | r_F valid | Supervised slots (mean / min) | Read age at F p50 / p99 / max (ms) | New native reads in (S, F] |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99.96 % | 98.32 % | 98.32 % | 16.00 / 16 | 6.8 / 13.2 / 15.3 | 0 |
| 50 | 99.96 % | 98.32 % | 98.36 % | 15.00 / 15 | 6.5 / 13.1 / 14.7 | 4 |
| 100 | 99.98 % | 98.32 % | 99.39 % | 14.72 / 14 | 6.4 / 13.4 / 16.6 | 8 |
| 150 | 99.98 % | 98.32 % | 99.96 % | 14.00 / 14 | 6.5 / 13.2 / 14.8 | 11–12 |

### Gate replay on held-out episodes ([gate_replay.json](gate_replay.json), vector R)

The replay covers 53 native validation episodes with a fast tick at every native read (93,524 ticks,
1,306.5 s). Thresholds come from training only.

| Setting | τ=0.2 | τ=0.4 | τ=0.6 | Transitions/s | Fallbacks |
|---|---:|---:|---:|---:|---:|
| no hysteresis, no dwell | 50.4 % | 34.3 % | 15.4 % | 1.45 | 432 (0.46 %) |
| h = 0.005, dwell 50 ms | 50.2 % | 34.6 % | 15.3 % | 1.04 | 432 |
| h = 0.005, dwell 100 ms | 50.3 % | 34.4 % | 15.3 % | 0.96 | 432 |

With the previous positive-only R the same settings gave 1.21 / 0.96 / 0.90 transitions/s: counting
decreases and direction changes makes the gate switch somewhat more often at equal occupancy.

Held-out occupancy matches the training-quantile design (50/35/15 %). Occupancy is strongly
task-dependent (h = 0.005, dwell 50 ms): grasp_plastic_glass 81/18/1 %, dosing_pipette 27/40/34 %,
clearing_table 56/13/31 %. Fallbacks (≈8 per episode, ≈100 ms at ~76 Hz) are mostly the first 100 ms of each segment, before the R100/V windows are full. A full per-tick diagnostic log
of one episode is in [gate_log_episode_000024.csv](gate_log_episode_000024.csv).

### End-to-end engine on real processed context ([model_replay.json](model_replay.json))

A held-out episode (24) runs with real images and caption through the Qwen processor, the tiny random
model, 6 slow ticks with ladder(r_S), and fast ticks at every native read (45). It used **all three
splits** (3 transitions with vector R), and all outputs were finite and `ok`, both for OURS and for the risk-free
model with the external gate. Toy CPU timings (ladder ≈ 19 ms, fast tick ≈ 7 ms) measure the toy
model only and say nothing about GPU latency of the 2B model.

Numeric outputs of the random-model tests (losses per split, regression values):
[test_numeric_results.txt](test_numeric_results.txt).

### Human review of the gate (in progress)

To judge whether r and the gate make sense, one validation episode per native task (599 clearing, 93
pipette, 98 glass, 781 credit card, 250 cups, 27 wiping) is shown in a private review artifact with
both cameras, per-finger force and a lane where the reviewer paints the tactile authority they would
choose (low / medium / high = τ 0.2 / 0.4 / 0.6) **before** the gate is revealed. On reveal the blind
annotation is frozen and compared with both variants (agreement on annotated time, Cohen's κ,
confusion matrix, mean bias). Annotations are stored in the artifact's database
(`annotations/ep_<id>`), so they can be read back and analyzed. Results will be added here once the
review is done.

---

## 8. Recommended commands

```bash
# Tests (CPU; from the repo root)
export TAPTIC_ROOT=<export @ 31d33a6, identity 1cb54e3d…>  TAPTIC_PROCESSOR=<Qwen3-VL processor dir>
TAPTIC_PYTHON=<venv python> PYTHONPATH=$PWD bash scripts/smoke_taptic.sh     # or: python -m pytest tests/taptic

# Risk statistics: already committed (docs/taptic_ours/risk_stats.json, vector R). To refit (~5 s):
python -m taptic.risk fit --root $TAPTIC_ROOT --taptic_stats docs/taptic_train_stats.json \
       --out docs/taptic_ours/risk_stats.json            # add --rate_mode positive for the previous R

# Data of the human gate-review tool (videos + signals + both gate variants)
python scripts/build_taptic_gate_review.py --root $TAPTIC_ROOT --out <dir> [--episodes 599 93 ...]

# Probe first (memory/throughput, no checkpoint), then full runs, same budget for every variant
COMMON="TAPTIC_ROOT=$TAPTIC_ROOT MODEL_PATH=$MODEL VQVAE_CKPT=$WORKDIR/vqvae/latest.pt OUTPUT_DIR=$WORKDIR/runs NUM_GPUS=8 TRAIN_BSZ=16"
env $COMMON VARIANT=ours MAX_TRAIN_STEPS=20 bash scripts/midtrain_taptic_ours.sh
env $COMMON VARIANT=multisplit bash scripts/midtrain_taptic_ours.sh     # configs 2 and 3
env $COMMON VARIANT=ours       bash scripts/midtrain_taptic_ours.sh     # config 4
# Config 1 is the running vanilla job (05c9328, scripts/midtrain_taptic.sh); VARIANT=vanilla reproduces it.
# Optional ablation separating delays from multi-split: VARIANT=multisplit TACTILE_DELAY_MS=0

# Offline replay / evaluation helpers
python scripts/validate_taptic_ours.py {alignment,gate_replay,model_replay} --root $TAPTIC_ROOT
```

Inference with a trained checkpoint: build the model with `use_risk_tokens` from
`training_args.json["architecture"]`, load it with `load_with_contract`, then call
`RiskAdaptivePolicy(model, RiskStats.load(risk_stats), GateConfig.from_risk_stats(stats, hysteresis=.005,
min_dwell_ms=50), force_stats)` with `slow_tick(...)` / `fast_tick(F)`. Use `GateConfig(mode='fixed')`
for the fixed-split configurations.

---

## 9. Limitations and open points

1. **Nothing was trained.** The tests use random weights. The tactile VQ-VAE in the tests is random (the
   production `latest.pt`, sha 839f…, is not on this machine), so no statement about learned behaviour,
   task success or the benefit of risk tokens or the gate is possible yet.
2. **GPU/DeepSpeed path not executed here** (no nvcc locally). The shared step, argument parsing,
   validation loop and `save_checkpoint` are tested; `train()` itself should go through the
   `MAX_TRAIN_STEPS` probe first. The cluster environment (torch 2.6, transformers 4.57.3) differs from
   the local one.
3. **The gate is an initialization**: training q50/q85 thresholds and a monotone mapping, with no
   optimization, no safety calibration and no outcome labels. Occupancy is strongly task-dependent. The
   validation split has now been inspected for the replay, so final evaluation should use held-out
   recordings.
4. **Risk definition is uncalibrated**: SDK counts, no normal/shear decomposition, and the coefficients
   are hyperparameters. Vector R also accumulates component quantization (±1 count flicker); its q99
   normalization absorbs the scale but not the extra small-amplitude activity. The statistics file does
   not yet check that the run's coefficients match the ones used for its gate thresholds (scales are
   coefficient-independent; thresholds are not). The 100 ms secant lowers joint noise but adds ≈50 ms of lag to V. M saturates for
   high-force tasks (clearing_table p90 r ≈ 0.33).
5. **The missing-risk embedding is confounded with task.** All legacy (10 Hz) samples, three tasks
   (35 % of episodes), have missing r_S/r_F. On native data, missing risk appears only at segment
   starts and under tactile dropout (r_F). A small risk-dropout on native samples would decorrelate
   this; it is not added, to keep the requested design.
6. **Pre-existing T-Rex behaviour kept for comparability**: the Action continuation and the tactile
   forwards after the first one do not carry the padding mask, so left-padded batches attend pad tokens
   in the no-grad cache. The ladder refuses padded contexts.
7. **Trained delays cover 0–150 ms.** Fast ticks later than S + 150 ms (a slow-tick overrun) are outside
   the trained range. The chunk keeps nominal 100 ms slots from S at inference and the engine only reports
   `first_unelapsed_slot`; no executor or robot server (`scripts/test.py`) was adapted and no hardware
   was commanded.
8. **The ladder costs 9 Action forwards per slow tick** (vs 7 for fixed 0.4). Selecting τ = 0.6 at a
   fast tick costs 6 tactile forwards (vs 4). Real GPU latency is unmeasured.
9. **Missing tactile at inference** uses zero inputs (the trained dropout condition) with r_F missing
   and τ = 0.4. This is a policy choice, not a validated fail-safe.
