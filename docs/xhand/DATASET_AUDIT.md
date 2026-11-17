# Dataset audit — `TAPTIC/insert_credit_card` (Panda + RobotEra XHand)

Every value below was measured on the data. The scripts that produce the numbers are `taptic/*` and `tests/taptic/*`.

## 1. Which copy is used and why

The Hub repo `TAPTIC/insert_credit_card` at revision `c3f6701aec8d3aafd9bfaef0a35c3df40d9547a5` (= `main`) is complete. It holds 297 files: `data/` (13 parquet), `videos/` (26 AV1 mp4), **`tactile_raw/` (201 files: 100 episodes × samples + frames, plus `info.json`)** and `meta/` with all annotation files.

| Copy | Path | Content | Supported |
|---|---|---|---|
| Full Hub download (`snapshot_download(..., local_dir=...)`) | `~/midtrain_merge/src/insert_credit_card` (722 MB, eps 0–99) | everything above | **yes** (`layout = standalone`) |
| Merged midtrain set | `~/midtrain_merge/TAPTIC/midtrain_dataset` (merged eps **602–701**) | same data + other tasks; annotation files under `meta/sources/insert_credit_card/` | **yes** (`layout = merged`) |
| Partial hub-cache snapshot | `~/.cache/huggingface/hub/datasets--TAPTIC--insert_credit_card` | only `meta/` + `data/` were downloaded there (no `videos/`, no `tactile_raw/`) | no (rejected with a clear error) |

The two supported copies are equivalent. The merged copy's `action`, `observation.state` and `subtask_index` (after label remapping) match the Hub revision exactly: 35,746 / 35,746 frames, checked element by element. Converting either copy gives identical samples, split, statistics and per-episode arrays (`tests/taptic/test_a2_hub_layout.py`). Local episode `i` corresponds to merged episode `602 + i`.

## 2. Global facts

| Item | Value |
|---|---|
| Format | LeRobot v3.0, `robot_type = panda_ee`, `fps = 10` |
| Episodes / frames | 100 / 35,746 (length min 15, median 336, max 918) |
| Tasks | 1: "Take the credit card and insert it in the wallet." |
| Existing splits | `splits: {"train": "0:100"}` → **no validation split shipped**; created here (§8) |
| Robot-state model input | not used (T-REX config `use_robot_state=0`) |

## 3. Features

| Key | dtype / shape | Meaning (verified) |
|---|---|---|
| `action` | float32 [18] | `ee.x,y,z` (m), `ee.wx,wy,wz` (rad, rotation vector, see §4), 12 × `xhand.action.right_hand_*` (rad, **absolute** joint targets) |
| `observation.state` | float32 [40] | [0:7] Panda joints **in degrees**; [7:19] XHand measured joints (rad); [19:34] XHand fingertip `calc_force` (raw counts, 5 tips × xyz); [34:40] measured EE pose (same convention as the action) |
| `observation.images.left` | AV1 640×480, 10 fps, GOP 2 | overhead scene camera (card, wallet, hand) → **slow/head image + FLARE targets** |
| `observation.images.wrist` | AV1 640×480, 10 fps, GOP 2 | close-up camera on hand and card → **fast image** |
| `observation.haptics.*` | 3 channels (thumb, index, middle) | **WEART teleoperation-glove** stream (operator side), not robot tactile → **not used** |
| `subtask_index`, `annotation_review_required`, `language_persistent` | per-frame | primitive annotation (§7) |
| `timestamp` | float32 | `frame_index / 10` (uniform); **not** the wall-clock time (§6) |

Example frame (merged ep 602, frame 0):
`action = [0.5541, 0.0018, 0.4571, 0.0019, -0.0054, -0.0117, 0.2094, 0, 0, 0, 0, 0, 0, 0.1222, 0, 0, 0, 0]`;
`state[0:7] = [0.137, 0.095, 0.032, -114.42, 0.117, 114.21, -0.547]` (deg).

## 4. Pose conventions

| Question | Answer | Evidence |
|---|---|---|
| What is `ee.wx/wy/wz`? | **rotation vector** (axis-angle, rad) of `Rx(π)ᵀ·R_base`, the recorder's `gripper_down` frame, so `R_base = Rx(π)·Rot(rotvec)` | FK of the recorded joints (deg) on `panda.urdf` → `panda_link8`: position error < 7e-8 m and rotation error < 2e-7 rad under this hypothesis; 3.14 rad under the "absolute rotvec" hypothesis (`tests/taptic/test_b_actions.py::test_orientation_convention_by_forward_kinematics`). The recorder code says the same (`lerobot_robot_panda/ee_safety.py`, `record_panda_metaquest.py --rotation-frame gripper_down`). |
| Position frame / point | `panda_link8` (flange, no hand offset), URDF base frame, metres | same FK check |
| Commanded or measured? | `action` = **commanded** target; state EE = **measured** (FK) | ‖a[t]−s[t+1]‖ median 4.5 mm < ‖a[t]−s[t]‖ median 7.2 mm |
| Same frame for action and state? | yes (both `gripper_down`) | recorder pipeline; FK check |
| XHand action | absolute targets (rad), range [0, 1.83]; mid/ring/pinky saturate at 1.0 | data |
| Joint order | thumb_bend, thumb_rota1, thumb_rota2, index_bend, index_rota1, index_rota2, mid1, mid2, ring1, ring2, pinky1, pinky2 (`meta/info.json`) | identical in action, state and sidecar `joints` |

## 5. XHand tactile

| Item | Value |
|---|---|
| Source | sidecar `tactile_raw/episode_XXXXXX_right.parquet`; `force` = SDK `calc_force` (15 = 5 tips × xyz); `values` = 1800 raw taxels (5 × 120 × 3) |
| Finger order | thumb, index, middle, ring, little |
| Units | **raw SDK counts**, integer valued; Fx, Fy int8, Fz uint8 ≥ 0 (`/usr/local/xhand_controller` `data_type.hpp`). **No counts→newton calibration is documented → no unit conversion is applied.** |
| Axes | fingertip-local sensor axes; **frame orientation is undocumented** |
| Consistency | the frame's `is_frame_sample` force equals `observation.state[19:34]` in every frame checked |
| Activity (train split) | only **thumb and index** register contact: middle/ring/little have q99−q01 < 1 count (non-zero in ≤ 0.1 % of samples) |
| Offsets / re-zero | `reset_sequence` changes once in the task (ep 606 at frame 324, with a 548 ms read gap); windows never cross it |
| Read errors / drops | 0 / 0 in all 100 episodes |
| Train q01/q99 (thumb, index) | Fx [-14, 9] / [-11, 12], Fy [-17, 4] / [-57, 16], Fz [0, 37] / [0, 61] counts |

`calc_force` is used as-is. The task requires 3-D resultant forces, and the taxel-sum check is not needed because the SDK value is what the recorder already stores in the state.

## 6. Timing (all clocks `CLOCK_MONOTONIC`)

| Quantity | Measured |
|---|---|
| Tactile sample interval | p1 12.0 ms, **median 13.09 ms (76.4 Hz)**, p99 14.2 ms; max 548 ms (the ep 606 reset) |
| Nominal "80 Hz" | not reached: 76.2–76.5 Hz per episode |
| Frame interval (real `observation_ns`) | p1 96.3 ms, median 100.1 ms, p99 104.3 ms; **0.31 % of intervals > 500 ms** (≈1 teleop pause per episode, 0.7–9.6 s, mid-episode). Over a pause the commanded pose moves only 1.2 mm (median). |
| Tactile during frame pauses | keeps streaming (no gaps) |
| Age of the freshest causal tactile sample at a frame's `observation_ns` | median 6.7 ms, p99 13.3 ms |
| Age of the `is_frame_sample` stored in state | up to ~50 ms; a fresher causal sample exists in 89 % of frames |
| Sidecar pre-roll before frame 0 | 1.6–65.6 s (median 7.8 s) → excluded |

## 7. Primitive annotations

See [ANNOTATION_REPORT.md](ANNOTATION_REPORT.md). Summary: there are 896 segments with labels reach, pinch, lift, move, approach, insert, release and push. Five sources agree exactly, with zero discrepancies: per-frame `subtask_index`, `segments.csv`, `reviewed_segments.csv`, `lerobot_annotations.json` and `language_persistent`.

## 8. Derived split and samples

- Episode-level split: `numpy.RandomState(42).permutation` over the sorted local ids, 10 % val. Val episodes are local {0, 10, 22, 39, 44, 45, 53, 70, 80, 83} = merged {602, 612, 624, 641, 646, 647, 655, 672, 682, 685}.
- Samples: **21,821** (train 19,864 / val 1,957). See ANNOTATION_REPORT §3 for the per-primitive table.

## 9. Missing / ambiguous information (not invented)

1. **Force calibration and units** (counts → N) and **fingertip sensor-frame orientation**: undocumented. The model works in raw counts, normalized.
2. **Only one fast camera.** The official T-REX uses two fast wrist views; this robot has one. The second view is **not** duplicated (the batch carries 1 slow + 1 fast image).
3. **The XHand sidecar is ~76.4 Hz, not 80 Hz.** `tactile_raw/info.json` states `sample_rate_hz: 80.0`, but the measured median interval is 13.09 ms.
4. A Hub download used for conversion must include `videos/` and `tactile_raw/`. A meta+data-only download (as in the local hub cache) is rejected.
5. The `observation.haptics.*` units are undocumented (operator glove). This stream is unused.
6. Annotations are visual estimates (README: indicative ±8–12 frame uncertainty). 61 % of segments carry `review_required`, which is kept per sample as metadata.
