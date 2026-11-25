"""Isolated TAPTIC adapter for the unchanged cascaded T-Rex loop.

Two tactile inputs (--taptic_tactile_input):
  taxels    : full taxels enter the vector embedder as five 360-D tokens.
              No VQ codes. Missing-taxel episodes get zeros (not masked).
  sdk_force : recorded SDK fingertip force. Current sample -> five 3-D tokens
              (normalized with the VQ-VAE train-only stats), plus a causal
              native-rate (~76 Hz) history window [W,5,3] raw that the embedded
              VQ-VAE turns into five per-finger codes. Legacy 10 Hz episodes and
              stale/missing samples are marked invalid and excluded from the
              tactile-expert loss via `tactile_loss_mask` (they still train the
              action expert and FLARE). Never interpolated.
No future observations, deformation images, contact labels or risk labels.
The historical `tactile_f6s` key carries taxels or forces; it does not imply six axes.

Delayed tactile conditioning (--tactile_delay_ms, sdk_force only). Each sample
has a slow anchor S = observation_ns of frame t (images, language, action
conditioning and r_S use data available by S) and a physical fast tick
F = S + delta, delta drawn uniformly from tactile_delay_ms (train) or
deterministically from the index (val). The current force, the VQ-VAE history
and r_F use only samples with read_ns <= F and available_ns <= F. The action
chunk keeps its recorded 10 Hz targets/timestamps (no shift, no interpolation);
target slot k is supervised for the tactile expert only if its recorded time
times[t+k] >= F (`tactile_slot_mask`), i.e. it has not elapsed at F.
With tactile_delay_ms=[0] (default) F = S and the batch is the baseline's.

Contact risk (--risk_stats): r_S at S and r_F at F from taptic.risk, with
explicit validity; legacy 10 Hz episodes are always risk-invalid.

Future-risk TARGETS (--risk_future_head_weight > 0 or --risk_future_probe 1):
`risk_future[k]` = the causal risk r(t_k) at the recorded slot time t_k = times[t+k]
(the value an online extractor would report at t_k), with `risk_future_valid`, and the
masks `risk_future_after_S` (t_k > S) / `risk_future_after_F` (t_k > F, inside the
aligned window). They are supervision for auxiliary heads only and never enter the
model as inputs.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from taptic.actions import chunk_21d, rotvec_gd_to_matrix
from taptic.data import (Source, CAMERAS, NORMALIZATIONS, finite, normalize, denormalize,
                         tactile_at, tactile_stream_between)
from taptic.video import VideoReader
from taptic.timing import build_quality, latest_quality
from taptic.force_history import build_force_track, history_at, spread_history_at
from taptic.risk import (RiskConfig, RiskStats, risk_features, features_at, joint_track_from_frames,
                         SCALAR_KEYS as RISK_KEYS)
from .flow_matching import validate_split_config

TACTILE_INPUTS = ('taxels', 'sdk_force')
TAPTIC_FORCE_SIGNAL = 'taptic_sdk_force'
MAX_TACTILE_DELAY_MS = 500


def risk_future_enabled(cfg):
    return bool(float(getattr(cfg, 'risk_future_head_weight', 0.) or 0.) > 0
                or int(getattr(cfg, 'risk_future_probe', 0) or 0))


def tactile_delays_ms(cfg):
    d = tuple(getattr(cfg, 'tactile_delay_ms', None) or (0,))
    if (len(set(d)) != len(d) or any(isinstance(x, bool) or int(x) != x or not 0 <= x <= MAX_TACTILE_DELAY_MS
                                     for x in d)):
        raise ValueError(f'tactile_delay_ms must be distinct integers in [0, {MAX_TACTILE_DELAY_MS}]')
    return tuple(int(x) for x in d)


def tactile_input(cfg):
    mode = getattr(cfg, 'taptic_tactile_input', 'taxels')
    if mode not in TACTILE_INPUTS:
        raise ValueError(f'taptic_tactile_input must be one of {TACTILE_INPUTS}')
    return mode


def tactile_history_points(cfg):
    """Points per finger of the long force history (0 = off: the token carries the current sample only).

    --tactile_history_span_ms > 0 (sdk_force only): each per-finger vector token carries
    --tactile_history_points force samples spread evenly over the last span (oldest first, the
    last one is the current sample), normalized with the same train-only force statistics.
    The VQ-VAE codes keep their native 16-sample window, so the token count is unchanged.
    """
    if tactile_input(cfg) != 'sdk_force' or float(getattr(cfg, 'tactile_history_span_ms', 0) or 0) <= 0:
        return 0
    return int(getattr(cfg, 'tactile_history_points', 8))


def tactile_feature_dim(cfg):
    """Width of each of the five per-finger vector tokens."""
    if tactile_input(cfg) == 'taxels':
        return 360
    n = tactile_history_points(cfg)
    return 3 * n if n else 3


def validate_vqvae_config(vq):
    """The embedded VQ-VAE must be the TAPTIC native SDK-force one."""
    got = (vq.get('input_signal'), vq.get('n_hands'), vq.get('n_fingers'), vq.get('per_finger_dim'))
    if got != (TAPTIC_FORCE_SIGNAL, 1, 5, 3):
        raise ValueError('TAPTIC needs a VQ-VAE trained with tactile_vqvae.train --data_format taptic '
                         f'(input_signal, n_hands, n_fingers, per_finger_dim) = {got}')


def load_force_stats(cfg):
    """Train-only SDK-force stats + provenance from the TAPTIC VQ-VAE checkpoint."""
    from tactile_vqvae.data.stats import TacF6Stats
    path = getattr(cfg, 'tactile_f6_stats_ckpt', '') or getattr(cfg, 'vqvae_ckpt', '')
    if not path:
        raise ValueError('sdk_force input needs --vqvae_ckpt or --tactile_f6_stats_ckpt '
                         '(TAPTIC VQ-VAE checkpoint carrying the force statistics)')
    blob = torch.load(path, map_location='cpu', weights_only=False)
    validate_vqvae_config(blob['config'])
    stats = TacF6Stats.from_dict(blob['stats'])
    if stats.dim != 15:
        raise ValueError('TAPTIC force stats must have 15 channels')
    return stats, blob.get('data', {})


def validate_config(cfg):
    if (cfg.action_dim, cfg.action_chunk) != (21, 16):
        raise ValueError('TAPTIC requires action_dim=21, action_chunk=16')
    mode = tactile_input(cfg)
    if getattr(cfg, 'use_tactile_deform', 0):
        raise ValueError('TAPTIC requires use_tactile_deform=0')
    if mode == 'taxels':
        for k in ('use_tactile_vqvae', 'use_tactile_code'):
            if getattr(cfg, k, 0):
                raise ValueError(f'TAPTIC raw taxels require {k}=0 (use --taptic_tactile_input sdk_force for VQ)')
    elif getattr(cfg, 'use_tactile_code', 0) and not getattr(cfg, 'use_tactile_vqvae', 0):
        raise ValueError('TAPTIC has no pre-computed code sidecars; use the embedded VQ-VAE (use_tactile_vqvae=1)')
    if not getattr(cfg, 'use_tactile_vec', 0):
        raise ValueError('TAPTIC requires use_tactile_vec=1')
    if tuple(getattr(cfg, 'tactile_delay_offsets', (0,))) != (0,):
        raise ValueError('TAPTIC causal baseline requires tactile_delay_offsets=[0] '
                         '(frame offsets; physical delays use --tactile_delay_ms)')
    if tactile_delays_ms(cfg) != (0,) and mode != 'sdk_force':
        raise ValueError('delayed tactile conditioning (tactile_delay_ms) requires sdk_force input')
    if float(getattr(cfg, 'tactile_history_span_ms', 0) or 0) > 0:
        if mode != 'sdk_force':
            raise ValueError('tactile_history_span_ms requires sdk_force input')
        if int(getattr(cfg, 'tactile_history_points', 8)) < 2:
            raise ValueError('tactile_history_points must be >= 2')
    if getattr(cfg, 'use_risk_tokens', 0) and not getattr(cfg, 'risk_stats', ''):
        raise ValueError('use_risk_tokens=1 requires --risk_stats (train-only risk statistics)')
    if getattr(cfg, 'risk_stats', '') and mode != 'sdk_force':
        raise ValueError('contact-risk features require sdk_force input')
    if risk_future_enabled(cfg) and not getattr(cfg, 'risk_stats', ''):
        raise ValueError('future-risk targets (risk_future_head_weight / risk_future_probe) require --risk_stats')
    validate_split_config(cfg)


class TRexTapticDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, split='train', episode_ids=None, _stats=None):
        validate_config(config)
        self.config, self.processor, self.accelerator = config, processor, accelerator
        self.source = Source(config.lerobot_root)
        self.stats_data = _stats if _stats is not None else json.loads(Path(config.taptic_stats).read_text())
        block = self.stats_data['taptic']
        # Evaluation only (jobs/eval/offline_eval.py --root on another dataset; docs/TOUCH_MODULES.md): score a
        # model on a dataset it was not trained on, with the model's own normalization and touch encoder. The
        # checks tying the statistics and the VQ-VAE to this dataset's training episodes are skipped.
        self.other_dataset_eval = bool(int(getattr(config, 'taptic_eval_other_dataset', 0) or 0))
        if self.other_dataset_eval and split == 'train':
            raise ValueError('taptic_eval_other_dataset is for evaluation splits only')
        if block['source_identity'] != self.source.identity and not self.other_dataset_eval:
            raise ValueError('statistics belong to a different dataset')
        ids = block['train_episodes']
        if not self.other_dataset_eval and (
                len(ids) != len(set(ids)) or any(self.source.episodes.loc[e, 'suggested_split'] != 'train' for e in ids)):
            raise ValueError('statistics contain non-training episodes')
        st = block['action']
        mode = block.get('normalization', 'affine_train_minmax_unclipped')
        if mode not in NORMALIZATIONS or (mode == 'trex_q01_q99_per_step_clipped') != ('q01' in st):
            raise ValueError(f'statistics: normalization {mode!r} does not match the stored action bounds')
        lo, hi = (np.asarray(st['q01']), np.asarray(st['q99'])) if 'q01' in st else \
                 (np.asarray(st['min']), np.asarray(st['max']))
        shape = (16, 21) if 'q01' in st else (21,)
        if lo.shape != shape or hi.shape != shape or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError('invalid action normalization statistics')
        self.action_stats = st
        self.scope = self.source.episodes if episode_ids is None else self.source.episodes.loc[episode_ids]
        # Dataset variant selection (docs/TOUCH_MODULES.md): drop every episode recorded in these source repos,
        # e.g. the old 10 Hz unscrew when the derived dataset also holds its re-collection.
        excl = set(getattr(config, 'taptic_exclude_sources', None) or ())
        if excl:
            unknown = excl - set(self.scope.source_repo.astype(str))
            if unknown:
                raise ValueError(f'taptic_exclude_sources not in this dataset: {sorted(unknown)}')
            self.scope = self.scope[~self.scope.source_repo.astype(str).isin(excl)]
        training = sorted(map(int, self.scope.index[self.scope.suggested_split == 'train']))
        if sorted(ids) != training and not self.other_dataset_eval:
            raise ValueError('statistics training episodes do not match dataset scope')
        self.image_size = tuple(config.image_size) if getattr(config, 'image_size', None) else None
        self.use_flare = bool(getattr(config, 'use_flare', 0))
        self.n_flare_steps = int(getattr(config, 'n_flare_steps', 0)) if self.use_flare else 0
        self.flare_frame_stride = int(getattr(config, 'flare_frame_stride', 1))
        if self.flare_frame_stride < 1:
            raise ValueError('FLARE stride must be positive')
        self.tactile_input = tactile_input(config)
        self.use_vqvae = bool(getattr(config, 'use_tactile_vqvae', 0))
        self.vq_window = int(getattr(config, 'vqvae_window', 16))
        if self.tactile_input == 'sdk_force':
            self.force_stats, vq_data = load_force_stats(config)
            parent = self.source.derived_from or {}
            vq_id = vq_data.get('source_identity')
            vq_train = set(vq_data.get('train_episodes', []))
            # Touch encoder trained on another dataset (docs/TOUCH_MODULES.md), e.g. the feedback-off recordings
            # with the owner's VQ-VAE: allowed only when its source dataset is named (>= 12 hex chars of the
            # identity). Its training episodes belong to that dataset, so the overlap check does not apply; the two
            # datasets must share no recordings (checked for feedback on/off in eval/nohaptic/audit.md).
            vq_other = str(getattr(config, 'taptic_vqvae_source_identity', '') or '')
            if vq_other:
                if len(vq_other) < 12 or not str(vq_id or '').startswith(vq_other):
                    raise ValueError(f'VQ-VAE source identity {vq_id!r} does not match '
                                     f'taptic_vqvae_source_identity {vq_other!r}')
                if vq_id == self.source.identity:
                    raise ValueError('taptic_vqvae_source_identity is only for a VQ-VAE trained on another dataset')
                vq_train = set()
            elif self.other_dataset_eval:
                vq_train = set()
            elif parent and vq_id == parent.get('parent_identity'):
                # Derived dataset (meta/derived_from.json, scripts/taptic/append_recordings.py): the VQ-VAE was
                # trained on the parent; valid only if every episode it saw is unchanged here.
                if not vq_train <= set(parent.get('unchanged_episodes', [])):
                    raise ValueError('VQ-VAE was trained on parent episodes that this derived dataset changed')
            elif vq_id not in (None, self.source.identity):
                raise ValueError('VQ-VAE force statistics belong to a different dataset '
                                 '(name its source with taptic_vqvae_source_identity to use it deliberately)')
            if vq_train and not vq_train <= set(ids):
                raise ValueError('VQ-VAE was trained on non-training episodes')
            if self.use_vqvae and vq_data.get('window', self.vq_window) != self.vq_window:
                raise ValueError('VQ-VAE window does not match vqvae_window')
        self.delays_ns = tuple(int(d) * 1_000_000 for d in tactile_delays_ms(config))
        self.hist_points = tactile_history_points(config)
        self.hist_span_ns = int(round(float(getattr(config, 'tactile_history_span_ms', 0) or 0) * 1e6))
        # Force-interaction tasks for the contact-weighted tactile loss (flag only; no model input).
        self.force_tasks = frozenset(getattr(config, 'tactile_force_tasks', None) or ())
        self.contact_threshold = float(getattr(config, 'force_contact_threshold', 2.0))
        # Touch events for the event-weighted tactile loss (flag only; no model input). taptic/touch_events.py.
        self.event_tasks = frozenset(getattr(config, 'tactile_event_tasks', None) or ())
        self.event_jump = {}
        self.event_flags = (self._touch_event_flags(config)
                            if float(getattr(config, 'tactile_event_weight', 1.) or 1.) != 1.
                            or getattr(config, 'emit_touch_events', False) else None)
        self.emit_slot_mask = self.delays_ns != (0,)
        self.risk_stats = self.risk_cfg = None
        self.risk_future = risk_future_enabled(config)
        if getattr(config, 'risk_stats', ''):
            self.risk_cfg = RiskConfig.from_args(config)
            self.risk_stats = RiskStats.load(config.risk_stats)
            self.risk_stats.check_config(self.risk_cfg)
            # Same rule as the touch encoder above (docs/TOUCH_MODULES.md): contact-risk statistics fitted on
            # another dataset are accepted in a derived dataset (fitted on unchanged parent episodes), when that
            # dataset is named with taptic_vqvae_source_identity, or in a cross-dataset evaluation.
            rs_id = str(self.risk_stats.source_identity or '')
            named = str(getattr(config, 'taptic_vqvae_source_identity', '') or '')
            parent = self.source.derived_from or {}
            if rs_id == self.source.identity:
                self.risk_stats.check_source(self.source, ids)
            elif self.other_dataset_eval or (len(named) >= 12 and rs_id.startswith(named)):
                pass
            elif parent and rs_id == parent.get('parent_identity'):
                if not set(self.risk_stats.train_episodes) <= set(parent.get('unchanged_episodes', [])) & set(ids):
                    raise ValueError('risk statistics were fitted on parent episodes this derived dataset changed '
                                     'or does not train on')
            else:
                raise ValueError('risk statistics belong to a different dataset')
        self._cache_key = None
        self._video = VideoReader()
        self._set_split(split)

    def _touch_event_flags(self, config):
        """{episode: bool [T]} = frame t lies in an event window (taptic/touch_events.py), for every episode in
        scope. The per-task `resistance` threshold comes from the native TRAINING episodes only."""
        from taptic.touch_events import (DEFAULT_LABELS, DEFAULT_TYPES, DEFAULT_WINDOW_MS, episode_events,
                                         finger_force, jump_threshold, window_flags)
        types = tuple(getattr(config, 'tactile_event_types', None) or DEFAULT_TYPES)
        pre_ms, post_ms = (list(getattr(config, 'tactile_event_window_ms', None) or DEFAULT_WINDOW_MS) * 2)[:2]
        names = self.source.table.schema.names
        cols = ['episode_index', 'frame_index', 'observation.state'] + (['subtask_index'] if 'subtask_index' in names else [])
        core = self.source.table.to_table(columns=cols).to_pandas()
        F_by, s_by = {}, {}
        for ep, d in core.groupby('episode_index', sort=False):
            if int(ep) in self.scope.index:
                d = d.sort_values('frame_index')
                F_by[int(ep)] = finger_force(np.stack(d['observation.state'].to_numpy()))
                s_by[int(ep)] = d['subtask_index'].to_numpy() if 'subtask_index' in d else None
        label_ids = ()
        if 'label' in types:
            want = set(getattr(config, 'tactile_event_labels', None) or DEFAULT_LABELS)
            idx = json.loads((self.source.root / 'meta/subtasks.json').read_text())['subtask_to_index']
            label_ids = tuple(int(v) for k, v in idx.items() if k in want)
        q = float(getattr(config, 'tactile_event_jump_quantile', .97))
        tasks = self.scope.source_task.astype(str)
        train = set(map(int, self.scope.index[self.scope.suggested_split == 'train']))
        flags = {}
        for task in sorted(set(tasks)):
            eps = [int(e) for e in self.scope.index[tasks == task]]
            native_train = [F_by[e] for e in eps if e in train and not bool(self.scope.loc[e, 'tactile_interpolated'])]
            self.event_jump[task] = jump_threshold(native_train, q) if native_train else float('inf')
            for e in eps:
                ev = episode_events(F_by[e], types=types, subtask=s_by[e], label_ids=label_ids,
                                    jump=self.event_jump[task])
                flags[e] = window_flags(ev, pre_ms / 1000., post_ms / 1000.)
        return flags

    def _set_split(self, split):
        self.split = split
        self._rows = []
        counts = []
        for ep in self.scope.index[self.scope.suggested_split == split]:
            anchors = self.source.anchors(int(ep))
            horizon = max(15, self.n_flare_steps * self.flare_frame_stride)
            times = self.source.clocks(int(ep))
            anchors = np.asarray([t for t in anchors if t + horizon < len(times)
                                  and not np.any(np.diff(times[t:t+horizon+1]) > 150_000_000)], dtype=np.int64)
            if len(anchors):
                self._rows.extend((int(ep), int(t)) for t in anchors)
                counts.append(len(anchors))
        self._cum_frames = np.cumsum(counts).astype(np.int64)
        self._num_episodes = len(counts)

    def create_val_split(self, val_ratio=None, seed=None):
        # Source-recording split shipped in metadata; never randomly split frames.
        val = copy.copy(self)
        val._cache_key = None
        val._video = VideoReader()
        val._set_split('val')
        return val

    def __len__(self):
        return len(self._rows)

    def _episode(self, ep):
        if self._cache_key != ep:
            d = self.source.frames(ep, haptics=True)
            s = finite(np.stack(d['observation.state']).astype(np.float32), 'state')
            a = finite(np.stack(d.action).astype(np.float32), 'action')
            times = self.source.clocks(ep)
            sm, available = self.source.tactile(ep)
            native = not bool(self.source.episodes.loc[ep, 'tactile_interpolated'])
            gap_ms = getattr(self.config, 'tactile_gap_ms', 50.) if native else 150.
            self._cache_quality = build_quality(sm, available, native, int(gap_ms*1e6))
            self._cache_force = (build_force_track(sm, available, times, int(gap_ms*1e6))
                                 if self.tactile_input == 'sdk_force' and native else None)
            self._cache_joint = (joint_track_from_frames(times, s)
                                 if self.risk_cfg is not None and self.risk_cfg.joint_source == 'state10hz'
                                 else None)
            self._cache = (d, s, a, times, sm, available)
            self._cache_key = ep
        return self._cache

    def _img(self, ep, cam, t):
        row = self.source.episodes.loc[ep]
        path = self.source.info['video_path'].format(video_key=cam,
            chunk_index=int(row[f'videos/{cam}/chunk_index']), file_index=int(row[f'videos/{cam}/file_index']))
        ts = float(row[f'videos/{cam}/from_timestamp']) + t / 10
        return self._video.frame(str(self.source.root / path), ts, 10)

    def _delay_ns(self, idx):
        if len(self.delays_ns) == 1:
            return self.delays_ns[0]
        if self.split == 'train':   # torch RNG: seeded per DataLoader worker
            return self.delays_ns[int(torch.randint(len(self.delays_ns), ()))]
        return self.delays_ns[idx % len(self.delays_ns)]   # reproducible validation

    def __getitem__(self, idx):
        ep, t = self._rows[idx]
        d, s, a, times, sm, available = self._episode(ep)
        slow_ns = int(times[t])
        delay_ns = self._delay_ns(idx)
        fast_ns = slow_ns + delay_ns
        slot_ns = times[t:t+16]
        # F stays inside this anchor's continuous chunk window (anchors exclude
        # recorder pauses), so no paused interval or other episode is touched.
        aligned = bool(fast_ns <= int(slot_ns[-1]) and not np.any(np.diff(times[t:t+16]) > 150_000_000))
        slot_mask = (slot_ns >= fast_ns) if aligned else np.zeros(16, bool)
        native = not bool(self.source.episodes.loc[ep, 'tactile_interpolated'])
        stale_ms = getattr(self.config, 'tactile_stale_ms', 50.) if native else getattr(self.config, 'tactile_legacy_stale_ms', 150.)
        stale_ns = int(stale_ms * 1e6)
        current_quality = latest_quality(self._cache_quality, int(times[t]), int(times[0]), native, stale_ns)
        taxels, force, valid, force_valid, sample_ns = tactile_at(
            sm, available, int(times[t]), int(times[0]), native,
            quality=self._cache_quality, stale_ns=stale_ns)
        stream = tactile_stream_between(sm, available, int(times[t]),
                                        int(times[t-1]) if t else None, native,
                                        quality=self._cache_quality, stale_ns=stale_ns)
        recorder_gap = bool(t and times[t]-times[t-1] > 150_000_000)
        current_quality['recording_gap_before_tick'] = recorder_gap
        if recorder_gap:
            # Keep every raw row for audit/replay, but do not present a paused
            # recorder interval as a continuous within-policy feature history.
            stream['history_valid'][:] = False
            stream['derivative_valid'][:] = False
        # Missing taxels use neutral zeros and a separate explicit validity mask.
        scaled = taxels / np.asarray([128., 128., 255.], np.float32) if valid else np.zeros_like(taxels)
        state = None
        if getattr(self.config, 'use_robot_state', 0):
            R = rotvec_gd_to_matrix(s[t, 37:40])
            state = np.concatenate([s[t, 34:37], R[:, 0], R[:, 1], s[t, 7:19]]).astype(np.float32)
            state = normalize(state, self.stats_data["taptic"]["state"])
        flare = []
        for k in range(self.n_flare_steps):
            ft = t + (k + 1) * self.flare_frame_stride
            # Full valid target windows only (no hidden clamping or time gaps).
            if ft >= len(times) or np.any(np.diff(times[t:ft+1]) > 150_000_000):
                raise ValueError('FLARE target outside valid temporal window; lower horizon or disable FLARE')
            flare.append(self._img(ep, CAMERAS[0], ft))
        haptics = {}
        for key in d.columns:
            if key.startswith('observation.haptics.'):
                v = np.asarray(d.iloc[t][key])
                finite(v, key)
                haptics[key.removeprefix('observation.haptics.')] = v.copy()
        force_history, force_valid_tick, n_real = None, False, 0
        if self.tactile_input == 'sdk_force':
            # Native ~76 Hz causal history; legacy (10 Hz) episodes are missing, never upsampled.
            if self._cache_force is not None:
                # History/current force end at the latest sample available by F (not S).
                force_history, force_valid_tick, n_real, _ = history_at(
                    self._cache_force, fast_ns, self.vq_window, stale_ns)
                force_valid_tick = bool(force_valid_tick and aligned and slot_mask.any())
            else:
                force_history = np.zeros((self.vq_window, 5, 3), np.float32)
            policy_tactile = (self.force_stats.normalize(force_history[-1]).astype(np.float32)
                              if force_valid_tick else np.zeros((5, 3), np.float32))
            if self.hist_points:
                policy_tactile = np.zeros((5, 3 * self.hist_points), np.float32)
                if force_valid_tick:
                    long_hist, ok, _ = spread_history_at(self._cache_force, fast_ns, self.hist_span_ns,
                                                         self.hist_points, stale_ns)
                    if not ok:
                        raise RuntimeError('long force history invalid at a valid tick')
                    policy_tactile = (self.force_stats.normalize(long_hist).astype(np.float32)
                                      .transpose(1, 0, 2).reshape(5, -1))       # [5 fingers, points x xyz]
        else:
            policy_tactile = scaled.reshape(5, 360)
        force_contact = bool(force_valid_tick and force_history is not None
                             and str(self.source.episodes.loc[ep, 'source_task']) in self.force_tasks
                             and np.linalg.norm(force_history[-1], axis=-1).max() > self.contact_threshold)
        touch_event = bool(self.event_flags is not None and force_valid_tick
                           and str(self.source.episodes.loc[ep, 'source_task']) in self.event_tasks
                           and self.event_flags[ep][t])
        risk_S = risk_F = None
        future = None
        if self.risk_stats is not None:
            ticks = [slow_ns, fast_ns] + ([int(x) for x in slot_ns] if self.risk_future else [])
            feats = risk_features(self._cache_force if native else None, np.array(ticks, dtype=np.int64),
                                  self.risk_stats, self.risk_cfg, self._cache_joint)
            risk_S, risk_F = features_at(feats, 0), features_at(feats, 1)
            if not aligned:
                risk_F['risk_valid'] = False
            if self.risk_future:
                # Targets only: causal r(t_k) at each recorded slot time, strictly after S / F.
                future = {'risk': feats['risk'][2:].astype(np.float32),
                          'valid': feats['risk_valid'][2:].astype(bool),
                          'after_S': np.asarray(slot_ns > slow_ns, bool),
                          'after_F': np.asarray(slot_ns > fast_ns, bool) & aligned}
        return {'head': self._img(ep, CAMERAS[0], t), 'fast': [self._img(ep, CAMERAS[1], t)],
                'language': self.source.episodes.loc[ep, 'caption'], 'action_chunk': chunk_21d(s, a, t),
                'tactile_f6': policy_tactile, 'tactile_taxels': taxels,
                'force_history': force_history, 'force_valid': force_valid_tick,
                'force_history_n_real': n_real,
                'tactile_valid': np.full(5, valid), 'sdk_force': force, 'sdk_force_valid': force_valid,
                'tactile_stream': stream, 'tactile_quality': current_quality,
                'state_raw': state, 'recorded_haptics': haptics, 'flare_frames': flare,
                'meta': {'episode': ep, 'frame': t, 'observation_ns': int(times[t]),
                         'tactile_available_ns': sample_ns, 'clock_is_nominal': not native,
                         'recording_gap_before_tick': recorder_gap,
                         'action_timestamps_ns': times[t:t+16].copy(),
                         'slow_ns': slow_ns, 'fast_ns': fast_ns, 'tactile_delay_ns': delay_ns,
                         'fast_tick_aligned': aligned}, 'delay_k': 0,
                'risk_S': risk_S, 'risk_F': risk_F, 'tactile_slot_mask': slot_mask,
                'risk_future': future, 'force_contact': force_contact, 'touch_event': touch_event}

    def anchor_risk(self, log=None):
        """r_S (risk at the slow tick S = times[t]) and its validity for every row of this split,
        computed exactly as __getitem__ does (same force track; legacy episodes are invalid).
        Used to build the high-risk oversampling sampler; no images or frames are decoded."""
        if self.risk_stats is None:
            raise ValueError('anchor_risk needs --risk_stats')
        rows = np.asarray(self._rows, dtype=np.int64).reshape(-1, 2)
        risk = np.zeros(len(rows), np.float32)
        valid = np.zeros(len(rows), bool)
        eps = rows[:, 0]
        for n, ep in enumerate(dict.fromkeys(eps.tolist())):
            if bool(self.source.episodes.loc[ep, 'tactile_interpolated']):
                continue
            idx = np.flatnonzero(eps == ep)
            times = self.source.clocks(ep)
            sm, available = self.source.tactile(ep)
            gap_ms = getattr(self.config, 'tactile_gap_ms', 50.)
            track = build_force_track(sm, available, times, int(gap_ms * 1e6))
            joint = None
            if self.risk_cfg.joint_source == 'state10hz':
                s = finite(np.stack(self.source.frames(ep)['observation.state']).astype(np.float32), 'state')
                joint = joint_track_from_frames(times, s)
            f = risk_features(track, np.asarray(times)[rows[idx, 1]].astype(np.int64),
                              self.risk_stats, self.risk_cfg, joint)
            risk[idx], valid[idx] = f['risk'], f['risk_valid']
            if log and (n + 1) % 200 == 0:
                log(f'[risk] anchor risk: {n + 1} episodes')
        return risk, valid

    def normalize_actions(self, x):
        return normalize(x, self.action_stats)

    def denormalize_actions(self, x):
        return denormalize(x, self.action_stats)

    def _pil(self, arr):
        img = PIL.Image.fromarray(arr)
        return img.resize(self.image_size, PIL.Image.Resampling.LANCZOS) if self.image_size is not None else img

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)
        actions = np.stack([x["action_chunk"] for x in batch]).astype(np.float32)        # [B,16,21]
        norm_actions = torch.tensor(self.normalize_actions(actions),
                                    dtype=torch.bfloat16)
        # ── flow matching (identical to MidtrainTacFlareDataset.collate_fn) ──
        d = torch.distributions.Beta(torch.tensor(1.5, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))
        time = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = d.sample((B,)) * 0.999 + 0.001
        if getattr(cfg, 'flow_time_precision', 'fp32') == 'legacy_bf16':
            time_r = time_r.to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        norm_tac = torch.from_numpy(np.stack([x["tactile_f6"] for x in batch]))
        norm_tac_d = norm_tac  # same inference tick; no future tactile leakage
        hist = None
        loss_mask = None
        if self.tactile_input == 'sdk_force':
            loss_mask = torch.tensor([bool(x['force_valid']) for x in batch])
            if self.use_vqvae:
                hist = torch.from_numpy(np.stack([x['force_history'] for x in batch]))  # [B,W,5,3] raw

        all_ids, all_pv, all_thw = [], [], []
        for x in batch:
            imgs = [self._pil(x["head"])] + [self._pil(f) for f in x["fast"]]
            content = ([{"type": "image"}, {"type": "text", "text": x["language"]}]
                       + [{"type": "image"} for _ in x["fast"]])
            text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True)
            inp = self.processor(text=text, images=imgs, return_tensors="pt", padding=False)
            all_ids.append(inp.input_ids[0])
            all_pv.append(inp.pixel_values)
            all_thw.append(inp.image_grid_thw)

        flare_pv = flare_thw = None
        if self.n_flare_steps > 0:
            finp = self.processor.image_processor([self._pil(f) for x in batch for f in x["flare_frames"]],
                                                  return_tensors="pt")
            flare_pv, flare_thw = finp.pixel_values.to(torch.bfloat16), finp.image_grid_thw

        pad_id = self.processor.tokenizer.pad_token_id or 0
        L = max(i.shape[0] for i in all_ids)
        ids, attn = [], []
        for i in all_ids:
            p = L - i.shape[0]
            ids.append(F.pad(i, (p, 0), value=pad_id))
            a = torch.ones(L, dtype=torch.long)
            a[:p] = 0
            attn.append(a)
        # Variable native sample counts, including empty intervals. Padding is
        # never a measurement: consumers must use sample_mask/lengths.
        risk = {}
        if batch[0]['risk_S'] is not None:
            for tag in ('S', 'F'):
                recs = [x['risk_' + tag] for x in batch]
                risk[f'risk_{tag}'] = torch.tensor([r['risk'] for r in recs], dtype=torch.float32)
                risk[f'risk_{tag}_valid'] = torch.tensor([r['risk_valid'] for r in recs], dtype=torch.bool)
                feats = {k: torch.tensor([r[k] for r in recs]) for k in RISK_KEYS}
                feats['magnitude'] = torch.tensor([r['magnitude'] for r in recs], dtype=torch.float32)
                feats['rate'] = torch.tensor([r['rate'] for r in recs], dtype=torch.float32)
                risk[f'risk_features_{tag}'] = feats
        else:
            risk = {f'risk_{tag}{suf}': None for tag in 'SF' for suf in ('', '_valid')}
            risk.update(risk_features_S=None, risk_features_F=None)
        fut = [x.get('risk_future') for x in batch]
        if fut[0] is not None:
            risk['risk_future'] = torch.from_numpy(np.stack([f['risk'] for f in fut]))
            for k in ('valid', 'after_S', 'after_F'):
                risk[f'risk_future_{k}'] = torch.from_numpy(np.stack([f[k] for f in fut]))
        else:
            risk.update(risk_future=None, risk_future_valid=None, risk_future_after_S=None,
                        risk_future_after_F=None)
        slot_mask = (torch.from_numpy(np.stack([x['tactile_slot_mask'] for x in batch]))
                     if self.emit_slot_mask else None)
        streams = [x['tactile_stream'] for x in batch]
        lengths = torch.tensor([len(s['available_ns']) for s in streams], dtype=torch.long)
        stream_batch = {key: torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(s[key]) for s in streams], batch_first=True)
            for key in streams[0]}
        stream_batch['lengths'] = lengths
        stream_batch['sample_mask'] = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
        return {
            "input_ids": torch.stack(ids), "attention_mask": torch.stack(attn),
            "pixel_values": torch.cat(all_pv), "image_grid_thw": torch.cat(all_thw),
            "n_slow_images": 1,
            "noisy_actions": x_t, "target": u_t, "timesteps": time, "norm_actions": norm_actions,
            "tactile_f6s": norm_tac, "tactile_deforms": None,
            "tactile_f6s_delayed": norm_tac_d, "tactile_deforms_delayed": None,
            "tactile_codes": None, "tactile_f6_history": hist,
            "delay_k": torch.tensor([x["delay_k"] for x in batch], dtype=torch.long),
            "time_r": time_r, "eps_r": eps_r, "state_raw": (torch.from_numpy(np.stack([x["state_raw"] for x in batch])).to(torch.bfloat16)
                if batch[0]["state_raw"] is not None else None),
            "flare_pixel_values": flare_pv, "flare_grid_thw": flare_thw,
            "meta": [x["meta"] for x in batch],
            "tactile_taxels": torch.from_numpy(np.stack([x["tactile_taxels"] for x in batch])),
            "tactile_valid": torch.from_numpy(np.stack([x["tactile_valid"] for x in batch])),
            "sdk_force": torch.from_numpy(np.stack([x["sdk_force"] for x in batch])),
            "sdk_force_valid": torch.tensor([x["sdk_force_valid"] for x in batch]),
            "tactile_stream": stream_batch,
            "tactile_quality": {k: torch.tensor([x['tactile_quality'][k] for x in batch])
                                for k in batch[0]['tactile_quality']},
            "recorded_haptics": {k: torch.from_numpy(np.stack([x["recorded_haptics"][k] for x in batch]))
                                 for k in batch[0]["recorded_haptics"]},
            "contact_features": None, "contact_risk": None,
            "tactile_loss_mask": loss_mask,
            "force_history_n_real": torch.tensor([x['force_history_n_real'] for x in batch]),
            "tactile_slot_mask": slot_mask,
            "force_contact": torch.tensor([bool(x.get('force_contact', False)) for x in batch]),
            "touch_event": torch.tensor([bool(x.get('touch_event', False)) for x in batch]),
            "tactile_delay_ns": torch.tensor([x['meta']['tactile_delay_ns'] for x in batch], dtype=torch.long),
            **risk,
        }
