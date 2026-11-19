"""Verified T-Rex-style TAPTIC source, causal observations and train-only stats.

No dependency on LeRobot internals. Uses the v3 episode/video offsets directly.
Ported action math is in actions.py; units and limitations are in docs/taptic.md.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from .actions import chunk_21d
from .timing import build_quality, latest_quality, interval_quality
from .constants import FINGERS, XHAND_JOINTS

CAMERAS = ('observation.images.left', 'observation.images.wrist')


def finite(x, name):
    if not np.isfinite(x).all():
        raise ValueError(f'nonfinite {name}')
    return x


class Source:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.tinfo = json.loads((self.root / 'tactile_raw/info.json').read_text())
        feats = self.info['features']
        expected = ['ee.x', 'ee.y', 'ee.z', 'ee.wx', 'ee.wy', 'ee.wz']
        if (self.info['codebase_version'] != 'v3.0' or self.info['fps'] != 10
                or feats['action']['shape'] != [18] or feats['observation.state']['shape'] != [40]
                or feats['action']['names'] != expected + ['xhand.action.' + j for j in XHAND_JOINTS]
                or feats['observation.state']['names'] != (
                    [f'panda_joint{i}.pos' for i in range(1, 8)] + ['xhand.joint.' + j for j in XHAND_JOINTS]
                    + ['xhand.force.' + f + '.' + a for f in FINGERS for a in 'xyz'] + expected)):
            raise ValueError('unsupported TAPTIC schema; verify conventions before converting')
        order = [f'{f}.taxel_{t:03d}.{a}' for f in FINGERS for t in range(120) for a in 'xyz']
        if self.tinfo['order']['values'] != order:
            raise ValueError('unverified taxel flattening order')
        for cam in CAMERAS:
            if feats[cam]['dtype'] != 'video':
                raise ValueError(f'missing camera {cam}')
        self.episodes = pd.concat([pd.read_parquet(p) for p in sorted(
            (self.root / 'meta/episodes').rglob('*.parquet'))]).set_index('episode_index').sort_index()
        if self.episodes.groupby(['source_task', 'source_episode_index']).suggested_split.nunique().max() != 1:
            raise ValueError('source recording crosses train/val split')
        if not set(self.episodes.suggested_split).issubset({'train', 'val'}):
            raise ValueError('unknown episode split')
        self.table = pads.dataset(self.root / 'data', format='parquet')
        # Content identity for statistics; small compared with raw tactile/video.
        h = hashlib.sha256()
        for folder in ['meta/episodes', 'data']:
            for p in sorted((self.root / folder).rglob('*.parquet')):
                with p.open('rb') as f:
                    for b in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(b)
        h.update((self.root / 'meta/info.json').read_bytes())
        self.identity = h.hexdigest()

    def frames(self, ep, haptics=False):
        cols = ['action', 'observation.state', 'frame_index', 'timestamp']
        if haptics:
            cols += [k for k in self.table.schema.names if k.startswith('observation.haptics.')]
        d = self.table.to_table(columns=cols, filter=pads.field('episode_index') == int(ep)).to_pandas()
        d = d.sort_values('frame_index').reset_index(drop=True)
        n = int(self.episodes.loc[ep, 'length'])
        if len(d) != n or not np.array_equal(d.frame_index, np.arange(n)):
            raise ValueError(f'episode {ep}: inconsistent frame indices')
        if not np.allclose(d.timestamp, np.arange(n) / 10, atol=1e-5):
            raise ValueError(f'episode {ep}: inconsistent nominal timestamps')
        return d

    def clocks(self, ep):
        fr = pd.read_parquet(self.root / f'tactile_raw/episode_{ep:06d}_frames.parquet')
        fr = fr.sort_values('frame_index')
        if not np.array_equal(fr.frame_index, np.arange(int(self.episodes.loc[ep, 'length']))):
            raise ValueError(f'episode {ep}: inconsistent tactile frames')
        t = fr.observation_ns.to_numpy(np.int64)
        if np.any(np.diff(t) <= 0):
            raise ValueError(f'episode {ep}: nonmonotonic frame clock')
        return t

    def anchors(self, ep, times=None):
        t = self.clocks(ep) if times is None else times
        # Full real action windows only: no end padding, no teleoperation pause.
        if len(t) < 16:
            return np.empty(0, np.int64)
        bad = np.r_[0, np.cumsum(np.diff(t) > 150_000_000)]
        starts = np.arange(len(t) - 15)
        return starts[bad[starts + 15] == bad[starts]]

    def tactile(self, ep):
        sm = pd.read_parquet(self.root / f'tactile_raw/episode_{ep:06d}_right.parquet')
        # Synthetic interpolation is never treated as a recorded observation.
        if 'interpolated' in sm:
            sm = sm[~sm.interpolated]
        sm = sm.sort_values('read_end_ns').reset_index(drop=True)
        t = sm.read_end_ns.to_numpy(np.int64)
        if np.any(np.diff(t) <= 0):
            raise ValueError(f'episode {ep}: nonmonotonic tactile clock')
        available = np.maximum(t, sm.available_ns.to_numpy(np.int64)) if 'available_ns' in sm else t
        return sm, available


def tactile_at(sm, available, tick, first_tick, native, *, quality=None, stale_ns=None):
    quality = build_quality(sm, available, native) if quality is None else quality
    current = latest_quality(quality, tick, first_tick, native, stale_ns)
    i = current['source_index']
    if i < 0:
        return np.zeros((5, 120, 3), np.float32), np.zeros((5, 3), np.float32), False, False, 0
    f = finite(np.asarray(sm.iloc[i]['force'], np.float32).reshape(5, 3), 'SDK force')
    raw = finite(np.asarray(sm.iloc[i]['values'], np.float32).reshape(5, 120, 3), 'taxels')
    if not native:
        raw = np.zeros_like(raw)  # placeholders explicitly masked, never recorded taxels
    return raw, f, native and current["fresh"], current["fresh"], int(available[i])


def tactile_stream_between(sm, available, tick, previous_tick, native, *, quality=None, stale_ns=None):
    """All recorded samples becoming available in (previous_tick, tick].

    Preserve native timestamps/rate, taxels, SDK forces and quality counters.
    At the first frame (previous_tick=None), return only the latest causal
    initial observation, excluding pre-roll. No fixed window, interpolation,
    averaging or truncation; native reset boundaries remain explicit.
    """
    read_ns = sm.read_end_ns.to_numpy(np.int64)
    eligible = (available <= tick) & (read_ns <= tick)
    if 'interpolated' in sm:
        eligible &= ~sm.interpolated.to_numpy(bool)
    if previous_tick is None:
        indices = np.flatnonzero(eligible)[-1:]
    else:
        if previous_tick >= tick:
            raise ValueError('tactile interval must have increasing boundaries')
        indices = np.flatnonzero(eligible & (available > previous_tick))
    quality = build_quality(sm, available, native) if quality is None else quality
    rows = sm.iloc[indices]
    n = len(rows)
    force = (np.stack(rows.force).astype(np.float32).reshape(n, 5, 3)
             if n else np.empty((0, 5, 3), np.float32))
    taxels = (np.stack(rows['values']).astype(np.float32).reshape(n, 5, 120, 3)
              if n and native else np.zeros((n, 5, 120, 3), np.float32))
    result = {'sdk_force': finite(force, 'native stream SDK force'),
              'taxels': finite(taxels, 'native stream taxels'),
              'read_end_ns': read_ns[indices].copy(),
              'available_ns': np.asarray(available[indices], np.int64).copy(),
              'taxel_valid': np.full((n, 5), native, dtype=bool),
              'clock_is_nominal': np.full(n, not native, dtype=bool)}
    for key in ('sequence', 'reset_sequence', 'read_errors', 'dropped'):
        result[key] = rows[key].to_numpy(np.int64).copy()
    result.update(interval_quality(sm, quality, indices, tick, native, stale_ns))
    return result


def normalize(x, stats):
    lo, hi = np.asarray(stats['min'], np.float32), np.asarray(stats['max'], np.float32)
    # Reversible affine normalization, including held-out outliers (no clipping).
    return ((np.asarray(x, np.float32) - lo) / (hi - lo) * 2 - 1).astype(np.float32)


def denormalize(x, stats):
    lo, hi = np.asarray(stats['min'], np.float32), np.asarray(stats['max'], np.float32)
    return ((np.asarray(x, np.float32) + 1) * .5 * (hi - lo) + lo).astype(np.float32)


def fit_stats(source, train_episodes=None):
    ids = sorted(map(int, train_episodes if train_episodes is not None else
                     source.episodes.index[source.episodes.suggested_split == 'train']))
    if not ids or any(source.episodes.loc[e, 'suggested_split'] != 'train' for e in ids):
        raise ValueError('statistics must use training episodes only')
    lo, hi = np.full(21, np.inf), np.full(21, -np.inf)
    n = 0
    slo, shi = np.full(21, np.inf), np.full(21, -np.inf)
    # Read core numeric fields once; parquet predicate pushdown cannot avoid
    # repeatedly decoding a large shared row group for every episode.
    core = source.table.to_table(columns=['episode_index', 'frame_index', 'timestamp', 'action', 'observation.state']).to_pandas()
    grouped = core.groupby('episode_index', sort=False)
    for ep in ids:
        d = grouped.get_group(ep).sort_values('frame_index')
        if not np.array_equal(d.frame_index, np.arange(int(source.episodes.loc[ep, 'length']))):
            raise ValueError(f'episode {ep}: inconsistent training frames')
        if not np.allclose(d.timestamp, np.arange(len(d)) / 10, atol=1e-5):
            raise ValueError(f'episode {ep}: inconsistent training timestamps')
        a = finite(np.stack(d.action).astype(np.float32), 'action')
        s = finite(np.stack(d['observation.state']).astype(np.float32), 'state')
        from .actions import rotvec_gd_to_matrix
        rot = rotvec_gd_to_matrix(s[:, 37:40])
        state21 = np.concatenate([s[:, 34:37], rot[:, :, 0], rot[:, :, 1], s[:, 7:19]], axis=-1)
        slo, shi = np.minimum(slo, state21.min(0)), np.maximum(shi, state21.max(0))
        for t in source.anchors(ep):
            c = chunk_21d(s, a, int(t))
            lo, hi = np.minimum(lo, c.min(0)), np.maximum(hi, c.max(0))
            n += 1
    if n == 0:
        raise ValueError('no valid training chunks')
    constant = hi - lo < 1e-6
    hi[constant] = lo[constant] + 1.0
    shi = np.where(shi - slo < 1e-6, slo + 1., shi)
    return {'taptic': {'state': {'min': slo.tolist(), 'max': shi.tolist()}, 'action': {'min': lo.tolist(), 'max': hi.tolist(),
                                 'mask': [True] * 21, 'constant': constant.tolist()},
                       'train_episodes': ids, 'source_identity': source.identity,
                       'num_chunks': n, 'normalization': 'affine_train_minmax_unclipped',
                       'taxel_scale': [128., 128., 255.],
                       'action_layout': 'flange delta xyz m, rotation columns 1/2, absolute hand rad'}}
