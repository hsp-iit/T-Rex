"""Causal contact-risk scalar for TAPTIC (Panda + RobotEra XHand).

Signals (no calibration to newtons is assumed anywhere):
  * SDK aggregate fingertip force F[i, f, xyz], raw SDK counts, five fingers,
    recorded at its native irregular read times (~76.4 reads/s).
  * Measured XHand joint positions q[i, 12] (rad). On native recordings they come
    from the SAME EtherCAT read as the force (sidecar column `joints`): a genuine
    ~76 Hz measured stream (0% repeated vectors; the frame sample equals
    observation.state[7:19] exactly). Legacy recordings only have 10 Hz frames.

Definitions at a physical tick t (integer CLOCK_MONOTONIC ns). Let i* be the
latest usable sample with read_ns <= t and available_ns <= t (nothing later is
ever read). Let m[i, f] = ||F[i, f, :]||_2.

  M      = max_f clip(m[i*, f] / sM[f], 0, 1)
  u_f(t) = sum_{i : t - W < read_i <= t, i <= i*} ||F[i,f,:] - F[i-1,f,:]||_2 / W    (counts/s)
           (only consecutive same-segment pairs contribute)              rate_mode='vector' (default)
  R_W    = max_f clip(u_f(t) / sR[f], 0, 1)

Vector variation counts force increases, DECREASES (release, loss of grip, the
pipette squeeze-release cycle) and direction changes at constant magnitude
(shear / slip candidates); it is >= the magnitude variation (triangle inequality).
rate_mode='positive' keeps the previous definition, sum max(m[i,f] - m[i-1,f], 0) / W
(increases of |F| only), for ablation. The positive (R_pos) and negative (R_neg)
magnitude variations are always computed and logged, each with its own scale.
  qdot   = (q[i*] - q[k]) / (read_i* - read_k),  k = latest same-segment sample
           with read_k <= read_i* - Wv     (causal secant on actual timestamps)
  V      = clip(||qdot||_2 / sV, 0, 1)
  r      = clip(a*M + b*R_W + c*M*V, 0, 1),     (a, b, c) = (0.25, 0.50, 0.25)

sM, sR, sV are train-only q99 scales (fit_risk_stats). W = 100 ms (R100) and
Wv = 100 ms by default (a 13 ms single-read difference is dominated by the
~0.5 mrad encoder quantization). The coefficients are fixed experimental
hyperparameters, not learned and not a probability of danger.

Validity (never confused with zero risk):
  M_valid : a usable sample exists and its read age <= stale_ms (50 ms);
  R_valid : M_valid and the sample's segment covers the whole window (t-W, t];
  V_valid : a fresh joint sample exists and a same-segment secant base exists;
  risk_valid = M_valid & R_valid & V_valid. Invalid entries carry 0 placeholders
  plus their masks. Legacy (10 Hz, nominal clock) episodes are always invalid.

The same `risk_features` core is used offline (dataset, fitting) and online
(`OnlineRiskExtractor`), so training and inference compute identical numbers.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .force_history import (FRAME_PAUSE_NS, N_JOINTS, build_force_track, history_at,
                            read_force_samples)

RISK_VERSION = 2          # 2: R = vector variation by default (+ R_pos / R_neg logged)
DEFAULT_COEFFICIENTS = (0.25, 0.50, 0.25)
JOINT_SOURCES = ('sidecar', 'state10hz')
RATE_MODES = ('vector', 'positive')
RATE_KINDS = ('vector', 'positive', 'negative')
N_FINGERS = 5


@dataclass(frozen=True)
class RiskConfig:
    """Hyperparameters of the risk scalar (not fitted)."""
    coefficients: tuple = DEFAULT_COEFFICIENTS     # (M, R_W, M*V)
    rate_window_ms: int = 100
    velocity_window_ms: int = 100
    stale_ms: float = 50.
    joint_source: str = 'sidecar'                 # sidecar: native ~76 Hz measured joints
    joint_stale_ms: Optional[float] = None         # default 50 (sidecar) / 150 (state10hz)
    rate_mode: str = 'vector'                      # R: vector variation | positive |F| increases

    def __post_init__(self):
        if self.rate_mode not in RATE_MODES:
            raise ValueError(f'rate_mode must be one of {RATE_MODES}')
        c = tuple(float(x) for x in self.coefficients)
        if len(c) != 3 or not all(math.isfinite(x) and x >= 0 for x in c):
            raise ValueError('risk coefficients must be three finite non-negative numbers')
        object.__setattr__(self, 'coefficients', c)
        if self.joint_source not in JOINT_SOURCES:
            raise ValueError(f'joint_source must be one of {JOINT_SOURCES}')
        for k in ('rate_window_ms', 'velocity_window_ms', 'stale_ms'):
            if not getattr(self, k) > 0:
                raise ValueError(f'{k} must be positive')
        if self.joint_stale_ms is None:
            object.__setattr__(self, 'joint_stale_ms', 50. if self.joint_source == 'sidecar' else 150.)
        if not self.joint_stale_ms > 0:
            raise ValueError('joint_stale_ms must be positive')

    rate_window_ns = property(lambda self: int(round(self.rate_window_ms * 1e6)))
    velocity_window_ns = property(lambda self: int(round(self.velocity_window_ms * 1e6)))
    stale_ns = property(lambda self: int(round(self.stale_ms * 1e6)))
    joint_stale_ns = property(lambda self: int(round(self.joint_stale_ms * 1e6)))

    def to_dict(self):
        return {'version': RISK_VERSION, 'coefficients': list(self.coefficients),
                'rate_window_ms': self.rate_window_ms, 'velocity_window_ms': self.velocity_window_ms,
                'stale_ms': self.stale_ms, 'joint_source': self.joint_source,
                'joint_stale_ms': self.joint_stale_ms, 'rate_mode': self.rate_mode,
                'formula': 'r = clip(a*M + b*R + c*M*V, 0, 1)'}

    @classmethod
    def from_dict(cls, d):
        if d.get('version', RISK_VERSION) != RISK_VERSION:
            raise ValueError('unsupported risk definition version')
        return cls(tuple(d['coefficients']), d['rate_window_ms'], d['velocity_window_ms'],
                   d['stale_ms'], d['joint_source'], d.get('joint_stale_ms'), d['rate_mode'])

    @classmethod
    def from_args(cls, args):
        return cls(tuple(getattr(args, 'risk_coefficients', DEFAULT_COEFFICIENTS)),
                   int(getattr(args, 'risk_rate_window_ms', 100)),
                   int(getattr(args, 'risk_velocity_window_ms', 100)),
                   float(getattr(args, 'tactile_stale_ms', 50.)),
                   getattr(args, 'risk_joint_source', 'sidecar'),
                   rate_mode=getattr(args, 'risk_rate_mode', 'vector'))


@dataclass(frozen=True)
class RiskStats:
    """Frozen train-only normalization scales (+ provenance, gate initialization)."""
    magnitude_scale: tuple          # [5] SDK counts
    rate_scale: tuple               # [5] SDK counts / s, for rate_window_ms
    speed_scale: float              # rad / s
    rate_window_ms: int
    velocity_window_ms: int
    joint_source: str
    quantile: float = .99
    source_identity: str = ''
    train_episodes: tuple = ()
    gate_thresholds: Optional[tuple] = None
    gate_quantiles: Optional[tuple] = None
    extra: dict = field(default_factory=dict, compare=False, repr=False)
    rate_mode: str = 'vector'
    rate_scales: Optional[dict] = None   # {'vector','positive','negative'}: [5] each (logging/ablation)

    def __post_init__(self):
        if self.rate_mode not in RATE_MODES:
            raise ValueError(f'rate_mode must be one of {RATE_MODES}')
        if self.rate_scales is not None:
            rs = {k: tuple(float(x) for x in self.rate_scales[k]) for k in RATE_KINDS}
            if any(len(v) != N_FINGERS or not all(math.isfinite(x) and x > 0 for x in v) for v in rs.values()):
                raise ValueError('rate_scales must hold positive finite [5] scales for vector/positive/negative')
            if not np.allclose(rs[self.rate_mode], self.rate_scale):
                raise ValueError('rate_scale must equal rate_scales[rate_mode]')
            object.__setattr__(self, 'rate_scales', rs)
        m = np.asarray(self.magnitude_scale, float)
        r = np.asarray(self.rate_scale, float)
        if (m.shape != (N_FINGERS,) or r.shape != (N_FINGERS,) or not np.isfinite([*m, *r, self.speed_scale]).all()
                or (m <= 0).any() or (r <= 0).any() or not self.speed_scale > 0):
            raise ValueError('risk scales must be positive finite: magnitude[5], rate[5], speed')
        object.__setattr__(self, 'magnitude_scale', tuple(m.tolist()))
        object.__setattr__(self, 'rate_scale', tuple(r.tolist()))
        object.__setattr__(self, 'train_episodes', tuple(int(e) for e in self.train_episodes))
        if self.gate_thresholds is not None:
            t = tuple(float(x) for x in self.gate_thresholds)
            if len(t) != 2 or not 0 <= t[0] < t[1] <= 1:
                raise ValueError('gate thresholds must satisfy 0 <= lo < hi <= 1')
            object.__setattr__(self, 'gate_thresholds', t)

    def check_config(self, cfg: RiskConfig):
        if (self.rate_window_ms, self.velocity_window_ms, self.joint_source, self.rate_mode) != (
                cfg.rate_window_ms, cfg.velocity_window_ms, cfg.joint_source, cfg.rate_mode):
            raise ValueError('risk statistics were fitted for different windows / joint source / R mode '
                             f'({self.rate_window_ms}, {self.velocity_window_ms}, {self.joint_source}, '
                             f'{self.rate_mode})')

    def check_source(self, source, train_episodes):
        """Same dataset revision; fitted on (native) training episodes of this split only."""
        if self.source_identity != source.identity:
            raise ValueError('risk statistics belong to a different dataset')
        allowed = set(map(int, train_episodes))
        e = source.episodes
        bad = [i for i in self.train_episodes if i not in allowed or e.loc[i, 'suggested_split'] != 'train'
               or bool(e.loc[i, 'tactile_interpolated'])]
        if not self.train_episodes or bad:
            raise ValueError(f'risk statistics must be fitted on native training episodes only (bad: {bad[:5]})')

    def to_dict(self):
        return {'version': RISK_VERSION, 'magnitude_scale': list(self.magnitude_scale),
                'rate_scale': list(self.rate_scale), 'speed_scale': self.speed_scale,
                'rate_window_ms': self.rate_window_ms, 'velocity_window_ms': self.velocity_window_ms,
                'joint_source': self.joint_source, 'quantile': self.quantile,
                'source_identity': self.source_identity, 'train_episodes': list(self.train_episodes),
                'gate_thresholds': list(self.gate_thresholds) if self.gate_thresholds else None,
                'gate_quantiles': list(self.gate_quantiles) if self.gate_quantiles else None,
                'rate_mode': self.rate_mode,
                'rate_scales': {k: list(v) for k, v in self.rate_scales.items()} if self.rate_scales else None,
                **self.extra}

    @classmethod
    def from_dict(cls, d):
        if d.get('version') != RISK_VERSION:
            raise ValueError('unsupported risk statistics version')
        known = {'version', 'magnitude_scale', 'rate_scale', 'speed_scale', 'rate_window_ms',
                 'velocity_window_ms', 'joint_source', 'quantile', 'source_identity',
                 'train_episodes', 'gate_thresholds', 'gate_quantiles', 'rate_mode', 'rate_scales'}
        return cls(d['magnitude_scale'], d['rate_scale'], float(d['speed_scale']),
                   int(d['rate_window_ms']), int(d['velocity_window_ms']), d['joint_source'],
                   float(d.get('quantile', .99)), d.get('source_identity', ''),
                   tuple(d.get('train_episodes', ())),
                   tuple(d['gate_thresholds']) if d.get('gate_thresholds') else None,
                   tuple(d['gate_quantiles']) if d.get('gate_quantiles') else None,
                   {k: v for k, v in d.items() if k not in known}, d['rate_mode'], d.get('rate_scales'))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def unit(cls, cfg: RiskConfig):
        """Scales of 1: features then carry raw magnitudes/rates/speeds (fitting only)."""
        return cls((1.,) * 5, (1.,) * 5, 1., cfg.rate_window_ms, cfg.velocity_window_ms, cfg.joint_source,
                   rate_mode=cfg.rate_mode, rate_scales={k: (1.,) * 5 for k in RATE_KINDS})


def combine_risk(M, R, V, coefficients=DEFAULT_COEFFICIENTS):
    a, b, c = coefficients
    return np.clip(a * np.asarray(M) + b * np.asarray(R) + c * np.asarray(M) * np.asarray(V), 0., 1.)


# ── shared causal core ───────────────────────────────────────────────────────

def _segment_first(segment):
    """Index of the first sample of each sample's contiguous segment."""
    n = len(segment)
    pair = np.zeros(n, bool)
    if n > 1:
        pair[1:] = (segment[1:] == segment[:-1]) & (segment[1:] >= 0)
    first = np.maximum.accumulate(np.where(~pair, np.arange(n), 0)) if n else np.zeros(0, np.int64)
    return pair, first


def _derived(track):
    d = track.get('_risk_derived')
    if d is None:
        seg = np.asarray(track['segment'], np.int64)
        force = np.asarray(track['force'], np.float64)
        mag = np.linalg.norm(force, axis=2)                                         # [N,5]
        pair, first = _segment_first(seg)
        inc = {k: np.zeros_like(mag) for k in RATE_KINDS}
        if len(mag) > 1:
            dm = (mag[1:] - mag[:-1]) * pair[1:, None]
            inc['positive'][1:] = np.maximum(dm, 0.)
            inc['negative'][1:] = np.maximum(-dm, 0.)
            inc['vector'][1:] = np.linalg.norm(force[1:] - force[:-1], axis=2) * pair[1:, None]
        avail = np.asarray(track['available_ns'], np.int64)
        d = {'mag': mag, 'cum': {k: np.cumsum(v, axis=0) for k, v in inc.items()}, 'first': first,
             'monotone': bool(np.all(np.diff(avail) >= 0)) if len(avail) else True}
        track['_risk_derived'] = d
    return d


def _latest(read, avail, ticks, monotone):
    """Latest index with read <= t and available <= t (-1 if none), per tick."""
    if monotone:
        return np.searchsorted(avail, ticks, side='right') - 1                    # avail >= read
    out = np.full(len(ticks), -1, np.int64)
    for j, t in enumerate(ticks):
        c = np.flatnonzero((read <= t) & (avail <= t))
        out[j] = c[-1] if len(c) else -1
    return out


def joint_track_from_frames(frame_ns, state):
    """10 Hz fallback joint stream: measured state[7:19] at frame observation times.

    Uses the observation time as read time (the SDK read precedes it by a few tens
    of ms on native recordings, so this ages are lower bounds). Segments split at
    recorder pauses (> 150 ms). Never interpolated."""
    t = np.asarray(frame_ns, np.int64)
    seg = np.r_[0, np.cumsum(np.diff(t) > FRAME_PAUSE_NS)].astype(np.int64)
    return {'read_ns': t, 'available_ns': t.copy(), 'segment': seg,
            'q': np.asarray(state, np.float64)[:, 7:19].copy()}


def joint_track_from_force_track(track):
    if track.get('joints') is None:
        return None
    return {'read_ns': track['read_ns'], 'available_ns': track['available_ns'],
            'segment': track['segment'], 'q': track['joints']}


def risk_features(track, ticks, stats: RiskStats, cfg: RiskConfig, joint_track=None):
    """Causal features at each tick (vectorized). See module docstring.

    track       : build_force_track() dict (force, read_ns, available_ns, segment[, joints]),
                  or None (no native tactile stream -> everything invalid).
    ticks       : int64 ns, any order.
    joint_track : dict(read_ns, available_ns, segment, q); default = the force
                  track's own measured joints (cfg.joint_source == 'sidecar').
    Returns a dict of numpy arrays with leading dim len(ticks).
    """
    stats.check_config(cfg)
    ticks = np.atleast_1d(np.asarray(ticks, np.int64))
    T = len(ticks)
    out = {k: np.zeros(T) for k in ('risk', 'M', 'R', 'V', 'joint_speed')}
    out.update({k: np.zeros(T, bool) for k in ('risk_valid', 'M_valid', 'R_valid', 'V_valid')})
    out.update({k: np.full(T, -1, np.int64) for k in ('force_age_ns', 'joint_age_ns', 'velocity_span_ns',
                                                      'source_index')})
    out['magnitude'] = np.zeros((T, N_FINGERS))
    out['rate'] = np.zeros((T, N_FINGERS))
    for k in RATE_KINDS:
        out['rate_' + k] = np.zeros((T, N_FINGERS))
    out['R_pos'] = np.zeros(T)
    out['R_neg'] = np.zeros(T)
    if track is None or not len(track['read_ns']):
        return out
    d = _derived(track)
    read, avail, seg = track['read_ns'], track['available_ns'], track['segment']
    end = _latest(read, avail, ticks, d['monotone'])
    has = end >= 0
    e = np.where(has, end, 0)
    age = ticks - read[e]
    fresh = has & (seg[e] >= 0) & (age >= 0) & (age <= cfg.stale_ns)
    out['source_index'] = np.where(has, end, -1)
    out['force_age_ns'] = np.where(has, age, -1)
    mag = d['mag'][e] * fresh[:, None]
    out['magnitude'] = mag
    out['M'] = np.where(fresh, np.clip(mag / np.asarray(stats.magnitude_scale), 0, 1).max(1), 0.)
    out['M_valid'] = fresh
    # R_W: force variation over (t - W, t], full same-segment window. All three
    # kinds (vector, positive and negative |F| changes) share the same window rule.
    start = ticks - cfg.rate_window_ns
    s = np.searchsorted(read, start, side='right')                    # first read > t - W
    full = fresh & (read[d['first'][e]] <= start)
    for k in RATE_KINDS:
        cum = d['cum'][k]
        before = cum[np.maximum(s - 1, 0)] * (s > 0)[:, None]      # increments up to s-1
        wsum = np.where((s <= e)[:, None], cum[e] - before, 0.)
        out['rate_' + k] = wsum / (cfg.rate_window_ns / 1e9) * full[:, None]
    rate = out['rate_' + cfg.rate_mode]
    out['rate'] = rate
    out['R'] = np.where(full, np.clip(rate / np.asarray(stats.rate_scale), 0, 1).max(1), 0.)
    out['R_valid'] = full
    if stats.rate_scales is not None:
        for key, kind in (('R_pos', 'positive'), ('R_neg', 'negative')):
            out[key] = np.where(full, np.clip(out['rate_' + kind] / np.asarray(stats.rate_scales[kind]),
                                              0, 1).max(1), 0.)
    # V: causal secant of the measured joints on actual timestamps.
    jt = joint_track if joint_track is not None else joint_track_from_force_track(track)
    if jt is not None and len(jt['read_ns']):
        jr, ja, js, q = jt['read_ns'], jt['available_ns'], jt['segment'], jt['q']
        if joint_track is None:                                     # same rows as the force
            jend, jfirst = end, d['first']
        else:
            jmono = bool(np.all(np.diff(ja) >= 0))
            jend = _latest(jr, ja, ticks, jmono)
            jfirst = _segment_first(js)[1]
        jhas = jend >= 0
        je = np.where(jhas, jend, 0)
        jage = ticks - jr[je]
        jfresh = jhas & (js[je] >= 0) & (jage >= 0) & (jage <= cfg.joint_stale_ns)
        k = np.searchsorted(jr, jr[je] - cfg.velocity_window_ns, side='right') - 1
        kv = jfresh & (k >= 0) & (k >= jfirst[je])
        kk = np.where(kv, k, 0)
        span = jr[je] - jr[kk]
        speed = np.where(kv, np.linalg.norm(q[je] - q[kk], axis=1) / np.maximum(span, 1) * 1e9, 0.)
        out['joint_age_ns'] = np.where(jhas, jage, -1)
        out['velocity_span_ns'] = np.where(kv, span, -1)
        out['joint_speed'] = speed
        out['V'] = np.where(kv, np.clip(speed / stats.speed_scale, 0, 1), 0.)
        out['V_valid'] = kv
    valid = out['M_valid'] & out['R_valid'] & out['V_valid']
    out['risk_valid'] = valid
    out['risk'] = np.where(valid, combine_risk(out['M'], out['R'], out['V'], cfg.coefficients), 0.)
    return out


SCALAR_KEYS = ('risk', 'risk_valid', 'M', 'R', 'V', 'M_valid', 'R_valid', 'V_valid',
               'force_age_ns', 'joint_age_ns', 'velocity_span_ns', 'joint_speed', 'R_pos', 'R_neg')


def features_at(feats, j):
    """One tick of a vectorized feature dict as plain Python values (logging)."""
    rec = {k: (bool(feats[k][j]) if feats[k].dtype == bool else
               int(feats[k][j]) if feats[k].dtype.kind == 'i' else float(feats[k][j])) for k in SCALAR_KEYS}
    rec['magnitude'] = feats['magnitude'][j].tolist()
    rec['rate'] = feats['rate'][j].tolist()
    return rec


# ── train-only fitting ───────────────────────────────────────────────────────

def _episode_tracks(source, ep, gap_ms, cfg):
    sm, available = read_force_samples(source.root, ep)
    frames = source.clocks(ep)
    track = build_force_track(sm, available, frames, int(gap_ms * 1e6))
    jt = None
    if cfg.joint_source == 'state10hz':
        s = np.stack(source.frames(ep)['observation.state']).astype(np.float64)
        jt = joint_track_from_frames(frames, s)
    return track, jt


def fit_risk_stats(source, train_episodes, cfg: RiskConfig = RiskConfig(), gap_ms=50., quantile=.99,
                   gate_quantiles=(.5, .85), max_episodes=0, log=print):
    """Train-only q99 scales, evaluated at every usable native read of every native
    training episode; then the r distribution and quantile gate thresholds."""
    e = source.episodes
    bad = [int(i) for i in train_episodes if e.loc[i, 'suggested_split'] != 'train']
    if bad:
        raise ValueError(f'risk statistics must use training episodes only (got {bad[:5]})')
    ids = sorted(int(i) for i in train_episodes if not bool(e.loc[i, 'tactile_interpolated']))
    if max_episodes:
        ids = ids[:max_episodes]
    if not ids:
        raise ValueError('no native training episodes')
    unit = RiskStats.unit(cfg)
    rows = {k: [] for k in ('magnitude', 'rate_vector', 'rate_positive', 'rate_negative', 'joint_speed',
                            'M_valid', 'R_valid', 'V_valid')}
    task_of = []
    for n, ep in enumerate(ids):
        track, jt = _episode_tracks(source, ep, gap_ms, cfg)
        ticks = track['available_ns'][track['segment'] >= 0]
        f = risk_features(track, ticks, unit, cfg, jt)
        for k in rows:
            rows[k].append(f[k])
        task_of.append(np.full(len(ticks), str(e.loc[ep, 'source_task']), dtype=object))
        if log and (n + 1) % 100 == 0:
            log(f'[risk] {n + 1}/{len(ids)} native training episodes')
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    tasks = np.concatenate(task_of)
    mv, rv, vv = cat['M_valid'], cat['R_valid'], cat['V_valid']
    sM = np.maximum(np.quantile(cat['magnitude'][mv], quantile, axis=0), 1.)
    scales = {}
    for kind in RATE_KINDS:                    # q99 of the nonzero window values, per finger
        sc = []
        for f in range(N_FINGERS):
            pos = cat['rate_' + kind][rv, f]
            pos = pos[pos > 0]
            sc.append(max(float(np.quantile(pos, quantile)), 1.) if len(pos) else 1.)
        scales[kind] = tuple(sc)
    sR = scales[cfg.rate_mode]
    sV = max(float(np.quantile(cat['joint_speed'][vv], quantile)), 1e-3)
    M = np.clip(cat['magnitude'] / sM, 0, 1).max(1)
    R = np.clip(cat['rate_' + cfg.rate_mode] / np.asarray(sR), 0, 1).max(1)
    Rk = {k: np.clip(cat['rate_' + k] / np.asarray(scales[k]), 0, 1).max(1) for k in ('positive', 'negative')}
    V = np.clip(cat['joint_speed'] / sV, 0, 1)
    valid = mv & rv & vv
    r = combine_risk(M, R, V, cfg.coefficients)[valid]
    thr = tuple(float(x) for x in np.quantile(r, gate_quantiles))
    if not thr[0] < thr[1]:
        raise ValueError(f'degenerate gate thresholds {thr}; choose other quantiles')
    qs = [0, .1, .25, .5, .75, .9, .95, .99, 1]
    per_task = {}
    for t in sorted(set(tasks.tolist())):
        sel = (tasks == t) & valid
        rt = combine_risk(M[sel], R[sel], V[sel], cfg.coefficients)
        per_task[t] = {'ticks': int((tasks == t).sum()), 'risk_valid_fraction': float(sel.sum() / max((tasks == t).sum(), 1)),
                       'r_quantiles': dict(zip(map(str, qs), np.quantile(rt, qs).round(4).tolist())) if len(rt) else None,
                       'split_occupancy_at_thresholds': np.bincount(np.searchsorted(thr, rt, side='right'),
                                                                    minlength=3).tolist() if len(rt) else None}
    extra = {'definition': cfg.to_dict(), 'gap_ms': gap_ms, 'n_episodes': len(ids),
             'n_ticks': int(len(valid)), 'evaluated_at': 'every usable native read (tick = available_ns)',
             'valid_fraction': {'M': float(mv.mean()), 'R': float(rv.mean()), 'V': float(vv.mean()),
                                'risk': float(valid.mean())},
             'distribution': {name: dict(zip(map(str, qs), np.quantile(x, qs).round(4).tolist()))
                              for name, x in (('M', M[mv]), ('R', R[rv]), ('R_pos', Rk['positive'][rv]),
                                              ('R_neg', Rk['negative'][rv]), ('V', V[vv]), ('r', r))},
             'per_task': per_task,
             'scale_definitions': {'magnitude_scale': f'per-finger q{quantile} of |F| over usable reads',
                                   'rate_scale': f'per-finger q{quantile} of NONZERO window values of the '
                                                 f'{cfg.rate_mode} variation rate (rate_scales: every kind)',
                                   'speed_scale': f'q{quantile} of ||qdot||_2 over V-valid reads'}}
    return RiskStats(tuple(sM.tolist()), tuple(sR), sV, cfg.rate_window_ms, cfg.velocity_window_ms,
                     cfg.joint_source, quantile, source.identity, tuple(ids), thr, tuple(gate_quantiles), extra,
                     rate_mode=cfg.rate_mode, rate_scales=scales)


# ── online (deployment / replay) ─────────────────────────────────────────────

class OnlineRiskExtractor:
    """Same features as `risk_features`, from native samples arriving online.

    Samples are processed in availability order at each `at(tick)`; reads not
    newer than the last accepted one are discarded (late), duplicates are
    counted. Segments split with the offline rules (taptic.timing.build_quality):
    read gap > gap_ms, reset/read-error/drop counter change, sequence jump.
    Only a bounded recent buffer is kept (it never truncates a feature window).
    """

    def __init__(self, stats: RiskStats, cfg: RiskConfig = RiskConfig(), gap_ms=50., vq_window=16,
                 keep_ms=None, min_keep=None):
        stats.check_config(cfg)
        if cfg.joint_source != 'sidecar':
            raise ValueError('online extractor reads joints from the native sensor stream')
        self.stats, self.cfg = stats, cfg
        self.gap_ns, self.vq_window = int(gap_ms * 1e6), int(vq_window)
        self.keep_ns = int((keep_ms if keep_ms is not None else
                            max(cfg.rate_window_ms, cfg.velocity_window_ms) + cfg.stale_ms + 150) * 1e6)
        self.min_keep = int(min_keep if min_keep is not None else 2 * vq_window + 2)
        self.start_episode()

    def start_episode(self):
        self.pending, self.rows, self.seen = [], [], set()
        self.segment, self.last, self.last_tick, self._order = 0, None, None, 0
        self.late_discarded = self.duplicates = 0

    def ingest(self, samples: Iterable[dict]):
        """samples: dicts with read_ns, available_ns, force[5,3], joints[12], sequence,
        reset_sequence, read_errors, dropped (counters default 0)."""
        for s in samples:
            key = (int(s['read_ns']), int(s.get('sequence', 0)))
            if key in self.seen:
                self.duplicates += 1
                continue
            self.seen.add(key)
            if int(s['available_ns']) < int(s['read_ns']):
                raise ValueError('availability precedes read completion')
            heapq.heappush(self.pending, (int(s['available_ns']), int(s['read_ns']), self._order, s))
            self._order += 1

    def _advance(self, tick):
        while self.pending and self.pending[0][0] <= tick:
            avail, read, _, s = heapq.heappop(self.pending)
            if self.last is not None and read <= self.last['read_ns']:
                self.late_discarded += 1
                continue
            f = np.asarray(s['force'], np.float64).reshape(5, 3)
            q = np.asarray(s['joints'], np.float64).reshape(N_JOINTS)
            counters = tuple(int(s.get(k, 0)) for k in ('reset_sequence', 'read_errors', 'dropped'))
            seq = int(s['sequence']) if 'sequence' in s else None
            p = self.last
            if (p is None or read - p['read_ns'] > self.gap_ns or counters != p['counters']
                    or (seq is not None and p['sequence'] is not None and seq != p['sequence'] + 1)):
                self.segment += 1
            if not (np.isfinite(f).all() and np.isfinite(q).all()):
                raise ValueError('nonfinite native sample')
            row = {'read_ns': read, 'available_ns': avail, 'force': f, 'joints': q, 'segment': self.segment,
                   'counters': counters, 'sequence': seq}
            self.rows.append(row)
            self.last = row
        left = tick - self.keep_ns
        while len(self.rows) > self.min_keep and self.rows[1]['read_ns'] <= left:
            self.rows.pop(0)
        if self.rows:
            oldest = self.rows[0]['read_ns']
            self.seen = {k for k in self.seen if k[0] >= oldest}

    def track(self):
        n = len(self.rows)
        return {'force': np.asarray([r['force'] for r in self.rows], np.float32).reshape(n, 5, 3),
                'joints': np.asarray([r['joints'] for r in self.rows], np.float64).reshape(n, N_JOINTS),
                'read_ns': np.asarray([r['read_ns'] for r in self.rows], np.int64),
                'available_ns': np.asarray([r['available_ns'] for r in self.rows], np.int64),
                'segment': np.asarray([r['segment'] for r in self.rows], np.int64)}

    def at(self, tick_ns):
        tick_ns = int(tick_ns)
        if self.last_tick is not None and tick_ns < self.last_tick:
            raise ValueError('physical ticks must be nondecreasing')
        self.last_tick = tick_ns
        self._advance(tick_ns)
        tr = self.track()
        feats = risk_features(tr, np.array([tick_ns]), self.stats, self.cfg)
        rec = features_at(feats, 0)
        hist, hvalid, n_real, age = history_at(tr, tick_ns, self.vq_window, self.cfg.stale_ns) if len(tr['read_ns']) \
            else (np.zeros((self.vq_window, 5, 3), np.float32), False, 0, -1)
        rec.update(tick_ns=tick_ns, force_history=hist, force_valid=bool(hvalid), force_history_n_real=int(n_real),
                   segment=self.segment, late_discarded=self.late_discarded, duplicates=self.duplicates)
        return rec


def samples_from_sidecar(sm, available):
    """Sidecar DataFrame rows -> online sample dicts (replay / tests)."""
    cols = {k: sm[k].to_numpy() for k in ('sequence', 'reset_sequence', 'read_errors', 'dropped') if k in sm}
    force, joints = sm.force.to_numpy(), sm.joints.to_numpy()
    read = sm.read_end_ns.to_numpy(np.int64)
    return [{'read_ns': int(read[i]), 'available_ns': int(available[i]), 'force': force[i], 'joints': joints[i],
             **{k: int(v[i]) for k, v in cols.items()}} for i in range(len(sm))]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    from .data import Source
    p = argparse.ArgumentParser(description='Fit train-only TAPTIC contact-risk statistics.')
    p.add_argument('cmd', choices=['fit'])
    p.add_argument('--root', required=True)
    p.add_argument('--taptic_stats', required=True, help='training split (same file as the midtrain)')
    p.add_argument('--out', required=True)
    p.add_argument('--rate_window_ms', type=int, default=100)
    p.add_argument('--velocity_window_ms', type=int, default=100)
    p.add_argument('--joint_source', default='sidecar', choices=JOINT_SOURCES)
    p.add_argument('--rate_mode', default='vector', choices=RATE_MODES)
    p.add_argument('--coefficients', type=float, nargs=3, default=list(DEFAULT_COEFFICIENTS))
    p.add_argument('--gate_quantiles', type=float, nargs=2, default=[.5, .85])
    p.add_argument('--max_episodes', type=int, default=0)
    a = p.parse_args()
    src = Source(a.root)
    block = json.loads(Path(a.taptic_stats).read_text())['taptic']
    if block['source_identity'] != src.identity:
        raise ValueError('taptic_stats belong to a different dataset revision')
    cfg = RiskConfig(tuple(a.coefficients), a.rate_window_ms, a.velocity_window_ms, joint_source=a.joint_source,
                     rate_mode=a.rate_mode)
    stats = fit_risk_stats(src, block['train_episodes'], cfg, gate_quantiles=tuple(a.gate_quantiles),
                           max_episodes=a.max_episodes)
    Path(a.out).write_text(json.dumps(stats.to_dict(), indent=2))
    print(json.dumps({k: v for k, v in stats.to_dict().items() if k not in ('per_task', 'train_episodes')}, indent=1))


if __name__ == '__main__':
    main()
