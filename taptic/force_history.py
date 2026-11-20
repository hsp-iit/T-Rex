"""Native-rate (~76 Hz) SDK-force history for the TAPTIC tactile VQ-VAE.

One definition shared by VQ-VAE training (tactile_vqvae/data/taptic.py) and the
midtrain loader (qwen_vla/taptic_dataset.py), so the codebook is trained on
exactly the windows the policy will later encode.

Signal: recorded SDK `calc_force` per fingertip, [5 fingers x (Fx,Fy,Fz)] raw
counts, at its native irregular read times. No interpolation, resampling or
averaging. Only native episodes (`tactile_interpolated == False`) qualify:
legacy episodes have 10 Hz force only and are reported as missing tactile.

Track: the episode's usable samples. A sample is excluded (segment -1) when it
precedes the episode's first causal observation (pre-roll), follows its last
frame, or falls inside a recorder pause (frame gap > 150 ms). Segments split at
read gaps > gap_ns, reset/error/drop changes and sequence jumps (taptic.timing),
and around excluded samples. Windows never cross a segment boundary: a short
history is left-padded by repeating the segment's first sample (the same rule
the original T-Rex loader used at episode start), and `n_real` reports how many
samples are real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .timing import build_quality

FRAME_PAUSE_NS = 150_000_000
_SIDE_COLUMNS = ('read_end_ns', 'available_ns', 'force', 'joints', 'sequence',
                 'reset_sequence', 'read_errors', 'dropped', 'interpolated')
N_JOINTS = 12


def read_force_samples(root, ep):
    """Lightweight sidecar read (no 1800-D taxel column). Mirrors Source.tactile."""
    path = Path(root) / f'tactile_raw/episode_{int(ep):06d}_right.parquet'
    names = set(pq.read_schema(path).names)
    sm = pd.read_parquet(path, columns=[c for c in _SIDE_COLUMNS if c in names])
    if 'interpolated' in sm:
        sm = sm[~sm.interpolated.to_numpy(bool)]
    sm = sm.sort_values('read_end_ns').reset_index(drop=True)
    t = sm.read_end_ns.to_numpy(np.int64)
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'episode {ep}: nonmonotonic tactile clock')
    available = np.maximum(t, sm.available_ns.to_numpy(np.int64)) if 'available_ns' in sm else t
    return sm, available


def build_force_track(sm, available, frame_ns, gap_ns=50_000_000):
    """Usable native force samples of one episode.

    Returns dict with force [N,5,3] float32 (raw counts), read_ns, available_ns,
    segment [N] int64 (-1 = excluded), and joints [N,12] float64 (measured XHand
    joint positions, rad, from the same EtherCAT read as the force) or None when
    the sidecar has no `joints` column. The joint stream shares the force rows,
    timestamps and segmentation; it is never resampled.
    """
    frame_ns = np.asarray(frame_ns, np.int64)
    q = build_quality(sm, available, True, gap_ns)
    read, avail = q['read_end_ns'], q['available_ns']
    n = len(read)
    force = (np.stack(sm.force.to_numpy()).astype(np.float32).reshape(n, 5, 3)
             if n else np.empty((0, 5, 3), np.float32))
    if not np.isfinite(force).all():
        raise ValueError('nonfinite SDK force')
    joints = None
    if 'joints' in sm:
        joints = (np.stack(sm.joints.to_numpy()).astype(np.float64).reshape(n, N_JOINTS)
                  if n else np.empty((0, N_JOINTS), np.float64))
        if not np.isfinite(joints).all():
            raise ValueError('nonfinite measured joints')
    initial = np.flatnonzero((read <= frame_ns[0]) & (avail <= frame_ns[0]))
    lower = int(initial[-1]) if len(initial) else 0
    excluded = (np.arange(n) < lower) | (read > frame_ns[-1])
    for i in np.flatnonzero(np.diff(frame_ns) > FRAME_PAUSE_NS):
        excluded |= (read > frame_ns[i]) & (read < frame_ns[i + 1])
        # As at episode start, the latest causal sample at the resuming frame
        # stays as that segment's initial observation.
        resume = np.flatnonzero((read <= frame_ns[i + 1]) & (avail <= frame_ns[i + 1]))
        if len(resume):
            excluded[resume[-1]] = False
    breaks = q['break_before'].copy()
    if n > 1:
        breaks[1:] |= excluded[:-1]
    segment = np.cumsum(breaks).astype(np.int64)
    segment[excluded] = -1
    return {'force': force, 'read_ns': read.copy(), 'available_ns': avail.copy(),
            'segment': segment, 'joints': joints}


def window_indices(segment, end, window):
    """Sample indices [window] ending at `end`, left-padded inside its segment."""
    seg = segment[end]
    if seg < 0:
        raise ValueError('window end is an excluded sample')
    start = end
    while start > 0 and end - start + 1 < window and segment[start - 1] == seg:
        start -= 1
    idx = np.arange(start, end + 1)
    n_real = len(idx)
    if n_real < window:
        idx = np.concatenate([np.full(window - n_real, start), idx])
    return idx, n_real


def history_at(track, tick_ns, window, stale_ns):
    """Causal force history for an observation tick.

    Returns (history [window,5,3] float32 raw, valid bool, n_real int, age_ns).
    Invalid (zeros) when no usable sample is available at the tick, the latest
    one is excluded, or it is older than `stale_ns`.
    """
    read, avail, seg = track['read_ns'], track['available_ns'], track['segment']
    cand = np.flatnonzero((read <= tick_ns) & (avail <= tick_ns))
    if not len(cand) or seg[cand[-1]] < 0:
        return np.zeros((window, 5, 3), np.float32), False, 0, -1
    end = int(cand[-1])
    age = int(tick_ns - read[end])
    if age > stale_ns:
        return np.zeros((window, 5, 3), np.float32), False, 0, age
    idx, n_real = window_indices(seg, end, window)
    return track['force'][idx].copy(), True, n_real, age


def training_window_ends(segment, stride=1, window=16, pad_short=True):
    """Window end indices for VQ-VAE training (stride in native samples).

    With pad_short, segment starts yield left-padded windows too, matching the
    histories the policy sees right after an episode start or sensor reset.
    """
    ends = []
    for s in np.unique(segment[segment >= 0]):
        members = np.flatnonzero(segment == s)
        first = 0 if pad_short else window - 1
        ends.extend(members[first::stride].tolist())
    return np.asarray(sorted(ends), np.int64)
