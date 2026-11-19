"""TAPTIC native SDK-force windows for the tactile VQ-VAE.

Each item is one [T, 5, 3] window of recorded fingertip SDK force (raw counts),
T consecutive native samples (~76.4 Hz; T=16 ~ 210 ms). Only native episodes:
legacy 10 Hz episodes are excluded, never upsampled. Split = the dataset's
source-recording `suggested_split` (no random episode split, no leakage).
Normalization: per-channel train-only q01/q99 min-max to [-1, 1], same form as
TacF6Stats for the original F6 VQ-VAE, stored with the checkpoint.

Window construction is shared with the midtrain loader (taptic/force_history.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from taptic.data import Source
from taptic.force_history import (build_force_track, read_force_samples,
                                  training_window_ends, window_indices)
from .stats import TacF6Stats

TAPTIC_FORCE_SIGNAL = 'taptic_sdk_force'


def native_episodes(source: Source, split: str) -> List[int]:
    e = source.episodes
    return sorted(map(int, e.index[(e.suggested_split == split) & (~e.tactile_interpolated.astype(bool))]))


def load_tracks(source: Source, episodes: Sequence[int], gap_ms: float = 50.,
                log=print) -> Dict[int, dict]:
    tracks = {}
    for k, ep in enumerate(episodes):
        sm, available = read_force_samples(source.root, ep)
        tracks[ep] = build_force_track(sm, available, source.clocks(ep), int(gap_ms * 1e6))
        if log and (k + 1) % 100 == 0:
            log(f'[taptic-vqvae] loaded {k + 1}/{len(episodes)} native force tracks')
    return tracks


def fit_force_stats(tracks: Dict[int, dict], q_lo=.01, q_hi=.99) -> TacF6Stats:
    """Train-only per-channel quantile range over usable native samples."""
    x = np.concatenate([t['force'][t['segment'] >= 0].reshape(-1, 15) for t in tracks.values()])
    if not len(x):
        raise ValueError('no usable native force samples')
    lo, hi = np.quantile(x, q_lo, axis=0), np.quantile(x, q_hi, axis=0)
    hi = np.where(hi - lo < 1e-6, lo + 1., hi)
    return TacF6Stats(tacf6_min=lo.astype(np.float32), tacf6_max=hi.astype(np.float32),
                      tacf6_mask=np.ones(15, bool))


class TapticForceWindowDataset(Dataset):
    def __init__(self, tracks: Dict[int, dict], stats: TacF6Stats, window: int = 16,
                 stride: int = 4, pad_short: bool = True):
        if stats.dim != 15:
            raise ValueError('TAPTIC force stats must have 15 channels (5 fingers x 3)')
        self.tracks, self.stats, self.window = tracks, stats, int(window)
        self.episodes = sorted(tracks)
        rows = []
        for ep in self.episodes:
            for end in training_window_ends(tracks[ep]['segment'], max(1, int(stride)),
                                            self.window, pad_short):
                rows.append((ep, int(end)))
        if not rows:
            raise ValueError('no TAPTIC force windows')
        self._rows = np.asarray(rows, np.int64)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self):
        return len(self._rows)

    def raw_window(self, i):
        ep, end = map(int, self._rows[i])
        t = self.tracks[ep]
        idx, n_real = window_indices(t['segment'], end, self.window)
        return t['force'][idx], n_real

    def __getitem__(self, i):
        raw, n_real = self.raw_window(i)
        ep, end = map(int, self._rows[i])
        return {'f6': torch.from_numpy(self.stats.normalize(raw).astype(np.float32)),
                'magnitude': torch.tensor(float(np.linalg.norm(raw)), dtype=torch.float32),
                'ep_idx': ep, 'frame': end, 'hand': 0, 'n_real': n_real}

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        return {'f6': torch.stack([b['f6'] for b in batch]),
                'magnitude': torch.stack([b['magnitude'] for b in batch]),
                'ep_idx': torch.tensor([b['ep_idx'] for b in batch]),
                'frame': torch.tensor([b['frame'] for b in batch]),
                'hand': torch.zeros(len(batch), dtype=torch.long),
                'n_real': torch.tensor([b['n_real'] for b in batch])}

    def magnitudes(self, max_items: Optional[int] = None, seed: int = 0) -> np.ndarray:
        n = len(self)
        ids = np.arange(n) if not max_items or n <= max_items else \
            np.random.default_rng(seed).choice(n, max_items, replace=False)
        return np.asarray([np.linalg.norm(self.raw_window(int(i))[0]) for i in ids])


def build_taptic_datasets(root: str, window: int = 16, stride: int = 4, gap_ms: float = 50.,
                          pad_short: bool = True, max_episodes: int = 0, log=print):
    """Train/val datasets from the shipped source-recording split; stats from train only."""
    source = Source(root)
    tr_eps, va_eps = native_episodes(source, 'train'), native_episodes(source, 'val')
    if max_episodes:
        tr_eps, va_eps = tr_eps[:max_episodes], va_eps[:max(1, max_episodes // 8)]
    if not tr_eps or not va_eps:
        raise ValueError('need native train and val episodes')
    log(f'[taptic-vqvae] native episodes: train={len(tr_eps)} val={len(va_eps)} '
        f'(legacy 10 Hz episodes excluded)')
    tr_tracks = load_tracks(source, tr_eps, gap_ms, log)
    stats = fit_force_stats(tr_tracks)
    train = TapticForceWindowDataset(tr_tracks, stats, window, stride, pad_short)
    val = TapticForceWindowDataset(load_tracks(source, va_eps, gap_ms, log), stats,
                                   window, stride, pad_short)
    meta = {'input_signal': TAPTIC_FORCE_SIGNAL, 'source_identity': source.identity,
            'train_episodes': tr_eps, 'val_episodes': va_eps, 'gap_ms': gap_ms,
            'window': window, 'stride': stride, 'pad_short': pad_short,
            'normalization': 'train-only per-channel q01/q99 min-max to [-1,1], clipped',
            'channel_order': 'thumb,index,middle,ring,little x (x,y,z) SDK calc_force counts'}
    return train, val, stats, meta
