"""XHand fingertip-force windows for the tactile VQ-VAE ([T=16, 5, 3]).

Reads the derived TAPTIC dataset written by ``python -m taptic.convert``.
A window is the 16 native-rate (~76.4 Hz, ~209 ms) samples ending at a
sample index, built with exactly the same rule the VLA dataloader uses at
training/inference time (``taptic.tactile.EpisodeTactile.window_indices``):
no pre-roll, no crossing of a tactile re-zero, left-pad with the oldest valid
sample.  Window ends are taken every ``stride`` samples inside the episode
([sample current at frame 0, sample current at the last frame]).

Items match ``F6WindowDataset`` (``f6`` normalized, ``magnitude`` raw L2).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .stats import TacF6Stats


def load_taptic_stats(taptic_dir: str) -> TacF6Stats:
    with open(os.path.join(taptic_dir, "norm_stats.json")) as f:
        blob = json.load(f)
    t = blob[next(iter(blob))]["tactile_f6"]
    return TacF6Stats(tacf6_min=np.asarray(t["q01"], np.float32),
                      tacf6_max=np.asarray(t["q99"], np.float32),
                      tacf6_mask=np.asarray(t["mask"], bool))


def taptic_split(taptic_dir: str) -> Dict[str, List[int]]:
    with open(os.path.join(taptic_dir, "split.json")) as f:
        s = json.load(f)
    return {"train": s["train_merged"], "val": s["val_merged"]}


class XHandWindowDataset(Dataset):
    def __init__(self, taptic_dir: str, episodes: Sequence[int], stats: TacF6Stats,
                 window: int = 16, stride: int = 4):
        from taptic.tactile import EpisodeTactile
        self.window, self.stride, self.stats = int(window), max(1, int(stride)), stats
        self._ep: List[EpisodeTactile] = []
        self._contact: List[np.ndarray] = []
        index = []
        for ep in episodes:
            z = np.load(os.path.join(taptic_dir, "episodes", f"episode_{ep:06d}.npz"))
            et = EpisodeTactile(t_ns=z["tac_t_ns"], force=z["tac_force"].astype(np.float32),
                                reset=z["tac_reset"], start_idx=int(z["tac_start_idx"]))
            last = int(np.searchsorted(et.t_ns, int(z["obs_ns"][-1]), side="right")) - 1
            first = int(z["cur_tac_idx"][0])
            k = len(self._ep)
            self._ep.append(et)
            for end in range(first, last + 1, self.stride):
                index.append((k, end, int(ep)))
        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 3)

    @property
    def num_episodes(self) -> int:
        return len(self._ep)

    def __len__(self) -> int:
        return len(self.index)

    def raw_window(self, i: int) -> np.ndarray:
        k, end, _ = self.index[i]
        et = self._ep[k]
        return et.force[et.window_indices(int(end), self.window)]          # [T,5,3]

    def __getitem__(self, i: int) -> Dict:
        raw = self.raw_window(i)
        k, end, ep = self.index[i]
        return {
            "f6": torch.from_numpy(self.stats.normalize(raw).astype(np.float32)),
            "magnitude": torch.tensor(float(np.linalg.norm(raw)), dtype=torch.float32),
            "ep_idx": int(ep), "frame": int(end), "hand": 0,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        return {
            "f6": torch.stack([b["f6"] for b in batch]),
            "magnitude": torch.stack([b["magnitude"] for b in batch]),
            "ep_idx": torch.tensor([b["ep_idx"] for b in batch]),
            "frame": torch.tensor([b["frame"] for b in batch]),
            "hand": torch.tensor([b["hand"] for b in batch]),
        }


def build_taptic_train_val(taptic_dir: str, window: int = 16, stride: int = 4,
                           stats: Optional[TacF6Stats] = None):
    stats = stats or load_taptic_stats(taptic_dir)
    sp = taptic_split(taptic_dir)
    return (XHandWindowDataset(taptic_dir, sp["train"], stats, window, stride),
            XHandWindowDataset(taptic_dir, sp["val"], stats, window, stride), stats)
