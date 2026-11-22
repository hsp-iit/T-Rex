"""
EgoDex large-scale pretraining script for Qwen3-VL VLA (Stage 1) WITH flare.

Extends train_qwen3vl_pretrain_egodex.py with future visual prediction for the
latent expert, using the same flare mechanism as train_qwen3vl_flare.py.

Key difference from post-training flare: future frames are read from the SAME
ego_view.mp4 video by seeking to frame_t + (k+1)*flare_frame_stride, rather
than from separate image files.

When --use_flare 0, this script behaves identically to the base pretrain script.
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import glob
import json
import math
import re
import shutil
import time
import logging
import argparse
import collections
import contextlib
import functools
import threading
import warnings

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import PIL.Image
from datetime import timedelta

from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from transformers import AutoProcessor, set_seed

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
from qwen_vla.lerobot_dataset import add_lerobot_args

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


# ---------------------------------------------------------------------------
# Tracking-error noise injection: augment robot state with realistic noise
# sampled from the tracking error distribution computed during data generation.
# ---------------------------------------------------------------------------

def _rot6d_to_mat(rot6d):
    """6D rotation (first two columns) → 3×3 rotation matrix."""
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    col3 = np.cross(col1, col2)
    return np.column_stack([col1, col2, col3])


def _arm9d_to_axis_angle(arm_9d):
    """[trans(3), rot6d(6)] → [trans(3), axis_angle(3)]."""
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    """[trans(3), axis_angle(3)] → [trans(3), rot6d(6)]."""
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def tracking_error_dim(action_dim, hand_dim):
    """Tracking-error width: per arm 3 xyz + 3 axis-angle + hand_dim joints."""
    return (action_dim // (9 + hand_dim)) * (6 + hand_dim)


def add_tracking_error_noise(state, te_mean, te_std, action_dim, hand_dim=22):
    """Add tracking-error noise to robot state.

    Per arm the state is [arm 9D | hand hand_dim] and the tracking error is
    [3 xyz + 3 axis-angle + hand_dim]. Examples: Sharpa bimanual
    (action_dim=62, hand_dim=22, te=56D); Panda + XHand single arm
    (action_dim=21, hand_dim=12, te=18D).
      1. Convert arm 9D [trans, rot6d] → [trans, axis_angle]
      2. Sample noise ~ N(te_mean, te_std)
      3. Add noise
      4. Convert back to [trans, rot6d]
    """
    noisy = state.copy()
    arm_w = 9 + hand_dim
    te_w = 6 + hand_dim
    n_arms = action_dim // arm_w
    for arm_idx in range(n_arms):
        offset = arm_idx * arm_w
        arm_9d = state[offset:offset + 9]
        hand = state[offset + 9:offset + arm_w]

        arm_aa = _arm9d_to_axis_angle(arm_9d)  # (6,)

        te_off = arm_idx * te_w
        noise_arm = np.random.normal(te_mean[te_off:te_off + 6],
                                     te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + te_w],
                                      te_std[te_off + 6:te_off + te_w]).astype(np.float32)

        noisy_arm_9d = _axis_angle_to_arm9d(arm_aa + noise_arm)
        noisy[offset:offset + 9] = noisy_arm_9d
        noisy[offset + 9:offset + arm_w] = hand + noise_hand
    return noisy


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.0, num_cycles=0.5):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


class EpisodeGroupedSampler(_DistributedSampler):
    """
    Distributed sampler over one frame order per epoch that does not depend on
    the number of ranks or the batch size, so a run can resume with other ones
    exactly where it stopped.

    Per epoch (seed + epoch) episodes are shuffled and their frames, laid end to
    end, form the epoch's sequence. With shuffle_frames the sequence is also
    shuffled inside consecutive windows of `window` frames (0 = one window, the
    whole epoch), so a batch mixes frames of many episodes (use it when there is
    no per-worker episode cache to exploit, e.g. LeRobot data); without it a
    batch holds consecutive frames of an episode, for per-worker episode caching.

    Global step k takes the N*B frames at positions [s + k*N*B, s + (k+1)*N*B)
    of the epoch's order (N ranks, B = batch_size, s = start_sample); rank r gets
    the r-th B of them. Whatever N and B were, the frames seen after any number
    of steps are a prefix of the order: resuming with start_sample = frames of
    the epoch already consumed continues exactly. The tail that does not fill a
    global batch is dropped. Subclasses DistributedSampler so accelerate/DeepSpeed
    doesn't replace it.

    The window may change when a run resumes in the middle of an epoch
    (resume_order): the frames already taken keep their order, and the rest of
    the epoch is the frames not taken yet, shuffled with the new window, so each
    frame is still taken once per epoch. Checkpoints record the epoch's order
    (order_state), so later resumes rebuild the same one.

    Memory-efficient: builds only this rank's indices (numpy, ~1/N of the epoch)
    and one shuffle window at a time (with window 0 the whole epoch: ~0.6 GB per
    rank while an epoch's order is built), never a full-dataset Python list.
    """
    SHUFFLE_WINDOW = 1 << 22      # also the window of checkpoints that have no order record (older code)

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
                 shuffle=True, seed=0, shuffle_frames=False, window=SHUFFLE_WINDOW):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=True)
        self.batch_size = int(batch_size)
        self.shuffle_frames = shuffle_frames
        # Frames shuffled in windows of this many (0 = whole epoch; None = not shuffled).
        self.window = int(window) if (shuffle and shuffle_frames) else None
        # Order of a resumed epoch that differs from self.window (see resume_order).
        self.order = None
        cum = np.asarray(dataset._cum_frames, dtype=np.int64)
        self._frame_counts = np.diff(cum, prepend=0)       # per episode (unshuffled)
        self._orig_starts = cum - self._frame_counts        # per episode start index
        self.total_frames = int(cum[-1]) if len(cum) else 0
        # Frames of the current epoch already consumed (all ranks together),
        # skipped on the next __iter__ (resume). Zero-cost: skipped indices
        # never reach the workers, so no data is loaded for them.
        self.start_sample = 0

    @property
    def global_batch(self):
        return self.num_replicas * self.batch_size

    def steps(self, start_sample=None):
        """Global steps left in the epoch from start_sample (default: self.start_sample)."""
        s = self.start_sample if start_sample is None else int(start_sample)
        return max(0, (self.total_frames - s) // self.global_batch)

    def __len__(self):
        return self.steps() * self.batch_size

    @staticmethod
    def window_name(W):
        return "(not shuffled)" if W is None else ("the whole epoch" if W == 0 else f"{W} frames")

    def epoch_order(self, epoch=None):
        """(window, switch_at, then) of an epoch: its frames are shuffled in windows of
        `window`; from position switch_at on (None = never), the frames not taken before
        it are shuffled in windows of `then` (0 = the whole rest of the epoch)."""
        epoch = self.epoch if epoch is None else epoch
        o = self.order
        if o is not None and o["epoch"] == epoch:
            return o["window"], o.get("switch_at"), o.get("then")
        return self.window, None, None

    def order_state(self, epoch=None):
        """The order of an epoch, for training_state.json ("data_order")."""
        epoch = self.epoch if epoch is None else epoch
        W, S, W1 = self.epoch_order(epoch)
        rec = {"epoch": int(epoch), "window": W}
        if S is not None:
            rec.update(switch_at=int(S), then=W1)
        return rec

    def resume_order(self, saved, epoch, start):
        """Set the order of `epoch` for a run that resumes at its frame `start`.
        `saved`: the checkpoint's "data_order" (None: a checkpoint of older code, which
        shuffled in windows of SHUFFLE_WINDOW). The first `start` frames keep the saved
        order; if this run's window differs, the rest of the epoch is the frames not taken
        yet, in this run's window. Returns a line for the log, or None."""
        self.order = None
        if start <= 0:
            return None                   # the epoch starts afresh, in this run's window
        if not saved or int(saved.get("epoch", -1)) != int(epoch):
            saved = {"epoch": int(epoch), "window": self.SHUFFLE_WINDOW}
        W0, S0, W1 = saved.get("window"), saved.get("switch_at"), saved.get("then")
        self.order = {"epoch": int(epoch), "window": W0}
        if S0 is not None:                # this epoch already changed its window once: keep that
            self.order.update(switch_at=int(S0), then=W1)
            if W1 == self.window:
                return None
            return (f"WARNING: epoch {epoch} already changed its frame window at frame {S0} (to "
                    f"{self.window_name(W1)}); a window of {self.window_name(self.window)} starts with "
                    f"epoch {epoch + 1}.")
        if W0 == self.window:
            return None
        self.order.update(switch_at=int(start), then=self.window)
        taken = ("in episode order (frames not shuffled)" if W0 is None else
                 f"in shuffle windows of {self.window_name(W0)}")
        return (f"Frame order of epoch {epoch}: its first {start} frames were taken {taken}; "
                f"the other {self.total_frames - start} (not taken yet) follow, shuffled "
                + ("over the whole rest of the epoch" if self.window == 0 else
                   f"in windows of {self.window_name(self.window)}")
                + ": each frame is still taken once in the epoch.")

    @staticmethod
    def _perm(key, w, size):
        return np.random.default_rng(key + [w]).permutation(size)

    def _shuffle(self, idx, W, n, key):
        """Index -> index under per-window permutations of [0, n) (windows of W, 0 = one
        window, None = no shuffle) drawn from key + [window]. `idx` is increasing, so each
        window's indices form one contiguous run."""
        if W is None or len(idx) == 0:
            return idx
        W = W or n
        out = np.empty_like(idx)
        win = idx // W
        cuts = np.flatnonzero(np.diff(win)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(idx)]):
            w = int(win[lo])
            base = w * W
            out[lo:hi] = base + self._perm(key, w, min(W, n - base))[idx[lo:hi] - base]
        return out

    def _untaken(self, W, S):
        """Positions of the epoch's sequence not among the first S of its order with window W, increasing."""
        n = self.total_frames
        if W is None:
            return np.arange(S, n, dtype=np.int64)
        W = W or n
        w = S // W
        base = w * W
        size = min(W, n - base)
        head = np.sort(self._perm([self.seed, self.epoch], w, size)[S - base:]) + base
        return np.concatenate([head, np.arange(base + size, n, dtype=np.int64)])

    def _order(self, pos):
        """Positions in this epoch's order (increasing) -> positions in its sequence."""
        W, S, W1 = self.epoch_order()
        key = [int(self.seed), int(self.epoch)]
        if S is None:
            return self._shuffle(pos, W, self.total_frames, key)
        k = int(np.searchsorted(pos, S))
        out = [self._shuffle(pos[:k], W, self.total_frames, key)]
        if k < len(pos):
            rest = self._untaken(W, S)
            out.append(rest[self._shuffle(pos[k:] - S, W1, len(rest), key + [7, int(S)])])
        return np.concatenate(out)

    def __iter__(self):
        n_ep = len(self._frame_counts)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ep_perm = torch.randperm(n_ep, generator=g).numpy()
        else:
            ep_perm = np.arange(n_ep)
        fc = self._frame_counts[ep_perm]
        ends = np.cumsum(fc)                     # end of each shuffled episode in the sequence
        N, B = self.num_replicas, self.batch_size
        steps = self.steps()
        if steps == 0:
            return iter([])
        # This rank's positions in the epoch's order, in the order it trains on them.
        pos = (self.start_sample
               + np.arange(steps, dtype=np.int64)[:, None] * (N * B)
               + self.rank * B + np.arange(B, dtype=np.int64)[None, :]).ravel()
        pos = self._order(pos)
        e = np.searchsorted(ends, pos, side="right")
        frames = self._orig_starts[ep_perm[e]] + (pos - (ends[e] - fc[e]))
        return iter(frames.tolist())


class EgoDexPretrainFlareDataset(Dataset):
    """
    Map-style dataset with a compact cumsum index (same as base pretrain).

    Uses per-worker episode caching: bulk-reads all HDF5 data + video frames
    once per episode, then serves individual frames from memory.  Combined
    with EpisodeGroupedSampler this reduces per-sample I/O from ~200ms to
    ~0.1ms.

    Compared to the base EgoDexPretrainDataset, __getitem__ also returns
    future frames for flare visual prediction, read from the cached video.
    """

    def __init__(self, data_root: str, config, processor, accelerator):
        super().__init__()
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        _episode_dirs = []
        _frame_counts = []
        all_action_q01 = []
        all_action_q99 = []
        all_state_q01 = []
        all_state_q99 = []
        all_te_mean = []
        all_te_std = []

        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
        )
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json found under {data_root}/*/")

        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            for ep in manifest["episodes"]:
                _episode_dirs.append(ep["episode_dir"])
                _frame_counts.append(ep["num_frames"])
            stats = manifest.get("statistics", {})
            if "action" in stats and "state" in stats:
                all_action_q01.append(np.array(stats["action"]["q01"], dtype=np.float32))
                all_action_q99.append(np.array(stats["action"]["q99"], dtype=np.float32))
                all_state_q01.append(np.array(stats["state"]["q01"], dtype=np.float32))
                all_state_q99.append(np.array(stats["state"]["q99"], dtype=np.float32))
            if "tracking_error" in stats:
                all_te_mean.append(np.array(stats["tracking_error"]["mean"], dtype=np.float32))
                all_te_std.append(np.array(stats["tracking_error"]["std"], dtype=np.float32))

        accelerator.print(f"Loaded {len(manifest_paths)} batch manifests")

        self.action_min = np.min(np.stack(all_action_q01), axis=0)
        self.action_max = np.max(np.stack(all_action_q99), axis=0)
        self.state_min = np.min(np.stack(all_state_q01), axis=0)
        self.state_max = np.max(np.stack(all_state_q99), axis=0)

        # Tracking error stats for state noise injection
        te_dim = tracking_error_dim(config.action_dim, config.hand_dim)
        if all_te_mean:
            self.te_mean = np.mean(np.stack(all_te_mean), axis=0)
            self.te_std  = np.mean(np.stack(all_te_std),  axis=0)
        else:
            self.te_mean = np.zeros(te_dim, dtype=np.float32)
            self.te_std  = np.zeros(te_dim, dtype=np.float32)

        self.action_mask = np.ones(config.action_dim, dtype=bool)
        self.state_mask = np.ones(config.action_dim, dtype=bool)

        if config.image_size:
            self.image_size = tuple(config.image_size)
        else:
            self.image_size = None

        # Flare config
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = getattr(config, "n_flare_steps", 0) if self.use_flare else 0
        self.flare_frame_stride = getattr(config, "flare_frame_stride", 4)

        # ── Build compact index via cumulative frame counts ──
        # np.searchsorted on this array maps flat idx → (ep_idx, frame_t)
        # in O(log N) with negligible memory (~8 bytes/episode vs ~64 bytes/frame).
        self._episode_dirs = _episode_dirs
        _frame_counts = np.array(_frame_counts, dtype=np.int64)
        self._cum_frames = np.cumsum(_frame_counts)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(_episode_dirs)

        accelerator.print(f"EgoDex pretrain (flare): {self._num_episodes} episodes, "
                          f"{self._total_transitions} transitions, "
                          f"flare={self.n_flare_steps} steps x stride {self.flare_frame_stride}")

    def __len__(self):
        return self._total_transitions

    @property
    def total_transitions(self):
        return self._total_transitions

    def create_val_split(self, val_ratio=0.02, seed=42):
        """
        Split episodes into train/val. Returns a new dataset for validation.
        Modifies *self* in-place to keep only training episodes.
        """
        import copy
        rng = np.random.RandomState(seed)
        n_val = max(1, int(self._num_episodes * val_ratio))
        perm = rng.permutation(self._num_episodes)
        val_eps = sorted(perm[:n_val].tolist())
        train_eps = sorted(perm[n_val:].tolist())

        frame_counts = np.diff(self._cum_frames, prepend=0)

        val_ds = copy.copy(self)
        val_ds._episode_dirs = [self._episode_dirs[i] for i in val_eps]
        val_fc = frame_counts[val_eps]
        val_ds._cum_frames = np.cumsum(val_fc)
        val_ds._total_transitions = int(val_ds._cum_frames[-1])
        val_ds._num_episodes = len(val_eps)

        self._episode_dirs = [self._episode_dirs[i] for i in train_eps]
        train_fc = frame_counts[train_eps]
        self._cum_frames = np.cumsum(train_fc)
        self._total_transitions = int(self._cum_frames[-1])
        self._num_episodes = len(train_eps)

        self.accelerator.print(
            f"Train/Val split: {self._num_episodes} train eps "
            f"({self._total_transitions} frames), "
            f"{val_ds._num_episodes} val eps "
            f"({val_ds._total_transitions} frames)")
        return val_ds

    @staticmethod
    def _normalize(values, mask, vmin, vmax):
        return np.where(
            mask,
            np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1),
            values,
        )

    @staticmethod
    def _find_head_video(ep_dir: str) -> Optional[str]:
        candidate = os.path.join(ep_dir, "ego_view.mp4")
        if os.path.isfile(candidate):
            return candidate
        matches = glob.glob(os.path.join(ep_dir, "*head*.mp4"))
        return matches[0] if matches else None

    def _load_episode_cache(self, ep_idx: int):
        """
        Preload all HDF5 data + video frames for an episode in one shot.

        HDF5:  single open -> bulk read states[:] and action_chunks[:] -> close.
        Video: single open -> sequential cap.read() for all frames -> close.

        With EpisodeGroupedSampler, consecutive __getitem__ calls hit the same
        episode, so this is called once per episode per worker — amortizing
        the I/O cost to near-zero per frame.
        """
        ep_dir = self._episode_dirs[ep_idx]
        pretrain_h5 = os.path.join(ep_dir, "pretrain.hdf5")
        video_path = self._find_head_video(ep_dir)

        self._cache_ep_idx = ep_idx
        self._cache_h5 = None
        self._cache_frames = None

        # ── Bulk-read HDF5 ──
        if pretrain_h5 and os.path.isfile(pretrain_h5):
            try:
                with h5py.File(pretrain_h5, "r") as f:
                    self._cache_h5 = {
                        "states": f["states"][:],
                        "action_chunks": f["action_chunks"][:],
                        "language": f.attrs.get("language", ""),
                    }
            except Exception:
                pass

        # ── Sequential video decode (no random seeks) ──
        if video_path and os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                self._cache_frames = frames

    def __getitem__(self, idx: int) -> Dict:
        # O(log N) lookup: flat idx -> (episode, frame) via cumulative sums
        ep_idx = int(np.searchsorted(self._cum_frames, idx, side="right"))
        frame_t = int(idx - (int(self._cum_frames[ep_idx - 1]) if ep_idx > 0 else 0))

        fallback_frame = np.zeros((288, 384, 3), dtype=np.uint8)
        fallback = {
            "frame": fallback_frame,
            "state": np.zeros(self.config.action_dim, dtype=np.float32),
            "action_chunk": np.zeros((self.config.action_chunk, self.config.action_dim), dtype=np.float32),
            "language": "",
            "flare_frames": [fallback_frame.copy() for _ in range(self.n_flare_steps)],
        }

        # ── Per-worker episode cache: load once, reuse for all frames ──
        if not hasattr(self, "_cache_ep_idx") or self._cache_ep_idx != ep_idx:
            self._load_episode_cache(ep_idx)

        if self._cache_h5 is None:
            return fallback

        try:
            state = self._cache_h5["states"][frame_t]
            action_chunk = self._cache_h5["action_chunks"][frame_t]
            language = self._cache_h5["language"]
        except (IndexError, KeyError):
            return fallback

        # Current frame from cache
        if self._cache_frames is not None and frame_t < len(self._cache_frames):
            current_frame = self._cache_frames[frame_t]
        else:
            current_frame = fallback_frame

        # Future frames for flare (also from cache)
        num_cached = len(self._cache_frames) if self._cache_frames else 0
        flare_frames = []
        for k in range(self.n_flare_steps):
            future_t = frame_t + (k + 1) * self.flare_frame_stride
            future_t = min(future_t, num_cached - 1) if num_cached > 0 else -1
            if future_t >= 0 and self._cache_frames is not None:
                flare_frames.append(self._cache_frames[future_t])
            else:
                flare_frames.append(current_frame.copy() if current_frame is not fallback_frame
                                    else fallback_frame.copy())

        return {
            "frame": current_frame,
            "state": state,
            "action_chunk": action_chunk,
            "language": language,
            "flare_frames": flare_frames,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)

        # ── Actions ──
        actions = np.stack([x["action_chunk"] for x in batch], axis=0)
        norm_actions = self._normalize(actions, self.action_mask, self.action_min, self.action_max)
        norm_actions = torch.tensor(norm_actions, dtype=torch.bfloat16)

        # ── Flow matching noise ──
        device_cpu = norm_actions.device
        d = torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device_cpu),
            torch.tensor(1.0, dtype=torch.float32, device=device_cpu),
        )
        time = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions

        # ── State ──
        state_raw_list = []
        if cfg.use_robot_state:
            for x in batch:
                s = np.array(x["state"], dtype=np.float32)
                s = add_tracking_error_noise(s, self.te_mean, self.te_std,
                                             cfg.action_dim, cfg.hand_dim)
                ns = self._normalize(s, self.state_mask, self.state_min, self.state_max)
                state_raw_list.append(torch.tensor(ns, dtype=torch.bfloat16))

        # ── Single message: [slow_img | text | fast_img] ──
        # Human video has no wrist camera, so the head frame is passed once more
        # as the single fast image (action expert). This matches a robot with
        # one wrist camera; a two-wrist robot sees two fast images later on.
        all_input_ids = []
        all_pixel_values = []
        all_grid_thw = []
        n_slow_images = 1

        for x in batch:
            img = PIL.Image.fromarray(x["frame"])
            if self.image_size is not None:
                img = img.resize(self.image_size, PIL.Image.LANCZOS)

            content = [
                {"type": "image"},
                {"type": "text", "text": x.get("language", "")},
                {"type": "image"},
            ]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inp = self.processor(
                text=text, images=[img, img],
                return_tensors="pt", padding=False,
            )
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── Flare future frames ──
        flare_pixel_values = None
        flare_grid_thw = None

        if self.n_flare_steps > 0:
            flare_pil_imgs = []
            for x in batch:
                for ff in x["flare_frames"]:
                    pil_img = PIL.Image.fromarray(ff)
                    if self.image_size is not None:
                        pil_img = pil_img.resize(self.image_size, PIL.Image.LANCZOS)
                    flare_pil_imgs.append(pil_img)

            flare_inp = self.processor.image_processor(flare_pil_imgs, return_tensors="pt")
            flare_pixel_values = flare_inp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = flare_inp.image_grid_thw

        # ── Padding ──
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)

        padded_ids = []
        attention_ms = []
        for ids in all_input_ids:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
            attn = torch.ones(max_len, dtype=torch.long)
            if pad_len > 0:
                attn[:pad_len] = 0
            attention_ms.append(attn)

        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attention_ms)
        pixel_values = (torch.cat(all_pixel_values, dim=0) if all_pixel_values else None)
        image_grid_thw = (torch.cat(all_grid_thw, dim=0) if all_grid_thw else None)
        state_raw = torch.stack(state_raw_list) if state_raw_list else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }


# ───────────────────────────────────────────────────────────────────
#  EMA of the weights; weights kept apart from the checkpoint rotation
# ───────────────────────────────────────────────────────────────────
class ShardedCPUEMA:
    """Exponential moving average (EMA) of the trainable weights, in fp32 in CPU RAM, split across the ranks.

    The GPUs have no room for it (4 x A100 at batch 40 peak at ~69 of 80 GiB) and an fp32 copy of the
    3.2 B trainable weights is 12.6 GB, so each rank averages its share of the parameters (whole tensors,
    balanced by size). Every `every` optimizer steps a rank copies its share of the bf16 weights to pinned
    memory (a fraction of a second) and folds it into the average in a background thread while training
    goes on:  ema <- ema + (1 - d) (w - ema),  d = min(decay, (1 + n) / (10 + n)) ** every,  n = optimizer
    steps since the average started (the usual warmup: the starting weights fade quickly).

    Checkpoints get <dir>/ema_state/ema_rank<r>.pt (to resume, also with another number of ranks) and
    <dir>/model_ema.pt: model.pt with the trainable weights replaced by their averages (same keys, bf16)."""

    def __init__(self, raw_model, decay, every, rank, world):
        self.decay, self.every, self.rank = float(decay), max(1, int(every)), int(rank)
        aliases = {}                                    # state_dict names of a parameter (tied weights: several)
        for name, p in raw_model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(p), []).append(name)
        loads = [0] * world
        self.params, self.keys = {}, {}
        for name, p in sorted(((n, p) for n, p in raw_model.named_parameters() if p.requires_grad),
                              key=lambda x: (-x[1].numel(), x[0])):
            r = loads.index(min(loads))
            loads[r] += p.numel()
            if r == self.rank:
                self.params[name] = p
                self.keys[name] = aliases.get(id(p), [name])
        self.ema = {n: p.detach().to("cpu").float() for n, p in self.params.items()}
        self.buf = {n: torch.empty(p.shape, dtype=p.dtype, pin_memory=True) for n, p in self.params.items()}
        self.numel = sum(p.numel() for p in self.params.values())
        self.n = 0
        self.thread = None
        self.last_fold_s = 0.0

    def step(self):
        """After every optimizer step."""
        self.n += 1
        if self.n % self.every:
            return
        self.wait()
        torch.cuda.synchronize()
        for name, p in self.params.items():
            self.buf[name].copy_(p.detach(), non_blocking=True)
        torch.cuda.synchronize()                        # copied before the next step changes the weights
        d = min(self.decay, (1 + self.n) / (10 + self.n)) ** self.every
        self.thread = threading.Thread(target=self._fold, args=(1.0 - d,), daemon=True)
        self.thread.start()

    def _fold(self, weight):
        t0 = time.time()
        for name, e in self.ema.items():
            e.lerp_(self.buf[name].float(), weight)
        self.last_fold_s = time.time() - t0

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def save_shard(self, ckpt_dir):
        self.wait()
        d = os.path.join(ckpt_dir, "ema_state")
        os.makedirs(d, exist_ok=True)
        torch.save({"ema": self.ema, "keys": self.keys, "n": self.n, "decay": self.decay, "every": self.every},
                   os.path.join(d, f"ema_rank{self.rank}.pt"))

    def load(self, ckpt_dir):
        """This rank's averages from <ckpt_dir>/ema_state (saved by any number of ranks); False if there is none."""
        files = sorted(glob.glob(os.path.join(ckpt_dir or "", "ema_state", "ema_rank*.pt")))
        if not files:
            return False
        found = set()
        for path in files:
            shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            self.n = max(self.n, int(shard.get("n", 0)))
            for name, t in shard["ema"].items():
                if name in self.ema:
                    self.ema[name].copy_(t)
                    found.add(name)
        if len(found) != len(self.ema):
            raise RuntimeError(f"{ckpt_dir}/ema_state lacks {len(self.ema) - len(found)} of this rank's "
                               f"{len(self.ema)} averaged parameters, e.g. {sorted(set(self.ema) - found)[:3]}")
        return True


def _write_ema_weights(sd, ckpt_dir):
    """<ckpt_dir>/model_ema.pt: the state dict `sd` of model.pt with the trainable weights replaced by the
    averages of all ranks' shards in <ckpt_dir>/ema_state. Returns the number of keys replaced."""
    out, n = dict(sd), 0
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "ema_state", "ema_rank*.pt"))):
        shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for name, t in shard["ema"].items():
            for key in shard["keys"][name]:
                if key in out:
                    out[key] = t.to(out[key].dtype)
                    n += 1
    torch.save(out, os.path.join(ckpt_dir, "model_ema.pt"))
    return n


_WEIGHT_FILES = ("model.pt", "model_ema.pt", "config.json", "stats_data.json", "training_args.json",
                 "training_state.json")
_background = []                                        # weight copies still running (joined at the end)


def _copy_weights(src, dst):
    """The weights and what loading them needs (no optimizer state) from checkpoint dir `src` to `dst`,
    written as dst.tmp and renamed when complete."""
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name in _WEIGHT_FILES:
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), tmp)
    if os.path.isdir(os.path.join(src, "processor")):
        shutil.copytree(os.path.join(src, "processor"), os.path.join(tmp, "processor"))
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp, dst)
    logger.info(f"Weights kept: {dst}")


def _write_weights(save_dir, sd, processor, args, dataset, state):
    """model.pt and what loading it needs: processor, base config, normalization stats, settings, state."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(sd, os.path.join(save_dir, "model.pt"))
    processor.save_pretrained(os.path.join(save_dir, "processor"))

    src_config = os.path.join(args.model_path, "config.json")
    if os.path.exists(src_config):
        shutil.copy(src_config, os.path.join(save_dir, "config.json"))

    with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
        json.dump({"egodex": {
            "action": {
                "mask": dataset.action_mask.tolist(),
                "q01": dataset.action_min.tolist(),
                "q99": dataset.action_max.tolist(),
            },
            "state": {
                "mask": dataset.state_mask.tolist(),
                "q01": dataset.state_min.tolist(),
                "q99": dataset.state_max.tolist(),
            },
        }}, f, indent=2)

    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        json.dump({
            "model_path": args.model_path,
            "action_dim": args.action_dim,
            "action_chunk": args.action_chunk,
            "use_robot_state": args.use_robot_state,
            "training_stage": 1,
            "data_root": args.data_root,
            "use_flare": args.use_flare,
            "n_flare_tokens_per_frame": args.n_flare_tokens_per_frame,
            "n_flare_steps": args.n_flare_steps,
            "flare_layer_index": args.flare_layer_index,
        }, f, indent=2)

    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def save_checkpoint(model, processor, accelerator, args, epoch, global_step, dataset,
                    save_full_state=True, lr_scheduler=None, position=None, ema=None):
    """Save a checkpoint.

    Always writes `model.pt` + metadata (compatible with downstream eval).
    When save_full_state=True, additionally writes a `state/` subdir via
    `accelerator.save_state` containing optimizer, LR scheduler and RNG state
    (sharded in DeepSpeed format), plus `training_state.json` with the
    global_step / epoch needed to resume.
    With `ema`, also `ema_state/` (each rank's averages) and `model_ema.pt`.
    Every --keep_weights_every steps the weights (model.pt, model_ema.pt and
    what loading them needs) are also copied to <output_dir>/weights/, which
    the rotation (--max_ckpts) never deletes.
    """
    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    state_dir = os.path.join(save_dir, "state")

    sd = None
    if accelerator.is_main_process:
        ckpts = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(ckpts) >= args.max_ckpts:
            oldest = min(ckpts, key=lambda f: os.path.getctime(os.path.join(args.output_dir, f)))
            shutil.rmtree(os.path.join(args.output_dir, oldest))

        sd = accelerator.get_state_dict(model)
        _write_weights(save_dir, sd, processor, args, dataset, {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            "n_epochs": int(args.n_epochs),
            "seed": int(args.seed),
            "train_bsz_per_gpu": int(args.train_bsz_per_gpu),
            "learning_rate": float(args.learning_rate),
            "warmup_rates": float(args.warmup_rates),
            "min_lr_ratio": float(args.min_lr_ratio),
            # For resuming with other ranks / batch size (see train()):
            # ranks, accumulation, and where in the data the run is.
            "world_size": int(accelerator.num_processes),
            "gradient_accumulation_steps": int(accelerator.gradient_accumulation_steps),
            **(position or {}),
        })

    accelerator.wait_for_everyone()

    if ema is not None:
        ema.save_shard(save_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            n = _write_ema_weights(sd, save_dir)
            accelerator.print(f"EMA weights: {n} tensors averaged -> {save_dir}/model_ema.pt "
                              f"(an update takes {ema.last_fold_s:.1f} s in the background on rank 0)")
    del sd

    # Collective: save optimizer / scheduler / RNG for exact resume.
    if save_full_state:
        accelerator.save_state(state_dir)
        # Accelerate's DeepSpeed path does NOT write `scheduler.bin` when the
        # LR scheduler was prepared separately from the model+optimizer (as
        # we do here, since num_training_steps depends on the prepared
        # world_size). Save it ourselves so resume can restore the cosine
        # curve position.
        if lr_scheduler is not None and accelerator.is_main_process:
            torch.save(lr_scheduler.state_dict(),
                       os.path.join(state_dir, "scheduler.bin"))

    accelerator.wait_for_everyone()
    logger.info(f"Checkpoint {epoch}-{global_step} saved"
                f"{' (with resumable state)' if save_full_state else ''}.")

    keep = getattr(args, "keep_weights_every", 0)
    if accelerator.is_main_process and keep > 0 and global_step > 0 and global_step % keep == 0:
        t = threading.Thread(target=_copy_weights, daemon=True, args=(
            save_dir, os.path.join(args.output_dir, "weights", f"checkpoint-{epoch}-{global_step}")))
        t.start()
        _background.append(t)


def _best_val_dir(args):
    """Shared by the jobs of an experiment: <output_dir>/<experiment_name>/best_val."""
    return os.path.join(os.path.dirname(os.path.normpath(args.output_dir)), "best_val")


def _val_signature(args, n_val_frames):
    """What makes two validation losses comparable (else the best is chosen anew)."""
    return {"val_fixed": int(args.val_fixed), "val_frames": int(n_val_frames), "seed": int(args.seed),
            "val_ratio": float(args.val_ratio), "lerobot_root": sorted(map(os.path.basename,
                                                                          getattr(args, "lerobot_root", None) or []))}


def _read_best_val(args, n_val_frames):
    path = os.path.join(_best_val_dir(args), "best_val.json")
    if os.path.isfile(path):
        with open(path) as f:
            best = json.load(f)
        if best.get("signature") == _val_signature(args, n_val_frames):
            return best
        print(f"{path}: another validation setting; the best validation weights are chosen anew", flush=True)
    return {"val_action_loss": float("inf")}


def _save_best_val(model, processor, accelerator, args, dataset, epoch, step, val_m, n_val_frames,
                   dataset_frames):
    """Main process: the current weights to best_val/checkpoint-<epoch>-<step> (replacing the previous best)."""
    root = _best_val_dir(args)
    dst = os.path.join(root, f"checkpoint-{epoch}-{step}")
    tmp = os.path.join(root, ".writing")
    shutil.rmtree(tmp, ignore_errors=True)
    best = {"val_action_loss": float(val_m["val/action_loss"]), "val_total_loss": float(val_m["val/total_loss"]),
            "epoch": int(epoch), "global_step": int(step), "run": os.path.basename(args.output_dir),
            "checkpoint": dst, "signature": _val_signature(args, n_val_frames)}
    _write_weights(tmp, accelerator.get_state_dict(model), processor, args, dataset, {
        "epoch": int(epoch), "global_step": int(step), "dataset_frames": int(dataset_frames),
        "wandb_run_id": wandb.run.id if wandb.run is not None else None, **{f"best_{k}": v for k, v in best.items()
                                                                          if k.startswith("val_")}})
    for old in glob.glob(os.path.join(root, "checkpoint-*")):
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, dst)
    with open(os.path.join(root, "best_val.json.tmp"), "w") as f:
        json.dump(best, f, indent=2)
    os.replace(os.path.join(root, "best_val.json.tmp"), os.path.join(root, "best_val.json"))
    return best


def _seed_val_worker(base, worker_id):
    """Same flow-matching noise and times for every validation (each worker serves the same batches)."""
    torch.manual_seed(base + worker_id)
    np.random.seed((base + worker_id) % 2 ** 32)


def _resolve_resume(resume_checkpoint):
    """Normalize the --resume_checkpoint arg to (model_pt_path, ckpt_dir, state_dir_or_None).

    Accepts either a path to `model.pt` or a checkpoint directory.
    `state_dir` is returned only when `<ckpt_dir>/state/` exists, signaling
    the new-style full-state checkpoint.
    """
    if not resume_checkpoint:
        return None, None, None
    path = resume_checkpoint.rstrip("/")
    if os.path.isdir(path):
        ckpt_dir = path
        model_pt = os.path.join(ckpt_dir, "model.pt")
    else:
        model_pt = path
        ckpt_dir = os.path.dirname(path)
    state_dir = os.path.join(ckpt_dir, "state")
    if not os.path.isdir(state_dir):
        state_dir = None
    return model_pt, ckpt_dir, state_dir


def _parse_step_from_ckpt_dir(ckpt_dir):
    """Parse (epoch, global_step) from 'checkpoint-{epoch}-{step}' dir name, or (None, None)."""
    base = os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else ""
    m = re.match(r"checkpoint-(\d+)-(\d+)$", base)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _read_training_state(ckpt_dir):
    """training_state.json of a checkpoint dir as a dict ({} when there is none)."""
    path = os.path.join(ckpt_dir, "training_state.json") if ckpt_dir else ""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _zero_optim_files(state_dir):
    """A DeepSpeed ZeRO-1/2 checkpoint's optimizer files, one per saving rank, in rank order."""
    if not state_dir:
        return []
    files = glob.glob(os.path.join(state_dir, "*", "*zero_pp_rank_*_optim_states.pt"))
    return sorted(files, key=lambda p: int(re.search(r"zero_pp_rank_(\d+)_", os.path.basename(p)).group(1)))


def _zero_partition_count(state_dir):
    return len(_zero_optim_files(state_dir))


def _copy_flat_range(pieces, lo, hi, dst):
    """dst[:hi-lo] = [lo, hi) of the concatenated 1-D `pieces`, reading only what overlaps."""
    off = 0
    for t in pieces:
        n = t.numel()
        a, b = max(lo, off), min(hi, off + n)
        if a < b:
            dst[a - lo:b - lo].copy_(t.reshape(-1)[a - off:b - off])
        off += n


def _load_zero_repartitioned(engine, state_dir, accelerator):
    """Restore a DeepSpeed ZeRO-1/2 checkpoint saved with another number of ranks.

    DeepSpeed's own loader needs the same number: each rank reads the optimizer
    partition its rank saved. Here, for each param group, the saved per-rank
    partitions of the fp32 master weights and of the Adam state are taken as one
    flat buffer and this rank keeps its slice of the current partitioning: a
    group's flat buffer is its params in order with padding only at the end,
    whatever the number of ranks. The files are memory-mapped, so a rank reads
    its slice only. The model weights must already hold the checkpoint's
    (model.pt); the trainable ones are then rebuilt from the fp32 masters. The
    RNG states are not restored."""
    zo = engine.optimizer
    if not hasattr(zo, "single_partition_of_fp32_groups"):
        raise RuntimeError("resuming with another number of ranks needs DeepSpeed ZeRO stage 1/2")
    files = _zero_optim_files(state_dir)
    if not files:
        raise FileNotFoundError(f"no ZeRO optimizer files (*zero_pp_rank_*_optim_states.pt) in {state_dir}")
    saved = [torch.load(p, map_location="cpu", weights_only=False, mmap=True)["optimizer_state_dict"]
             for p in files]
    for p, sd in zip(files, saved):
        if not isinstance(sd["base_optimizer_state"], dict):
            raise RuntimeError(f"{p}: elastic ZeRO checkpoint, not supported")
    opt = zo.optimizer
    sums = []
    for i, part in enumerate(zo.single_partition_of_fp32_groups):
        if opt.param_groups[i]["params"][0] is not part:
            raise RuntimeError(f"param group {i}: unexpected DeepSpeed optimizer layout")
        rank = dist.get_rank(group=zo.real_dp_process_group[i])
        numel = sum(p.numel() for p in zo.bit16_groups[i])
        size = part.numel()
        lo, hi = rank * size, min((rank + 1) * size, numel)
        fp32 = [sd["single_partition_of_fp32_groups"][i] for sd in saved]
        if sum(t.numel() for t in fp32) != numel:
            raise RuntimeError(f"param group {i}: the checkpoint has {sum(t.numel() for t in fp32)} "
                               f"elements, the model {numel}")
        part.data.zero_()
        _copy_flat_range(fp32, lo, hi, part.data)
        states = []
        for sd in saved:
            bos = sd["base_optimizer_state"]
            states.append(bos["state"].get(bos["param_groups"][i]["params"][0]))
        if all(st is not None for st in states):
            new = {}
            for key, v in states[0].items():
                if torch.is_tensor(v) and v.dim() > 0:
                    dst = torch.zeros_like(part)
                    _copy_flat_range([st[key] for st in states], lo, hi, dst)
                    new[key] = dst
                else:
                    new[key] = v.clone() if torch.is_tensor(v) else v
            opt.state[part] = new
        st = opt.state.get(part, {})
        sums += [part.double().square().sum(),
                 st["exp_avg_sq"].double().sum() if "exp_avg_sq" in st else torch.zeros((), device=part.device, dtype=torch.float64)]
    del saved
    zo.update_lp_params()        # fp32 masters -> bf16 params, all-gathered (collective)
    sums = torch.stack(sums)
    dist.all_reduce(sums)
    accelerator.print(f"Re-split ZeRO state of {len(files)} ranks over {dist.get_world_size()}: "
                      + "; ".join(f"group {i}: sum(w^2)={sums[2 * i]:.6e}, sum(exp_avg_sq)={sums[2 * i + 1]:.6e}"
                                  for i in range(len(sums) // 2)))


def _set_scheduler_step(lr_scheduler, n):
    """Put an LR scheduler (maybe wrapped by accelerate) at step n of its schedule."""
    sched = getattr(lr_scheduler, "scheduler", lr_scheduler)
    sched.last_epoch = n - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # "lr_scheduler.step() before optimizer.step()"
        sched.step()


class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_loss = torch.tensor(0.0, device=device)
        self.flare_loss = torch.tensor(0.0, device=device)
        self.total_loss = torch.tensor(0.0, device=device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def update(self, total, action, flare=0.0):
        self.n_step += 1
        self.total_loss += total.item() if torch.is_tensor(total) else total
        self.action_loss += action.item() if torch.is_tensor(action) else action
        self.flare_loss += flare.item() if torch.is_tensor(flare) else flare

    def get_metric(self, reset=True):
        if dist.is_initialized():
            for t in [self.total_loss, self.action_loss, self.flare_loss]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = self.world_size * max(self.n_step, 1)
        m = {
            "total_loss": self.total_loss.item() / denom,
            "action_loss": self.action_loss.item() / denom,
            "flare_loss": self.flare_loss.item() / denom,
        }
        if reset:
            self.n_step = 0
            for t in [self.total_loss, self.action_loss, self.flare_loss]:
                t.fill_(0)
        return m


class MemoryMonitor:
    """GPU and host memory for wandb, in GiB, read at each logging step.

    VRAM, per GPU, the maximum over ranks:
      mem/vram_peak      PyTorch allocator peak since the previous log (OOM margin)
      mem/vram_alloc     allocated now
      mem/vram_reserved  held by the PyTorch caching allocator now
      mem/vram_used      used on the device as the driver reports it (adds the CUDA
                         context, NCCL buffers and other processes)
    RAM of the whole Slurm job on this node (all ranks and data workers), from its
    cgroup; left out when that cgroup isn't found (e.g. outside Slurm):
      mem/ram_job          everything charged to the job, page cache included
      mem/ram_job_process  anonymous + shared memory: what the processes hold
      mem/ram_job_cache    reclaimable file cache
      mem/ram_job_peak     peak of mem/ram_job since the job started
      mem/ram_job_limit    the job's limit (--mem)
    and mem/ram_main_rss, the resident memory of rank 0's training process.

    collect() does one small all_reduce, so every rank must call it at the same step."""
    GiB = float(2 ** 30)

    def __init__(self, device):
        self.device = device
        self.cgroup = self._job_cgroup()

    @staticmethod
    def _job_cgroup():
        """The cgroup v2 dir of this Slurm job (.../job_<id>, above its steps and tasks)."""
        try:
            with open("/proc/self/cgroup") as f:
                rel = next(line.split(":", 2)[2].strip() for line in f if line.startswith("0::"))
        except (OSError, StopIteration, IndexError):
            return None
        parts = rel.strip("/").split("/")
        for i, part in enumerate(parts):
            if part.startswith("job_"):
                path = os.path.join("/sys/fs/cgroup", *parts[:i + 1])
                return path if os.path.isfile(os.path.join(path, "memory.current")) else None
        return None

    def _cgroup_file(self, name):
        try:
            with open(os.path.join(self.cgroup, name)) as f:
                return f.read().strip()
        except OSError:
            return None

    def _ram(self):
        out = {}
        try:
            with open("/proc/self/statm") as f:
                out["mem/ram_main_rss"] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / self.GiB
        except (OSError, ValueError, IndexError):
            pass
        if self.cgroup is None:
            return out
        for key, name in (("mem/ram_job", "memory.current"), ("mem/ram_job_peak", "memory.peak"),
                          ("mem/ram_job_limit", "memory.max")):
            value = self._cgroup_file(name)
            if value and value.isdigit():
                out[key] = int(value) / self.GiB
        stat = self._cgroup_file("memory.stat")
        if stat:
            s = dict((k, int(v)) for k, v in (line.split() for line in stat.splitlines()) if v.isdigit())
            if "anon" in s:
                out["mem/ram_job_process"] = (s["anon"] + s.get("shmem", 0)) / self.GiB
            if "file" in s:   # cgroup v2 counts shmem/tmpfs under file too
                out["mem/ram_job_cache"] = (s["file"] - s.get("shmem", 0)) / self.GiB
        return out

    def collect(self, is_main_process=True):
        free, total = torch.cuda.mem_get_info(self.device)
        vram = torch.tensor([torch.cuda.max_memory_allocated(self.device),
                             torch.cuda.memory_allocated(self.device),
                             torch.cuda.memory_reserved(self.device),
                             total - free], dtype=torch.float64, device=self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        if dist.is_initialized():
            dist.all_reduce(vram, op=dist.ReduceOp.MAX)
        if not is_main_process:
            return {}
        out = dict(zip(("mem/vram_peak", "mem/vram_alloc", "mem/vram_reserved", "mem/vram_used"),
                       (vram / self.GiB).tolist()))
        out.update(self._ram())
        return out


@torch.no_grad()
def run_validation(model, val_dataloader, accelerator, raw_model_fn,
                   use_flare, K, T_per_frame, flare_layer_idx, args):
    """Run validation and return averaged metrics."""
    model.eval()
    val_loss_sum = torch.tensor(0.0, device=torch.cuda.current_device())
    val_act_sum = torch.tensor(0.0, device=torch.cuda.current_device())
    val_flare_sum = torch.tensor(0.0, device=torch.cuda.current_device())
    n_val = 0
    max_val_batches = getattr(args, "max_val_batches", 50)
    rng_guard = contextlib.ExitStack()
    if getattr(args, "val_fixed", 0):     # same noise also when the batches are collated here (0 workers)
        rng_guard.enter_context(torch.random.fork_rng(devices=[]))
        torch.manual_seed(args.seed * 1000 + accelerator.process_index * 64 + 7)
    for i, batch in enumerate(val_dataloader):
        if i >= max_val_batches:
            break
        raw_model = raw_model_fn()

        inputs_embeds = raw_model.prepare_inputs_embeds(
            input_ids=batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
        )

        n_slow_imgs = batch["n_slow_images"]
        grid_thw = batch.get("image_grid_thw")
        B = inputs_embeds.shape[0]
        if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
            merge = getattr(raw_model.visual, "spatial_merge_size", 2)
            n_imgs_per_sample = grid_thw.shape[0] // B
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs]
            )
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, batch["input_ids"],
                raw_model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds
            fast_embeds = inputs_embeds[:, :0]

        L_slow = slow_embeds.shape[1]
        pos_ids, _ = raw_model.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch["attention_mask"],
        )
        pos_ids = pos_ids[:, :, :L_slow]

        if use_flare:
            flare_q = raw_model.flare_queries.expand(B, -1, -1).to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
            pos_ids = extend_position_ids_for_flare(pos_ids, K)
            L_latent = slow_embeds_ext.shape[1]
        else:
            slow_embeds_ext = slow_embeds
            L_latent = L_slow

        if args.use_robot_state and batch["state_raw"] is not None:
            state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
            state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
        else:
            state_embeds = torch.empty((B, 0, slow_embeds.shape[2]),
                                        device=slow_embeds.device, dtype=slow_embeds.dtype)
        n_state = state_embeds.shape[1]

        noisy_actions = raw_model.x_embedder(batch["noisy_actions"].to(slow_embeds.dtype))
        timesteps = raw_model.t_embedder(batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)
        chunk = args.action_chunk
        target = batch["target"].to(slow_embeds.dtype)

        full_embeds = torch.cat([
            slow_embeds_ext, fast_embeds, state_embeds, timesteps, noisy_actions,
        ], dim=1)
        n_fast = fast_embeds.shape[1]
        L_total = full_embeds.shape[1]

        outputs = model.model(
            inputs_embeds=full_embeds,
            position_ids=pos_ids,
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=use_flare and (flare_layer_idx != -1),
            latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
            action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
            tactile_indexes=torch.arange(0, 0, device=full_embeds.device),
        )
        hidden = outputs.last_hidden_state

        act_pred_start = L_latent + n_fast + n_state + 1
        v_act = raw_model.final_layer(hidden[:, act_pred_start: act_pred_start + chunk, :])
        loss_act = nn.MSELoss()(v_act, target)

        loss_flare = torch.tensor(0.0, device=loss_act.device)
        if use_flare and batch["flare_pixel_values"] is not None:
            if flare_layer_idx == -1:
                flare_source = hidden
            else:
                all_hs = outputs.hidden_states
                n_layers = len(all_hs) - 1
                li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
                li = max(0, min(li, n_layers - 1))
                flare_source = all_hs[li + 1]

            flare_hidden = flare_source[:, L_slow: L_slow + K, :]
            flare_pred = raw_model.flare_proj(flare_hidden)
            f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
            f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)
            vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
            features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out
            merge = getattr(raw_model.visual, "spatial_merge_size", 2)
            frame_feats = []
            offset = 0
            for g in f_thw:
                n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                pooled = F.adaptive_avg_pool1d(
                    features[offset: offset + n_tok].unsqueeze(0).permute(0, 2, 1),
                    T_per_frame).permute(0, 2, 1).squeeze(0)
                frame_feats.append(pooled)
                offset += n_tok
            flare_targets = torch.stack(frame_feats).view(B, K, -1)
            pred_norm = F.normalize(flare_pred, dim=-1)
            tgt_norm = F.normalize(flare_targets, dim=-1)
            loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

        loss = loss_act + args.flare_loss_weight * loss_flare
        val_loss_sum += loss.item()
        val_act_sum += loss_act.item()
        val_flare_sum += loss_flare.item()
        n_val += 1
    rng_guard.close()

    # All-reduce across ranks
    for t in [val_loss_sum, val_act_sum, val_flare_sum]:
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    ws = dist.get_world_size() if dist.is_initialized() else 1
    denom = ws * max(n_val, 1)

    model.train()
    return {
        "val/total_loss": val_loss_sum.item() / denom,
        "val/action_loss": val_act_sum.item() / denom,
        "val/flare_loss": val_flare_sum.item() / denom,
    }


# ───────────────────────────────────────────────────────────────────
#  Main training loop
# ───────────────────────────────────────────────────────────────────
def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False, even_batches=True),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))],
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        # One wandb run per training lineage: a resumed job continues the run of its checkpoint
        # (training_state.json "wandb_run_id"), so the whole pretraining is one curve across jobs.
        # WANDB_RUN_ID overrides. If wandb can't start online (no login, no network), log offline.
        _, _ckpt_dir, _ = _resolve_resume(args.resume_checkpoint)
        run_id = os.environ.get("WANDB_RUN_ID") or _read_training_state(_ckpt_dir).get("wandb_run_id")
        kw = dict(project=args.experiment_name, dir=args.log_dir, resume="allow",
                  id=run_id or wandb.util.generate_id())
        if not run_id:
            kw["name"] = args.run_name
        try:
            wandb.init(**kw)
        except Exception as e:
            accelerator.print(f"wandb could not start ({type(e).__name__}: {str(e)[:200]}); logging offline")
            wandb.init(mode="offline", **kw)
        wandb.config.update(vars(args), allow_val_change=True)
        accelerator.print(f"wandb run {wandb.run.id} ({'continued' if run_id else 'new'}, "
                          f"{'offline' if wandb.run.settings._offline else 'online'})")

    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_batch_size"] = (args.train_bsz_per_gpu
                               * (dist.get_world_size() if dist.is_initialized() else 1)
                               * accelerator.gradient_accumulation_steps)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=args.model_path,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_tactile_deform=False,
        use_robot_state=bool(args.use_robot_state),
        torch_dtype=torch.bfloat16,
        n_flare_tokens_per_frame=args.n_flare_tokens_per_frame if args.use_flare else 0,
        n_flare_steps=args.n_flare_steps if args.use_flare else 0,
        flare_layer_index=args.flare_layer_index,
    )

    model.initialize_vla_weights()
    accelerator.print("VLA weights initialized.")

    T_per_frame = args.n_flare_tokens_per_frame
    S_steps = args.n_flare_steps
    K = T_per_frame * S_steps
    use_flare = bool(args.use_flare and K > 0)
    flare_layer_idx = args.flare_layer_index

    if use_flare:
        accelerator.print(
            f"Flare: {S_steps} steps x {T_per_frame} tok/frame = {K} total, "
            f"layer_index={flare_layer_idx}, stride={args.flare_frame_stride}")

    # Resolve resume paths (supports either a model.pt file or a ckpt directory).
    resume_model_pt, resume_ckpt_dir, resume_state_dir = _resolve_resume(args.resume_checkpoint)

    # Init action expert from latent expert
    if not args.resume_checkpoint:
        named_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "_action" in name:
                base = name.replace("_action", "")
                if base in named_params:
                    param.data.copy_(named_params[base].data)
        accelerator.print("Action expert initialized from latent expert.")

    # A full checkpoint saved with another number of ranks can't go through
    # DeepSpeed's loader (its ZeRO optimizer state is split per rank): its
    # weights come from model.pt now, its optimizer state is re-split after
    # prepare (_load_zero_repartitioned).
    resume_tstate = _read_training_state(resume_ckpt_dir)
    saved_world = int(resume_tstate.get("world_size") or _zero_partition_count(resume_state_dir) or 0)
    repartition = (resume_state_dir is not None and saved_world > 0
                   and saved_world != accelerator.num_processes)
    if repartition:
        accelerator.print(f"Checkpoint saved with {saved_world} ranks, running with "
                          f"{accelerator.num_processes}: re-splitting its optimizer state.")

    # Old-style resume (weights-only): load model weights now, before accelerator.prepare.
    # New-style (state/ present): model weights are restored by accelerator.load_state
    # *after* prepare — skip here so the later load takes effect.
    if args.resume_checkpoint and (resume_state_dir is None or repartition):
        resume_sd = torch.load(resume_model_pt, map_location="cpu")
        if "state_dict" in resume_sd:
            resume_sd = resume_sd["state_dict"]
        missing, unexpected = model.load_state_dict(resume_sd, strict=False)
        accelerator.print(f"Resumed weights (old-style) from {resume_model_pt}: "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")

    # Freeze vision + tactile, train latent + action + flare
    for name, param in model.named_parameters():
        if name.startswith("visual") or name.startswith("deform_encoder"):
            param.requires_grad = False
        elif "_tactile" in name or "final_layer_tactile" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total: {total/1e9:.2f}B  Trainable: {trainable/1e9:.2f}B "
                      f"({trainable/total*100:.1f}%)")

    no_decay = ["bias", "norm.weight", "q_norm.weight", "k_norm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    if getattr(args, "data_format", "json") == "lerobot":
        from qwen_vla.lerobot_dataset import TRexLeRobotDataset
        dataset = TRexLeRobotDataset(args, processor, accelerator)
    else:
        dataset = EgoDexPretrainFlareDataset(args.data_root, args, processor, accelerator)

    # Train/val split
    val_dataloader = None
    if args.val_ratio > 0:
        val_dataset = dataset.create_val_split(val_ratio=args.val_ratio, seed=args.seed)
        val_source, val_worker_init = val_dataset, None
        if args.val_fixed:
            # The loader is sharded by accelerate and not shuffled, so without this every validation scores
            # the first max_val_batches x bsz x ranks frames of the val split: a few episodes of its first
            # dataset. Instead: the same frames spread over the whole split, and the same noise and flow times.
            n_fix = min(len(val_dataset), args.max_val_batches * args.train_bsz_per_gpu * accelerator.num_processes)
            pick = np.sort(np.random.RandomState(args.seed).choice(len(val_dataset), n_fix, replace=False))
            val_source = torch.utils.data.Subset(val_dataset, pick.tolist())
            val_worker_init = functools.partial(_seed_val_worker,
                                                args.seed * 1000 + accelerator.process_index * 64 + 7)
            if hasattr(val_dataset, "_src_cum") and hasattr(val_dataset, "sources"):
                src = np.searchsorted(np.asarray(val_dataset._src_cum), pick, side="right")
                fam = collections.Counter(re.sub(r"_\d+x\d+(-\d+)?$", "", os.path.basename(
                    str(val_dataset.sources[i].root)).split("__")[-1]) for i in src)
                accelerator.print(f"Fixed validation: {n_fix} of {len(val_dataset)} val frames, "
                                  + ", ".join(f"{k} {v}" for k, v in sorted(fam.items())))
        val_dataloader = DataLoader(
            val_source,
            batch_size=args.train_bsz_per_gpu,
            shuffle=False,
            drop_last=True,
            collate_fn=val_dataset.collate_fn,
            # Forked anew for every validation: each worker copies pages of its parent, so with 5 per GPU a
            # validation added ~90 GiB and pushed a 4-GPU job against its memory limit (hours of thrashing).
            num_workers=(args.val_num_workers if args.val_num_workers >= 0
                         else max(1, args.num_workers // 2)),
            pin_memory=True,
            worker_init_fn=val_worker_init,
        )

    # Episode-grouped sampler: clusters frame indices by episode so that
    # consecutive __getitem__ calls hit the same episode.  Combined with
    # per-worker episode caching, this eliminates per-frame HDF5 open/close
    # and video seek — reducing per-sample I/O from ~200ms to ~0.1ms.
    sampler = EpisodeGroupedSampler(
        dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, seed=args.seed,
        shuffle_frames=bool(args.shuffle_frames), window=args.shuffle_window,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_bsz_per_gpu,
        sampler=sampler,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # IMPORTANT: Do NOT prepare the training dataloader with accelerate.
    # Our EpisodeGroupedSampler already shards indices by rank.  accelerate's
    # prepare() wraps the batch_sampler in BatchSamplerShard which shards
    # *batches* a second time → each rank sees 1/N^2 of the data (the
    # "2137 steps" bug).  We prepare model + optimizer only, and move train
    # batches to device manually.
    if val_dataloader is not None:
        model, optimizer, val_dataloader = accelerator.prepare(
            model, optimizer, val_dataloader)
    else:
        model, optimizer = accelerator.prepare(model, optimizer)

    # steps_per_epoch from ground truth
    world_size = accelerator.num_processes
    train_samples = len(dataset)
    samples_per_rank = train_samples // world_size
    steps_per_epoch = sampler.steps(0)       # = train_samples // (world_size * bsz)
    num_training_steps = steps_per_epoch * args.n_epochs // accelerator.gradient_accumulation_steps
    accelerator.print(f"Dataset: {train_samples} training samples, "
                      f"world_size={world_size}")
    accelerator.print(f"Estimated {steps_per_epoch} steps/epoch "
                      f"(samples_per_rank={samples_per_rank}), "
                      f"{num_training_steps} total training steps, "
                      f"warmup={int(args.warmup_rates * num_training_steps)}")
    accelerator.print("Frames not shuffled (consecutive frames of each episode)" if sampler.window is None else
                      "Frames shuffled over the whole epoch" if sampler.window == 0 else
                      f"Frames shuffled in windows of {sampler.window} frames of the epoch's episode order")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # ── Resume: weights + optimizer, data position, step counter, LR ──
    # The data position (epoch, frames of it consumed) is kept in frames, and
    # the step counter and LR are re-derived for this run's ranks x batch size,
    # so a run can resume with other ones: the sampler continues at the same
    # frame and the LR at the same point of the schedule (same fraction of the
    # training data). The step counter comes from training_state.json of a full
    # checkpoint, else from --resume_step (one-time override for legacy
    # weights-only checkpoints). Dir-name parsing is NOT used automatically
    # because `checkpoint-0-125000` may mean "weights-only init from some other
    # run", not "resume step 125000".
    global_batch = sampler.global_batch
    ga = accelerator.gradient_accumulation_steps
    resume_global_step = 0
    start_epoch, start_sample, samples_seen = 0, 0, 0
    if args.resume_checkpoint:
        saved_step = 0
        scheduler_restored = False
        if resume_state_dir is not None:
            accelerator.print(f"Loading full resumable state from {resume_state_dir}")
            if repartition:
                _load_zero_repartitioned(model, resume_state_dir, accelerator)
            else:
                # PyTorch 2.6 flipped torch.load's default to weights_only=True,
                # but older DeepSpeed serializes classes (LossScaler, etc.) that
                # aren't on the safe allowlist. Force weights_only=False for the
                # duration of this call — the checkpoint was written by us and is
                # trusted.
                _orig_torch_load = torch.load
                def _torch_load_trusted(*a, **kw):
                    kw["weights_only"] = False
                    return _orig_torch_load(*a, **kw)
                torch.load = _torch_load_trusted
                try:
                    accelerator.load_state(resume_state_dir)
                finally:
                    torch.load = _orig_torch_load
                # Restore LR scheduler from our explicit scheduler.bin (Accelerate
                # + DeepSpeed does not save it when prepared separately).
                sched_path = os.path.join(resume_state_dir, "scheduler.bin")
                if os.path.isfile(sched_path):
                    sched_sd = torch.load(sched_path, map_location="cpu",
                                          weights_only=False)
                    lr_scheduler.load_state_dict(sched_sd)
                    scheduler_restored = True
            if resume_tstate:
                saved_step = int(resume_tstate.get("global_step", 0))
                accelerator.print(f"Full resume: global_step={saved_step} from training_state.json")
            else:
                _, parsed = _parse_step_from_ckpt_dir(resume_ckpt_dir)
                saved_step = parsed or 0
                accelerator.print(
                    f"Full resume: no training_state.json; global_step from dir name={saved_step}")
        if args.resume_step > 0:
            saved_step = args.resume_step
            accelerator.print(f"--resume_step override: global_step={saved_step}")

        # Where the saved run was in the data, in frames.
        saved_bsz = int(resume_tstate.get("train_bsz_per_gpu") or args.train_bsz_per_gpu)
        saved_global = (saved_world or accelerator.num_processes) * saved_bsz
        if "data_epoch" in resume_tstate and args.resume_step <= 0:
            start_epoch = int(resume_tstate["data_epoch"])
            start_sample = int(resume_tstate["data_sample"])
            samples_seen = int(resume_tstate.get("samples_seen", 0))
        else:   # older checkpoints: from the saving run's step counter and batch
            steps_then = max(1, sampler.total_frames // saved_global)
            start_epoch = saved_step // steps_then
            start_sample = (saved_step % steps_then) * saved_global
            samples_seen = saved_step * saved_global
        frames_then = resume_tstate.get("dataset_frames")
        if frames_then is not None and int(frames_then) != sampler.total_frames:
            accelerator.print(
                f"WARNING: the checkpoint's run had {frames_then} training frames, this one "
                f"{sampler.total_frames}: the data changed, the resume position is approximate.")
        if sampler.steps(start_sample) == 0:          # that epoch was used up
            start_epoch, start_sample = start_epoch + 1, 0
        resume_global_step = start_epoch * steps_per_epoch + start_sample // global_batch

        if not (scheduler_restored and resume_global_step == saved_step):
            # Other ranks x batch size (or no scheduler.bin): put the LR at the
            # same fraction of the schedule as the saved run.
            _set_scheduler_step(lr_scheduler, resume_global_step // ga)
        if resume_global_step != saved_step:
            accelerator.print(
                f"Saved with {saved_global} frames/step (step {saved_step}), now "
                f"{global_batch} frames/step: global_step={resume_global_step}.")
        accelerator.print(f"LR at resume: {lr_scheduler.get_last_lr()[0]:.3e}")
        if not args.resume_skip_data:
            accelerator.print("--resume_skip_data 0: the data restarts from the beginning "
                              "of the epoch (some overlap).")
            start_sample = 0
        # The first start_sample frames of the epoch keep the order they were taken in; a new
        # --shuffle_window applies to the frames of the epoch not taken yet.
        order_msg = sampler.resume_order(resume_tstate.get("data_order"), start_epoch, start_sample)
        if order_msg:
            accelerator.print(order_msg)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    mem_monitor = MemoryMonitor(device=torch.cuda.current_device())
    global_step = resume_global_step

    # ── EMA of the weights (after the resumed weights are in place) ──
    ema = None
    if args.ema_decay > 0:
        ema = ShardedCPUEMA(accelerator.unwrap_model(model), args.ema_decay, args.ema_every,
                            accelerator.process_index, accelerator.num_processes)
        loaded = ema.load(resume_ckpt_dir) if args.resume_checkpoint else False
        accelerator.print(
            f"EMA of the weights: decay {args.ema_decay}/step, folded in every {args.ema_every} steps, "
            f"{f'continued from {resume_ckpt_dir} ({ema.n} steps)' if loaded else 'starting from the current weights'}; "
            f"rank {accelerator.process_index} keeps {ema.numel / 1e9:.2f} B weights ({ema.numel * 4 / 2**30:.1f} GiB) in RAM")
    # ── best validation weights, shared by the jobs of this experiment ──
    n_val_frames = 0
    if val_dataloader is not None:
        n_val_frames = min(len(val_dataset), args.max_val_batches * args.train_bsz_per_gpu * accelerator.num_processes)
    best_val = None
    if args.keep_best_val and val_dataloader is not None and accelerator.is_main_process:
        best_val = _read_best_val(args, n_val_frames)
        accelerator.print(f"Best validation weights -> {_best_val_dir(args)} (best val/action_loss so far: "
                          f"{best_val['val_action_loss']:.6f})")
    model.train()

    device = accelerator.device

    if start_epoch >= args.n_epochs:
        accelerator.print(
            f"Resume position (epoch {start_epoch}) is at/past the end of training "
            f"({args.n_epochs} epochs). Nothing to do.")
        return
    if args.resume_checkpoint:
        accelerator.print(
            f"Resuming at epoch={start_epoch}, frame {start_sample}/{sampler.total_frames} of it, "
            f"global_step={global_step}, {samples_seen} frames trained on so far.")

    for epoch in range(start_epoch, args.n_epochs):
        sampler.set_epoch(epoch)

        # Data-skip for resume: drop the already-consumed portion of this
        # epoch at the *sampler* level so workers never load those samples.
        # Previously we iterated and `continue`d, which still ran image
        # loading + collate for every skipped batch — slow enough that GPUs
        # stayed at 0% and the NCCL watchdog could kill the job.
        sampler.start_sample = start_sample if epoch == start_epoch else 0
        epoch_sample = sampler.start_sample          # frames of this epoch consumed, all ranks
        epoch_total = sampler.steps()
        if epoch_sample > 0:
            accelerator.print(
                f"Skipping the first {epoch_sample} frames of epoch {epoch} via the sampler "
                f"(no data loaded). Epoch will run {epoch_total} steps.")

        from tqdm import tqdm
        desc = (f"Epoch {epoch} (resumed at frame {epoch_sample})"
                if epoch_sample > 0 else f"Epoch {epoch}")
        it = (tqdm(dataloader, total=epoch_total, desc=desc)
              if accelerator.is_main_process else dataloader)

        for batch in it:
            # Move batch to device (since dataloader is not prepared by accelerate)
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            raw_model = accelerator.unwrap_model(model)

            inputs_embeds = raw_model.prepare_inputs_embeds(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )

            n_slow_imgs = batch["n_slow_images"]
            grid_thw = batch.get("image_grid_thw")
            B = inputs_embeds.shape[0]
            if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
                merge = getattr(raw_model.visual, "spatial_merge_size",
                                getattr(dataset.processor.image_processor, "merge_size", 2))
                n_imgs_per_sample = grid_thw.shape[0] // B
                n_slow_img_tokens = sum(
                    int(g[0] * (g[1] // merge) * (g[2] // merge))
                    for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs]
                )
                slow_embeds, fast_embeds = split_slow_fast_embeds(
                    inputs_embeds, batch["input_ids"],
                    raw_model.image_token_id, n_slow_img_tokens)
            else:
                slow_embeds = inputs_embeds
                fast_embeds = inputs_embeds[:, :0]

            L_slow = slow_embeds.shape[1]

            pos_ids, _ = raw_model.get_rope_index(
                input_ids=batch["input_ids"],
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch["attention_mask"],
            )
            pos_ids = pos_ids[:, :, :L_slow]

            # ── Flare: append query tokens to latent expert ──
            if use_flare:
                flare_q = raw_model.flare_queries.expand(B, -1, -1).to(
                    device=slow_embeds.device, dtype=slow_embeds.dtype)
                slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
                pos_ids = extend_position_ids_for_flare(pos_ids, K)
                L_latent = slow_embeds_ext.shape[1]
            else:
                slow_embeds_ext = slow_embeds
                L_latent = L_slow

            if args.use_robot_state and batch["state_raw"] is not None:
                state_vec = batch["state_raw"].to(slow_embeds.device, dtype=slow_embeds.dtype)
                state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
            else:
                state_embeds = torch.empty(
                    (B, 0, slow_embeds.shape[2]),
                    device=slow_embeds.device, dtype=slow_embeds.dtype)
            n_state = state_embeds.shape[1]

            noisy_actions = raw_model.x_embedder(
                batch["noisy_actions"].to(slow_embeds.dtype))
            timesteps = raw_model.t_embedder(
                batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)

            chunk = args.action_chunk
            target = batch["target"].to(slow_embeds.dtype)

            # Stage 1: latent + action expert only (no tactile)
            full_embeds = torch.cat([
                slow_embeds_ext,
                fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)

            n_fast = fast_embeds.shape[1]
            n_action = n_fast + n_state + 1 + chunk
            L_total = full_embeds.shape[1]
            latent_indexes = torch.arange(0, L_latent, device=full_embeds.device)
            action_indexes = torch.arange(L_latent, L_total, device=full_embeds.device)
            tactile_indexes = torch.arange(0, 0, device=full_embeds.device)

            if global_step == 0 and accelerator.is_main_process:
                print(f"\n[Layout] slow={L_slow} flare={K} fast={n_fast} "
                      f"state={n_state} t=1 act={chunk} total={L_total}")

            outputs = model.model(
                inputs_embeds=full_embeds,
                position_ids=pos_ids,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=use_flare and (flare_layer_idx != -1),
                latent_indexes=latent_indexes,
                action_indexes=action_indexes,
                tactile_indexes=tactile_indexes,
            )
            hidden = outputs.last_hidden_state

            act_pred_start = L_latent + n_fast + n_state + 1
            v_act = raw_model.final_layer(
                hidden[:, act_pred_start: act_pred_start + chunk, :])
            loss_act = nn.MSELoss()(v_act, target)

            # ── Flare loss ──
            loss_flare = 0.0
            if use_flare and batch["flare_pixel_values"] is not None:
                if flare_layer_idx == -1:
                    flare_source = hidden
                else:
                    all_hs = outputs.hidden_states
                    n_layers = len(all_hs) - 1
                    li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
                    li = max(0, min(li, n_layers - 1))
                    flare_source = all_hs[li + 1]

                flare_hidden = flare_source[:, L_slow: L_slow + K, :]
                flare_pred = raw_model.flare_proj(flare_hidden)

                f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
                f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)

                with torch.no_grad():
                    vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                    features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

                    merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                    frame_feats = []
                    offset = 0
                    for g in f_thw:
                        n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                        frame_tokens = features[offset: offset + n_tok]
                        pooled = F.adaptive_avg_pool1d(
                            frame_tokens.unsqueeze(0).permute(0, 2, 1),
                            T_per_frame,
                        ).permute(0, 2, 1).squeeze(0)
                        frame_feats.append(pooled)
                        offset += n_tok
                    flare_targets = torch.stack(frame_feats).view(B, K, -1)

                pred_norm = F.normalize(flare_pred, dim=-1)
                tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
                loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

            loss = loss_act
            if torch.is_tensor(loss_flare):
                loss = loss + args.flare_loss_weight * loss_flare

            metric.update(loss, loss_act, loss_flare)
            accelerator.backward(loss)
            epoch_sample += global_batch
            samples_seen += global_batch

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.step()

                m = metric.get_metric()
                mem = mem_monitor.collect(accelerator.is_main_process)   # all ranks: all_reduce
                if accelerator.is_main_process:
                    lr_now = lr_scheduler.get_last_lr()[0]
                    if hasattr(it, "set_postfix"):
                        it.set_postfix(epoch=epoch, step=global_step,
                                       loss=f"{m['total_loss']:.6f}",
                                       act=f"{m['action_loss']:.6f}",
                                       flare=f"{m['flare_loss']:.6f}",
                                       lr=f"{lr_now:.2e}")
                    wandb.log({
                        "total_loss": m["total_loss"],
                        "action_loss": m["action_loss"],
                        "flare_loss": m["flare_loss"],
                        "lr": lr_now,
                        "epoch": epoch,
                        "samples_seen": samples_seen,
                        **mem,
                    }, step=global_step)

            # ── Validation ──
            if (val_dataloader is not None
                    and args.val_freq > 0
                    and (global_step + 1) % args.val_freq == 0):
                if ema is not None:
                    ema.wait()                # no EMA update running while validation forks its workers
                val_m = run_validation(
                    model, val_dataloader, accelerator,
                    lambda: accelerator.unwrap_model(model),
                    use_flare, K, T_per_frame, flare_layer_idx, args)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"  [Val step={global_step}] "
                        f"loss={val_m['val/total_loss']:.6f} "
                        f"act={val_m['val/action_loss']:.6f} "
                        f"flare={val_m['val/flare_loss']:.6f}")
                    wandb.log(val_m, step=global_step)
                    if best_val is not None and val_m["val/action_loss"] < best_val["val_action_loss"]:
                        best_val = _save_best_val(model, processor, accelerator, args, dataset, epoch,
                                                  global_step + 1, val_m, n_val_frames, sampler.total_frames)
                        accelerator.print(f"  best validation so far: weights in {best_val['checkpoint']}")

            # ── Step-level checkpoint ──
            if (args.save_steps > 0
                    and (global_step + 1) % args.save_steps == 0):
                accelerator.wait_for_everyone()
                save_checkpoint(model, processor, accelerator, args,
                                epoch, global_step + 1, dataset,
                                save_full_state=bool(args.save_full_state),
                                lr_scheduler=lr_scheduler, ema=ema,
                                position={"data_epoch": epoch, "data_sample": epoch_sample,
                                          "data_order": sampler.order_state(),
                                          "samples_seen": samples_seen,
                                          "dataset_frames": sampler.total_frames})

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            accelerator.wait_for_everyone()
            save_checkpoint(model, processor, accelerator, args,
                            epoch, global_step, dataset,
                            save_full_state=bool(args.save_full_state),
                            lr_scheduler=lr_scheduler, ema=ema,
                            position={"data_epoch": epoch + 1, "data_sample": 0,
                                      "data_order": sampler.order_state(epoch + 1),
                                      "samples_seen": samples_seen,
                                      "dataset_frames": sampler.total_frames})

    if ema is not None:
        ema.wait()
    for t in _background:                 # weights still being copied to weights/
        t.join()
    accelerator.print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="qwen3vl_egodex_pretrain_flare")
    parser.add_argument("--run_name", type=str, default="run_1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="",
                        help="EgoDex episode root (required for --data_format json).")
    parser.add_argument("--data_format", type=str, default="json", choices=["json", "lerobot"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--max_ckpts", type=int, default=5)

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--train_bsz_per_gpu", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_rates", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_frames", type=int, default=0,
                        help="1 = shuffle frames across the episodes of each rank's share "
                             "(recommended for LeRobot data); 0 = consecutive frames per episode.")
    parser.add_argument("--shuffle_window", type=int, default=EpisodeGroupedSampler.SHUFFLE_WINDOW,
                        help="With --shuffle_frames 1: frames are shuffled within consecutive windows of "
                             "this many frames of the epoch's episode order (0 = over the whole epoch). "
                             "Changing it when resuming mid-epoch reshuffles only the frames of the "
                             "epoch not taken yet (each frame still once per epoch).")

    parser.add_argument("--action_dim", type=int, default=62)
    parser.add_argument("--hand_dim", type=int, default=22,
                        help="Hand joints per arm (22 Sharpa Wave, 12 XHand). "
                             "Each arm block is [arm 9D | hand hand_dim].")
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to a model.pt file or to a checkpoint dir. "
                             "If the dir contains a `state/` subfolder, full "
                             "optimizer+scheduler+RNG state is restored.")
    parser.add_argument("--resume_step", type=int, default=0,
                        help="Explicit step to resume at (overrides any saved "
                             "value). Use this with a weights-only checkpoint "
                             "(e.g. the legacy checkpoint-0-30000) to fast-"
                             "forward the LR scheduler and wandb step counter.")
    parser.add_argument("--resume_skip_data", type=int, default=0,
                        help="If 1, fast-forward the dataloader past the "
                             "already-consumed portion of the first epoch on "
                             "resume. If 0, data iteration restarts from the "
                             "beginning of the epoch (some overlap).")
    parser.add_argument("--save_full_state", type=int, default=1,
                        help="If 1 (default), every checkpoint also writes a "
                             "`state/` subdir with optimizer+scheduler+RNG "
                             "state for exact resume. Set to 0 to save disk.")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save checkpoint every N steps (0=disable, epoch-only)")

    # Validation
    parser.add_argument("--val_ratio", type=float, default=0.02,
                        help="Fraction of episodes to hold out for validation (0=disable)")
    parser.add_argument("--val_freq", type=int, default=1000,
                        help="Run validation every N training steps (0=disable)")
    parser.add_argument("--max_val_batches", type=int, default=50,
                        help="Max batches per validation run")
    parser.add_argument("--val_num_workers", type=int, default=-1,
                        help="DataLoader workers per GPU for validation (-1 = num_workers // 2). They are started "
                             "for every validation and each costs several GiB of RAM while it runs.")
    parser.add_argument("--val_fixed", type=int, default=0,
                        help="1 = every validation scores the same max_val_batches x bsz x ranks frames, spread "
                             "over the whole val split, with the same noise and flow times (comparable losses). "
                             "0 = the first frames of the val split, new noise each time.")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="> 0: keep an EMA of the trainable weights with this decay per optimizer step (e.g. "
                             "0.9999, ~10k steps), in CPU RAM split across the ranks; checkpoints get model_ema.pt.")
    parser.add_argument("--ema_every", type=int, default=10,
                        help="Fold the weights into the EMA every this many optimizer steps (decay**every each time).")
    parser.add_argument("--keep_weights_every", type=int, default=0,
                        help="> 0: also copy the weights (model.pt, model_ema.pt, no optimizer state) of every "
                             "checkpoint at a multiple of this step to <run>/weights/, never rotated away. "
                             "A multiple of --save_steps.")
    parser.add_argument("--keep_best_val", type=int, default=0,
                        help="1: keep the weights with the lowest val/action_loss in <output_dir>/<experiment>/"
                             "best_val/ (shared by the jobs of the experiment; use with --val_fixed 1).")

    # Flare
    parser.add_argument("--use_flare", type=int, default=1)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=4)
    parser.add_argument("--n_flare_steps", type=int, default=8)
    parser.add_argument("--flare_loss_weight", type=float, default=0.5)
    parser.add_argument("--flare_frame_stride", type=int, default=4)
    parser.add_argument("--flare_layer_index", type=int, default=-1)
    add_lerobot_args(parser)

    args = parser.parse_args()

    # VS Code attach point (no-op unless TREX_DEBUGPY_PORT is set; see scripts/pretrain_debug.sh).
    from utils.debug_attach import maybe_wait_for_debugger
    maybe_wait_for_debugger()

    if args.data_format == "lerobot":
        if not args.lerobot_root:
            parser.error("--data_format lerobot requires --lerobot_root")
    elif not args.data_root:
        parser.error("--data_format json requires --data_root")
    if args.action_dim % (9 + args.hand_dim) != 0:
        parser.error(f"--action_dim {args.action_dim} is not a multiple of "
                     f"9 + --hand_dim ({9 + args.hand_dim})")

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
