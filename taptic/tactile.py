"""XHand fingertip forces -> T-REX tactile tensors ([5,3] current, [16,5,3] history).

Force source: SDK ``calc_force`` per fingertip from the native sidecar
(``tactile_raw/episode_XXXXXX_right.parquet``, column ``force`` = 15 values,
order thumb, index, middle, ring, little x (x, y, z)).  It is the same value the
recorder copied into ``observation.state[19:34]`` for the frame's
``is_frame_sample`` (verified equal).  Units are raw SDK counts (int8 Fx/Fy,
uint8 Fz per data_type.hpp); no counts->newton calibration is documented, so
no unit conversion is applied.  Fingertip-local axes, undocumented.

Timing (all on CLOCK_MONOTONIC, shared by recorder and EtherCAT worker):
  * a sample is available at ``read_end_ns``;
  * a frame's observation time is ``observation_ns`` (frames parquet);
  * the CURRENT tactile observation of a frame is the latest sample with
    read_end_ns <= observation_ns (causal; median age 6.7 ms).  This is fresher
    than the ``is_frame_sample`` copied into observation.state in 89% of frames;
  * a HISTORY window is the W latest samples up to the same anchor, i.e. native
    rate (~76.4 Hz, W=16 -> ~209 ms), never extended past the anchor.
  * windows never cross a tactile re-zero (``reset_sequence`` change) nor reach
    into the pre-roll recorded before the episode's first frame (unless
    ``allow_preroll``); missing history is left-padded by repeating the oldest
    valid sample, as T-REX pads with the episode's first frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .constants import N_FINGERS, TAC_DIM, TAC_FLAT, VQVAE_WINDOW


@dataclass
class EpisodeTactile:
    t_ns: np.ndarray        # [M] int64 read_end_ns (sorted)
    force: np.ndarray       # [M,5,3] float32 raw counts
    reset: np.ndarray       # [M] int64 reset_sequence
    start_idx: int          # first sample allowed in windows (episode start)

    @classmethod
    def from_frame(cls, samples, first_obs_ns: int, allow_preroll: bool = False) -> "EpisodeTactile":
        t = samples.read_end_ns.to_numpy(np.int64)
        if np.any(np.diff(t) <= 0):
            order = np.argsort(t, kind="stable")
            samples = samples.iloc[order]
            t = t[order]
        f = np.stack(samples.force.to_numpy()).astype(np.float32).reshape(-1, N_FINGERS, TAC_DIM)
        r = samples.reset_sequence.to_numpy(np.int64)
        start = 0 if allow_preroll else max(0, int(np.searchsorted(t, first_obs_ns, side="right")) - 1)
        return cls(t_ns=t, force=f, reset=r, start_idx=start)

    def latest_index(self, anchor_ns: int) -> int:
        """Index of the latest sample available at ``anchor_ns`` (>= start_idx), or -1."""
        i = int(np.searchsorted(self.t_ns, anchor_ns, side="right")) - 1
        return i if i >= self.start_idx else -1

    def window_indices(self, end_idx: int, window: int = VQVAE_WINDOW) -> np.ndarray:
        """``window`` sample indices ending at ``end_idx`` (left-padded)."""
        if end_idx < 0:
            raise ValueError("no causal tactile sample at this anchor")
        lo = max(self.start_idx, end_idx - window + 1)
        same = self.reset[lo:end_idx + 1] == self.reset[end_idx]
        # first index of the trailing run with the same reset_sequence
        run = np.flatnonzero(~same)
        first_ok = lo + (int(run[-1]) + 1 if len(run) else 0)
        idx = np.arange(first_ok, end_idx + 1)
        if len(idx) < window:
            idx = np.r_[np.full(window - len(idx), idx[0]), idx]
        return idx

    def current(self, anchor_ns: int) -> np.ndarray:
        i = self.latest_index(anchor_ns)
        if i < 0:
            raise ValueError("no causal tactile sample at this anchor")
        return self.force[i]

    def history(self, anchor_ns: int, window: int = VQVAE_WINDOW) -> np.ndarray:
        return self.force[self.window_indices(self.latest_index(anchor_ns), window)]


# ── normalization (T-REX q01/q99 min-max to [-1, 1], with clipping) ─────────
@dataclass
class TactileNorm:
    q01: np.ndarray       # [15]
    q99: np.ndarray       # [15]
    mask: np.ndarray      # [15] bool
    degenerate: np.ndarray  # [15] bool: q99-q01 < min_range in the training split

    @classmethod
    def fit(cls, flat: np.ndarray, min_range: float = 1.0) -> "TactileNorm":
        """``flat``: [N,15] raw forces from TRAINING episodes only.

        Channels whose q99-q01 is below one sensor count (a finger that is
        practically never in contact in the training data, e.g. middle/ring/
        little in insert_credit_card) would make T-REX's formula divide by
        ~1e-8 and turn any non-zero reading into +-1.  For those channels the
        range is set to [median-1, median+1] counts (zero force -> ~0).
        """
        flat = np.asarray(flat, dtype=np.float64).reshape(-1, TAC_FLAT)
        q01 = np.quantile(flat, 0.01, axis=0)
        q99 = np.quantile(flat, 0.99, axis=0)
        deg = (q99 - q01) < min_range
        med = np.median(flat, axis=0)
        q01 = np.where(deg, med - 1.0, q01)
        q99 = np.where(deg, med + 1.0, q99)
        return cls(q01.astype(np.float32), q99.astype(np.float32),
                   np.ones(TAC_FLAT, dtype=bool), deg)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Exactly T-REX ``_normalize`` on the flattened 15 channels."""
        shp = x.shape
        flat = np.asarray(x, dtype=np.float32).reshape(-1, TAC_FLAT)
        out = np.where(self.mask, np.clip(2 * (flat - self.q01) / (self.q99 - self.q01 + 1e-8) - 1, -1, 1), flat)
        return out.reshape(shp).astype(np.float32)

    def to_dict(self) -> Dict:
        return {"q01": self.q01.tolist(), "q99": self.q99.tolist(), "mask": self.mask.tolist(),
                "degenerate": self.degenerate.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "TactileNorm":
        return cls(np.asarray(d["q01"], np.float32), np.asarray(d["q99"], np.float32),
                   np.asarray(d["mask"], bool), np.asarray(d.get("degenerate", [False] * TAC_FLAT), bool))

    # VQ-VAE checkpoint schema (tactile_vqvae TacF6Stats / model buffers)
    def to_vqvae_stats(self) -> Dict:
        return {"tacf6_min": self.q01.tolist(), "tacf6_max": self.q99.tolist(), "tacf6_mask": self.mask.tolist()}


def magnitude(window_raw: np.ndarray) -> float:
    """T-REX VQ-VAE loss weighting input: L2 norm of the raw window."""
    return float(np.linalg.norm(window_raw))
