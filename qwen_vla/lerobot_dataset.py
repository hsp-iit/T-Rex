"""TRexLeRobotDataset — load LeRobot v3.0 datasets and emit the EXACT batch
contract the T-Rex trainers already consume (drop-in for SftDataset, the
midtrain dataset and the pretrain dataset).

The model and the cascaded-flow training loop are unchanged: this dataset +
`collate_fn` reproduce every key the JSON pipelines return
(input_ids/pixel_values/noisy_actions/target/timesteps/tactile_*/flare_*/...),
sourcing frames from LeRobot instead.

Two action schemas are accepted (`--lerobot_action_format`, default auto):
  * chunk — the T-Rex schema written by utils/convert_*_to_lerobot.py:
            `action` is the baked delta-base chunk [ACTION_CHUNK, D] and
            `observation.state` the 9D-arm state [D].
  * eef   — absolute end-effector sources (egomix retargets, Panda/XHand
            recordings): `action` is one row per frame, per arm
            [x, y, z, rotation vector | hand_dim joints]. The delta-base chunk
            and the 9D-arm state are built on the fly (utils/lerobot_common.py).
            `--lerobot_base action`: chunk base = action[t], targets
            action[t+1 .. t+chunk] (a pose per video frame, e.g. retargeted human
            video). `--lerobot_base state`: base = the EE pose in
            observation.state at t, targets action[t .. t+chunk-1] (robot
            recordings, where action[t] is reached at t+1).

Design:
  * Several roots (e.g. one per human-video source) are concatenated; each may
    name its head camera differently. A root with a single video uses it as head.
  * Fast images are `--lerobot_fast_keys` in order; a key a root lacks is
    replaced by its head frame (the JSON pretrain does the same for human video).
  * Temporal windows (action targets, FLARE future head frames, F6 history,
    delayed tactile) come from LeRobot `delta_timestamps`, clamped to the
    episode like the JSON pipelines.
  * Normalization uses the q01/q99 `meta/trex_norm_stats.json` sidecar of a
    single root, or is computed from the parquet tables (no video) otherwise.
    Computed stats are cached in a JSON file (`--lerobot_stats_path`, default
    <output_dir>/lerobot_norm_stats.json) together with the data and settings
    they were built from; a later run with the same ones loads the file. Only
    one process per node computes, the others wait for the file.
  * `_cum_frames` / `_num_episodes` follow episode order, so EpisodeGroupedSampler
    (pretrain, midtrain) groups frames by episode.
  * The LeRobot import is lazy and `collate_fn` consumes plain item dicts, so this
    module imports (and the collate is testable) without lerobot installed.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import socket
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from utils.lerobot_common import (
    ACTION_DIM, ACTION_CHUNK, STATS_KEY, EE_SRC_DIM,
    KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L, KEY_STATE, KEY_ACTION,
    KEY_TACF6, DEFORM_KEYS, NORM_STATS_FILENAME, calculate_stats, eef_to_trex,
    load_norm_stats,
)

import cv2  # for the tracking-error-noise rotation conversions


def add_lerobot_args(parser):
    """CLI options shared by pretrain.py, midtrain.py and train.py."""
    g = parser.add_argument_group("LeRobot data (--data_format lerobot)")
    g.add_argument("--lerobot_root", type=str, nargs="+", default=[],
                   help="One or more LeRobot v3.0 dataset dirs, or dirs that contain them.")
    g.add_argument("--lerobot_repo_id", type=str, default="",
                   help="Optional repo_id for a single root; defaults to basename(root).")
    g.add_argument("--lerobot_head_key", type=str, default=KEY_HEAD,
                   help="Head (slow) camera key; a root with a single video uses it.")
    g.add_argument("--lerobot_fast_keys", type=str, nargs="*", default=[KEY_WRIST_R, KEY_WRIST_L],
                   help="Fast camera keys in prompt order; a missing key is replaced by the head frame.")
    g.add_argument("--lerobot_action_format", choices=["auto", "chunk", "eef"], default="auto")
    g.add_argument("--lerobot_base", choices=["action", "state"], default="action",
                   help="eef only: where the chunk-base pose comes from.")
    g.add_argument("--lerobot_state_ee_index", type=int, default=0,
                   help="eef + base state: first of the n_arms*6 EE values in observation.state.")
    g.add_argument("--lerobot_state_hand_index", type=int, default=6,
                   help="eef + base state: first of the n_arms*hand_dim hand joints in observation.state.")
    g.add_argument("--lerobot_tactile_key", type=str, default=KEY_TACF6,
                   help="Feature holding fingertip forces, e.g. observation.state for TAPTIC.")
    g.add_argument("--lerobot_tactile_index", type=int, default=0,
                   help="First force value inside that feature (TAPTIC: 19).")
    g.add_argument("--lerobot_tactile_fingers", type=int, default=10,
                   help="Fingertips stored there (T-Rex 10, TAPTIC 5).")
    g.add_argument("--lerobot_tactile_axes", type=int, choices=[3, 6], default=6,
                   help="Values per fingertip: 6 (F + M) or 3 (F only; moments padded with zeros).")
    g.add_argument("--lerobot_video_backend", type=str, default="",
                   help="LeRobot video decoder (torchcodec, pyav); empty = LeRobot default. "
                        "Use pyav when FFmpeg shared libraries are not available for torchcodec.")
    g.add_argument("--image_fit", choices=["stretch", "crop", "pad"], default="stretch",
                   help="How frames reach --image_size: stretch, centre crop, or letterbox.")
    g.add_argument("--lerobot_stats_path", type=str, default="",
                   help="Normalization-stats cache (JSON). Loaded when it was built from the same "
                        "data and settings, otherwise computed and written there. Overrides a "
                        "root's meta/trex_norm_stats.json. Default: <output_dir>/"
                        f"{STATS_CACHE_NAME} (none when there is no --output_dir).")
    if not any(a.dest == "hand_dim" for a in parser._actions):
        parser.add_argument("--hand_dim", type=int, default=22,
                            help="Hand joints per arm (22 Sharpa Wave, 12 XHand).")
    return g


def find_lerobot_roots(roots) -> List[str]:
    """Every LeRobot dataset dir (has meta/info.json) under the given roots."""
    if isinstance(roots, str):
        roots = [roots]
    found = []
    for r in roots:
        if os.path.isfile(os.path.join(r, "meta", "info.json")):
            found.append(r)
        else:
            found += sorted(os.path.dirname(os.path.dirname(p))
                            for p in glob.glob(os.path.join(r, "*", "meta", "info.json")))
    if not found:
        raise FileNotFoundError(f"no LeRobot dataset (meta/info.json) under {roots}")
    return found


# ── state-noise helpers (mirror scripts/train.py) ────────────────────────────
def _rot6d_to_mat(rot6d):
    col1, col2 = rot6d[:3], rot6d[3:6]
    return np.column_stack([col1, col2, np.cross(col1, col2)])


def _arm9d_to_axis_angle(arm_9d):
    R = _rot6d_to_mat(arm_9d[3:9])
    aa, _ = cv2.Rodrigues(R)
    return np.concatenate([arm_9d[:3], aa.flatten()])


def _axis_angle_to_arm9d(arm_aa):
    R, _ = cv2.Rodrigues(arm_aa[3:6])
    return np.concatenate([arm_aa[:3], R[:, 0], R[:, 1]])


def add_tracking_error_noise(state, te_mean, te_std, action_dim, hand_dim=22):
    arm_w, te_w = 9 + hand_dim, 6 + hand_dim
    noisy = state.copy()
    for arm_idx in range(action_dim // arm_w):
        off = arm_idx * arm_w
        arm_aa = _arm9d_to_axis_angle(state[off:off + 9])
        te_off = arm_idx * te_w
        noise_arm = np.random.normal(te_mean[te_off:te_off + 6], te_std[te_off:te_off + 6]).astype(np.float32)
        noise_hand = np.random.normal(te_mean[te_off + 6:te_off + te_w], te_std[te_off + 6:te_off + te_w]).astype(np.float32)
        noisy[off:off + 9] = _axis_angle_to_arm9d(arm_aa + noise_arm)
        noisy[off + 9:off + arm_w] = state[off + 9:off + arm_w] + noise_hand
    return noisy


def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1), values)


# ── normalization-stats computation and cache ────────────────────────────────
STATS_CACHE_NAME = "lerobot_norm_stats.json"
# Bump when the stats math changes, so caches built by older code are rebuilt.
STATS_CACHE_VERSION = 1
# Frames converted per vectorized call when building chunk steps (bounds the
# float64 temporaries of the pose math to a few MB).
_STATS_BLOCK = 1 << 16
# How long the processes that don't compute wait for the cache file.
_STATS_WAIT_S = 3 * 3600


def _fill_from_arrow(dest: np.ndarray, column) -> None:
    """Copy an Arrow column of (nested) lists into dest [rows, *shape].

    Goes through the flat value buffer, so no per-row Python objects are made
    (pandas would hold one numpy array per row and cell)."""
    import pyarrow as pa
    pos = 0
    for arr in column.chunks:
        m = len(arr)
        while True:
            if isinstance(arr, pa.ExtensionArray):          # e.g. HF datasets Array2D
                arr = arr.storage
            elif (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)
                  or pa.types.is_fixed_size_list(arr.type)):
                arr = arr.flatten()                          # honours slice offsets
            else:
                break
        dest[pos:pos + m] = arr.to_numpy(zero_copy_only=False).reshape(dest[pos:pos + m].shape)
        pos += m


def _read_columns(root: str, spec: Dict[str, Tuple[tuple, np.dtype]]) -> Dict[str, np.ndarray]:
    """Read `spec` = {column: (per-row shape, dtype)} from every parquet file of a
    root into preallocated numpy arrays, one file at a time."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))
    n_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    out = {c: np.empty((n_rows,) + tuple(shape), dtype=dtype) for c, (shape, dtype) in spec.items()}
    pos = 0
    for p in files:
        table = pq.read_table(p, columns=list(spec))
        m = table.num_rows
        for c in spec:
            _fill_from_arrow(out[c][pos:pos + m], table.column(c))
        pos += m
        del table
    return out


def _read_stats_cache(path: str, settings: dict):
    """(stats, None) when `path` holds stats built with `settings`, else (None, reason)."""
    try:
        with open(path) as f:
            data = json.load(f)
        saved = data[next(iter(data))].get("settings")
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError, StopIteration, AttributeError, TypeError) as e:
        return None, f"is unreadable ({e})"
    want = json.loads(json.dumps(settings))                 # tuples -> lists, like the file
    if saved != want:
        saved = saved if isinstance(saved, dict) else {}
        diff = sorted(k for k in set(saved) | set(want) if saved.get(k) != want.get(k))
        return None, f"was built with other data/settings ({', '.join(diff)})"
    return data, None


def _write_json_atomic(path: str, obj) -> None:
    """Write via a temp file + rename, so a reader never sees a partial file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


class _Source:
    """One LeRobot root: its feature layout and the per-root LeRobotDataset."""

    def __init__(self, root: str, cfg, repo_id: str):
        self.root = root
        self.repo_id = repo_id
        with open(os.path.join(root, "meta", "info.json")) as f:
            info = json.load(f)
        self.fps = int(info["fps"])
        self.feats = info["features"]
        # Recorded in the stats cache, so a dataset that changed is not matched.
        self.total_frames = info.get("total_frames")
        self.total_episodes = info.get("total_episodes")
        videos = [k for k, v in self.feats.items() if v["dtype"] == "video"]
        if cfg.head_key in self.feats:
            self.head_key = cfg.head_key
        elif len(videos) == 1:
            self.head_key = videos[0]
        else:
            raise ValueError(f"{root}: head camera {cfg.head_key!r} not found; "
                             f"videos are {videos}, pass --lerobot_head_key")
        self.fast_keys = [k if k in self.feats else None for k in cfg.fast_keys]
        a_shape = self.feats[KEY_ACTION]["shape"]
        fmt = cfg.action_format
        if fmt == "auto":
            fmt = "chunk" if len(a_shape) == 2 else "eef"
        self.action_format = fmt
        self.src_dim = a_shape[-1]
        self.has_tactile = cfg.tactile_key in self.feats
        self.has_deform = DEFORM_KEYS[0] in self.feats
        self.ds = None

    def episode_lengths(self, episodes: Optional[List[int]]) -> np.ndarray:
        import pandas as pd
        files = sorted(glob.glob(os.path.join(self.root, "meta", "episodes", "*", "*.parquet")))
        eps = pd.concat([pd.read_parquet(p, columns=["episode_index", "length"]) for p in files])
        eps = eps.sort_values("episode_index")
        if episodes is not None:
            eps = eps[eps.episode_index.isin(episodes)]
        return eps.length.to_numpy(dtype=np.int64)

    def episode_indices(self) -> List[int]:
        import pandas as pd
        files = sorted(glob.glob(os.path.join(self.root, "meta", "episodes", "*", "*.parquet")))
        return sorted(pd.concat([pd.read_parquet(p, columns=["episode_index"]) for p in files])
                      .episode_index.astype(int).tolist())


class TRexLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, episodes: Optional[Dict[str, List[int]]] = None):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator

        roots = config.lerobot_root
        roots = find_lerobot_roots(roots if isinstance(roots, (list, tuple)) else [roots])
        self.roots = roots
        repo_id_cli = getattr(config, "lerobot_repo_id", "")

        # flow / flare / tactile config (mirror SftDataset)
        self.image_size = tuple(config.image_size) if getattr(config, "image_size", None) else None
        self.image_fit = getattr(config, "image_fit", "stretch")
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = int(getattr(config, "n_flare_steps", 0)) if self.use_flare else 0
        self.flare_stride = int(getattr(config, "flare_frame_stride", 1))
        self.use_tactile_vec = bool(getattr(config, "use_tactile_vec", 0))
        self.use_tactile_deform = bool(getattr(config, "use_tactile_deform", 0))
        self.use_tactile_vqvae = bool(getattr(config, "use_tactile_vqvae", 0))
        self.use_robot_state = bool(getattr(config, "use_robot_state", 0))
        self.vqvae_window = int(getattr(config, "vqvae_window", 16))
        self.delay_offsets = [int(d) for d in getattr(config, "tactile_delay_offsets", [0]) or [0]]
        self.action_dim = int(getattr(config, "action_dim", ACTION_DIM))
        self.action_chunk = int(getattr(config, "action_chunk", ACTION_CHUNK))
        self.hand_dim = int(getattr(config, "hand_dim", 22))
        self.base = getattr(config, "lerobot_base", "action")
        self.state_ee_index = int(getattr(config, "lerobot_state_ee_index", 0))
        self.state_hand_index = int(getattr(config, "lerobot_state_hand_index", 6))
        self.tac_key = getattr(config, "lerobot_tactile_key", KEY_TACF6)
        self.tac_index = int(getattr(config, "lerobot_tactile_index", 0))
        self.tac_fingers = int(getattr(config, "lerobot_tactile_fingers", 10))
        self.tac_axes = int(getattr(config, "lerobot_tactile_axes", 6))

        src_cfg = type("SrcCfg", (), dict(
            head_key=getattr(config, "lerobot_head_key", KEY_HEAD),
            fast_keys=list(getattr(config, "lerobot_fast_keys", [KEY_WRIST_R, KEY_WRIST_L])),
            action_format=getattr(config, "lerobot_action_format", "auto"),
            tactile_key=self.tac_key))
        self.sources = [
            _Source(r, src_cfg, repo_id_cli if (repo_id_cli and len(roots) == 1)
                    else os.path.basename(os.path.normpath(r)))
            for r in roots]
        self.fps = self.sources[0].fps
        for s in self.sources:
            if s.fps != self.fps:
                raise ValueError(f"all roots must share one fps; {s.root} is {s.fps}, expected {self.fps}")
            if s.action_format == "eef" and s.src_dim % (EE_SRC_DIM + self.hand_dim):
                raise ValueError(f"{s.root}: eef action width {s.src_dim} is not a multiple of "
                                 f"6 + hand_dim ({EE_SRC_DIM + self.hand_dim})")
        self.has_tactile = all(s.has_tactile for s in self.sources)
        self.has_deform = all(s.has_deform for s in self.sources)
        self.n_fast = len(src_cfg.fast_keys)

        # ── normalization stats ──
        if (not getattr(config, "lerobot_stats_path", "") and len(roots) == 1
                and os.path.isfile(os.path.join(roots[0], NORM_STATS_FILENAME))):
            self.stats_data = load_norm_stats(roots[0])
        else:
            self.stats_data = self._load_or_compute_stats()
        block = self.stats_data[next(iter(self.stats_data))]
        def _arr(key, sub):
            return np.array(block[key][sub])
        self.action_mask = _arr("action", "mask"); self.action_min = _arr("action", "q01"); self.action_max = _arr("action", "q99")
        self.state_mask  = _arr("state", "mask");  self.state_min  = _arr("state", "q01");  self.state_max  = _arr("state", "q99")
        if self.has_tactile and "tactile_f6" in block:
            self.tacf6_mask = _arr("tactile_f6", "mask"); self.tacf6_min = _arr("tactile_f6", "q01"); self.tacf6_max = _arr("tactile_f6", "q99")
        te = block.get("tracking_error", {})
        te_dim = (self.action_dim // (9 + self.hand_dim)) * (6 + self.hand_dim)
        self.te_mean = np.array(te.get("mean", np.zeros(te_dim)), dtype=np.float32)
        self.te_std  = np.array(te.get("std",  np.zeros(te_dim)), dtype=np.float32)

        self._build(episodes)
        accelerator.print(
            f"[LeRobot] {len(self.sources)} root(s), {self._num_episodes} episodes, "
            f"{len(self)} frames, fps={self.fps}, fast images={self.n_fast}, "
            f"tactile={self.has_tactile}; "
            + ", ".join(f"{s.repo_id}:{s.action_format}/head={s.head_key}" for s in self.sources))

    # ── LeRobot datasets + episode index ──
    def _build(self, episodes: Optional[Dict[str, List[int]]]):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        lengths, sizes = [], []
        self._selected = {}
        for s in self.sources:
            sel = None if episodes is None else episodes.get(s.root, [])
            if sel is not None and not sel:
                s.ds = None
                sizes.append(0)
                continue
            s.ds = LeRobotDataset(s.repo_id, root=s.root, episodes=sel,
                                  delta_timestamps=self._delta_timestamps(s),
                                  video_backend=getattr(self.config, "lerobot_video_backend", None) or None)
            self._selected[s.root] = sel
            lengths.append(s.episode_lengths(sel))
            sizes.append(len(s.ds))
        ep_lengths = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
        self._cum_frames = np.cumsum(ep_lengths)
        self._num_episodes = len(ep_lengths)
        self._src_cum = np.cumsum(sizes)
        if self._num_episodes and int(self._cum_frames[-1]) != int(self._src_cum[-1]):
            raise RuntimeError("episode lengths do not match the LeRobot frame count")

    def _delta_timestamps(self, s: _Source) -> Dict[str, list]:
        fps = s.fps
        dt = {s.head_key: [0.0] + [(k + 1) * self.flare_stride / fps for k in range(self.n_flare_steps)]}
        if s.action_format == "eef":
            start = 0 if self.base == "state" else 1
            offs = list(range(start, start + self.action_chunk))
            dt[KEY_ACTION] = ([0.0] if self.base == "action" else []) + [k / fps for k in offs]
        max_d = max(self.delay_offsets)
        if self._uses_tactile(s):
            W = self.vqvae_window
            dt[self.tac_key] = [(i - (W - 1)) / fps for i in range(W + max_d)]   # history + delays
        if s.has_deform and self.use_tactile_deform:
            for k in DEFORM_KEYS:
                dt[k] = [d / fps for d in self.delay_offsets]
        return dt

    def _uses_tactile(self, s: _Source) -> bool:
        return s.has_tactile and (self.use_tactile_vec or self.use_tactile_vqvae)

    def _tactile_rows(self, arr, s: _Source):
        """Force feature [..., *feature shape] -> T-Rex F6 [..., fingers, 6]; a
        3-axis source gets zero moments (masked out by the stats)."""
        n_feat = len(s.feats[self.tac_key]["shape"])
        arr = np.asarray(arr, dtype=np.float32)
        lead = arr.shape[:arr.ndim - n_feat]
        flat = arr.reshape(lead + (-1,))
        n = self.tac_fingers * self.tac_axes
        f = flat[..., self.tac_index:self.tac_index + n].reshape(lead + (self.tac_fingers, self.tac_axes))
        if self.tac_axes < 6:
            f = np.concatenate([f, np.zeros(lead + (self.tac_fingers, 6 - self.tac_axes), np.float32)], axis=-1)
        return f

    def _current(self, item, key, s: _Source):
        """Value of `key` at frame t; the tactile window holds it at index W-1."""
        v = item[key]
        if key == self.tac_key and self._uses_tactile(s):
            v = v[self.vqvae_window - 1]
        return v

    # ── normalization stats ──
    def _stats_cache_path(self) -> str:
        path = getattr(self.config, "lerobot_stats_path", "") or ""
        if path:
            return path
        out_dir = getattr(self.config, "output_dir", "") or ""
        return os.path.join(out_dir, STATS_CACHE_NAME) if out_dir else ""

    def _stats_settings(self) -> dict:
        """Everything `_compute_stats` depends on. A cache recorded with other
        settings is rebuilt instead of reused."""
        settings = {
            "version": STATS_CACHE_VERSION,
            "roots": sorted(({"root": os.path.realpath(s.root), "action_format": s.action_format,
                              "total_frames": s.total_frames, "total_episodes": s.total_episodes}
                             for s in self.sources), key=lambda r: r["root"]),
            "action_dim": self.action_dim,
        }
        if any(s.action_format == "eef" for s in self.sources):
            settings.update(action_chunk=self.action_chunk, hand_dim=self.hand_dim, base=self.base)
            if self.base == "state":
                settings.update(state_ee_index=self.state_ee_index,
                                state_hand_index=self.state_hand_index)
        if self.has_tactile:
            settings["tactile"] = {"key": self.tac_key, "index": self.tac_index,
                                   "fingers": self.tac_fingers, "axes": self.tac_axes}
        return settings

    def _load_or_compute_stats(self) -> dict:
        """Stats from the cache file when it matches, else computed and cached.

        Only the local main process of each node computes; the other processes
        poll for the file it writes (no NCCL barrier, so a long computation can't
        hit the collective timeout). With a shared filesystem a second node finds
        the file if it was already written, otherwise it computes too."""
        acc = self.accelerator
        path = self._stats_cache_path()
        if not path:
            return self._compute_stats()
        settings = self._stats_settings()
        stats, why = _read_stats_cache(path, settings)
        if stats is not None:
            acc.print(f"[LeRobot] normalization stats loaded from {path}")
            return stats

        if getattr(acc, "is_local_main_process", True):
            if why != "missing":
                acc.print(f"[LeRobot] stats cache {path} {why}; recomputing")
            stats = self._compute_stats()
            stats[next(iter(stats))]["settings"] = settings
            try:
                _write_json_atomic(path, stats)
            except OSError as e:
                if getattr(acc, "num_processes", 1) > 1:     # the others are waiting for it
                    raise RuntimeError(f"cannot write the stats cache {path} ({e}); "
                                       "pass a writable --lerobot_stats_path") from e
                acc.print(f"[LeRobot] WARNING: cannot write the stats cache {path} ({e})")
                return stats
            acc.print(f"[LeRobot] normalization stats cached to {path}")
            return stats

        deadline = time.time() + _STATS_WAIT_S
        while True:
            time.sleep(5)
            stats, _ = _read_stats_cache(path, settings)
            if stats is not None:
                return stats
            if time.time() > deadline:
                raise TimeoutError(f"no matching stats cache at {path} after {_STATS_WAIT_S} s; "
                                   "is it on storage this node's main process writes to?")

    def _stats_columns(self, s: _Source) -> dict:
        """One root's raw parquet columns for the stats, sorted by (episode, frame),
        plus `ep_last` [frames]: the row index of each frame's episode end."""
        def feat(key):
            f = s.feats[key]
            return tuple(f["shape"]), np.dtype(f["dtype"])

        spec = {"episode_index": ((), np.int64), "frame_index": ((), np.int64),
                KEY_ACTION: feat(KEY_ACTION)}
        if s.action_format == "chunk" or self.base == "state":
            if KEY_STATE not in s.feats:
                raise ValueError(f"{s.root}: the stats need {KEY_STATE}, which this dataset lacks")
            spec[KEY_STATE] = feat(KEY_STATE)
        if self.has_tactile:
            spec.setdefault(self.tac_key, feat(self.tac_key))
        cols = _read_columns(s.root, spec)

        ep, fr = cols.pop("episode_index"), cols.pop("frame_index")
        if len(ep) > 1 and not np.all((ep[1:] > ep[:-1]) | ((ep[1:] == ep[:-1]) & (fr[1:] > fr[:-1]))):
            order = np.lexsort((fr, ep))
            ep = ep[order]
            for k in cols:
                cols[k] = cols[k][order]
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]) if len(ep) else np.zeros(0, np.int64)
        ends = np.r_[starts[1:], len(ep)].astype(np.int64)
        cols["ep_last"] = np.repeat(ends - 1, ends - starts)
        return cols

    def _eef_rows(self, d: dict, k: int, lo: int, hi: int):
        """States [n, D] and chunk step k [n, D] for rows [lo, hi) of one eef root.
        Same math as __getitem__: targets past the episode end repeat its last row."""
        A = d[KEY_ACTION]
        t = np.arange(lo, hi)
        if self.base == "action":
            base_ee, base_hand = self._split_src(A[lo:hi].astype(np.float64))
            tgt = np.minimum(t + 1 + k, d["ep_last"][lo:hi])
        else:
            base_ee, base_hand = self._split_state(d[KEY_STATE][lo:hi].astype(np.float64))
            tgt = np.minimum(t + k, d["ep_last"][lo:hi])
        st, ch = eef_to_trex(base_ee, base_hand, A[tgt][:, None].astype(np.float64), self.hand_dim)
        return st, ch[:, 0]

    def _compute_stats(self) -> dict:
        """q01/q99 over every frame of every root, from the parquet tables only.

        Only the raw per-frame columns are kept in memory. The action chunk is
        never built whole as [frames, chunk, D]: each (step, dim) stat only needs
        that step over all frames, so steps are built and reduced one at a time
        on a single reused [frames, D] buffer. The result is the same."""
        D = self.action_dim
        data = [self._stats_columns(s) for s in self.sources]
        sizes = [len(d["ep_last"]) for d in data]
        n = int(sum(sizes))
        if n == 0:
            raise ValueError(f"no frames in {self.roots}")

        # Chunk length and output width per root; all roots must agree.
        n_steps = set()
        for s, d, m in zip(self.sources, data, sizes):
            if m == 0:
                continue
            if s.action_format == "chunk":
                if d[KEY_ACTION].size % (m * D) or d[KEY_STATE].size != m * D:
                    raise ValueError(f"{s.root}: action {list(d[KEY_ACTION].shape[1:])} / state "
                                     f"{list(d[KEY_STATE].shape[1:])} don't fit --action_dim {D}")
                d[KEY_ACTION] = d[KEY_ACTION].reshape(m, -1, D)
                n_steps.add(d[KEY_ACTION].shape[1])
            else:
                st, ch = self._eef_rows(d, 0, 0, 1)
                if st.shape[-1] != D or ch.shape[-1] != D:
                    raise ValueError(f"data gives {st.shape[-1]}-D states / {ch.shape[-1]}-D actions, "
                                     f"but --action_dim is {D}")
                n_steps.add(self.action_chunk)
        if len(n_steps) != 1:
            raise ValueError(f"roots give different action chunk lengths {sorted(n_steps)}")
        n_steps = n_steps.pop()

        # Action steps, one at a time; the eef states come out of the step-0 pass.
        S = np.empty((n, D), np.float32)
        C = np.empty((n, D), np.float32)
        action = {}
        for k in range(n_steps):
            pos = 0
            for s, d, m in zip(self.sources, data, sizes):
                if s.action_format == "chunk":
                    C[pos:pos + m] = d[KEY_ACTION][:, k]
                    if k == 0:
                        S[pos:pos + m] = d[KEY_STATE].reshape(m, D)
                else:
                    for lo in range(0, m, _STATS_BLOCK):
                        hi = min(lo + _STATS_BLOCK, m)
                        st, ch = self._eef_rows(d, k, lo, hi)
                        C[pos + lo:pos + hi] = ch
                        if k == 0:
                            S[pos + lo:pos + hi] = st
                pos += m
            for name, v in calculate_stats(C, [True] * D).items():
                if name != "mask":
                    action.setdefault(name, []).append(v)
            if k == 0:
                state = calculate_stats(S, [True] * D)
                del S
        del C
        action["mask"] = [True] * D
        block = {"action": action, "state": state, "num_transitions": n}   # action is per (step, dim)

        if self.has_tactile:
            T = np.empty((n, self.tac_fingers * 6), np.float32)
            pos = 0
            for s, d, m in zip(self.sources, data, sizes):
                for lo in range(0, m, _STATS_BLOCK):
                    hi = min(lo + _STATS_BLOCK, m)
                    T[pos + lo:pos + hi] = self._tactile_rows(d[self.tac_key][lo:hi], s).reshape(hi - lo, -1)
                pos += m
            del data
            # Channels with no spread (padded moments, fingers that never touch)
            # stay unnormalized instead of collapsing to -1.
            live = (np.quantile(T, 0.99, axis=0) - np.quantile(T, 0.01, axis=0)) > 1e-6
            block["tactile_f6"] = calculate_stats(T, live.tolist())
            self.accelerator.print(f"[LeRobot] tactile channels normalized: {int(live.sum())}/{T.shape[-1]}")
        self.accelerator.print(f"[LeRobot] normalization stats computed over {n} frames")
        return {STATS_KEY: block}

    def _split_src(self, rows):
        """Source rows [..., n_arms*(6+hand)] -> EE [..., n_arms*6], hand [..., n_arms*hand]."""
        w = EE_SRC_DIM + self.hand_dim
        n_arms = rows.shape[-1] // w
        ee = np.concatenate([rows[..., a * w:a * w + EE_SRC_DIM] for a in range(n_arms)], axis=-1)
        hand = np.concatenate([rows[..., a * w + EE_SRC_DIM:(a + 1) * w] for a in range(n_arms)], axis=-1)
        return ee, hand

    def _split_state(self, state):
        n_arms = self.action_dim // (9 + self.hand_dim)
        e, h = self.state_ee_index, self.state_hand_index
        return state[..., e:e + n_arms * EE_SRC_DIM], state[..., h:h + n_arms * self.hand_dim]

    def __len__(self):
        return int(self._src_cum[-1]) if len(self._src_cum) else 0

    def __getitem__(self, idx):
        si = int(np.searchsorted(self._src_cum, idx, side="right"))
        local = idx - (int(self._src_cum[si - 1]) if si > 0 else 0)
        s = self.sources[si]
        item = s.ds[local]

        # Frames become image_size PILs here, not in collate_fn: a raw decoded
        # frame is float32 at full resolution (~33 MB at 1920x1456), and a worker
        # holds a whole batch of items before collating.
        head_t = item[s.head_key]
        if head_t.dim() == 3:              # a single timestamp comes back as [3, H, W]
            head_t = head_t.unsqueeze(0)
        head = [self._img_to_pil(f) for f in head_t]             # 1+S PILs
        out = {
            "head": head,
            "head_is_pad": item.get(f"{s.head_key}_is_pad"),
            "fast": [self._img_to_pil(item[k]) if k is not None else head[0] for k in s.fast_keys],
            "task": item.get("task", ""),
        }
        if s.action_format == "chunk":
            out["action_chunk"] = np.asarray(item[KEY_ACTION], dtype=np.float32)
            out["state"] = (np.asarray(self._current(item, KEY_STATE, s), dtype=np.float32)
                            if KEY_STATE in item else None)
        else:
            rows = np.asarray(item[KEY_ACTION], dtype=np.float64)     # [(1+)chunk, src]
            if self.base == "action":
                base_ee, base_hand = self._split_src(rows[0])
                targets = rows[1:]
            else:
                state_t = np.asarray(self._current(item, KEY_STATE, s), dtype=np.float64)
                base_ee, base_hand = self._split_state(state_t)
                targets = rows
            out["state"], out["action_chunk"] = eef_to_trex(base_ee, base_hand, targets, self.hand_dim)

        # delayed tactile: one delay per sample, like the midtrain JSON path
        di = int(np.random.randint(len(self.delay_offsets)))
        out["delay_k"] = self.delay_offsets[di]
        if self._uses_tactile(s):
            W = self.vqvae_window
            f6 = torch.from_numpy(self._tactile_rows(item[self.tac_key], s))   # [W+max_d, fingers, 6]
            d = out["delay_k"]
            out["f6_hist"] = f6[d:d + W]                             # history ending at t+delay
        if self.use_tactile_deform and s.has_deform:
            out["deform"] = torch.stack([item[k][di][0] if item[k].dim() == 4 else item[k][0]
                                         for k in DEFORM_KEYS], dim=0)   # [10, H, W]
        return out

    def create_val_split(self, val_ratio=0.05, seed=42):
        """Split each root by episode; returns a val TRexLeRobotDataset (shares stats)."""
        rng = np.random.RandomState(seed)
        train_sel, val_sel = {}, {}
        for s in self.sources:
            eps = self._selected.get(s.root) or s.episode_indices()
            perm = rng.permutation(len(eps))
            n_val = max(1, int(len(eps) * val_ratio)) if len(eps) > 1 else 0
            val_sel[s.root] = sorted(eps[i] for i in perm[:n_val])
            train_sel[s.root] = sorted(eps[i] for i in perm[n_val:])
        val = copy.copy(self)
        val.sources = [copy.copy(s) for s in self.sources]
        val._build(val_sel)
        self._build(train_sel)
        self.accelerator.print(f"[LeRobot] train/val split: {self._num_episodes}/{val._num_episodes} episodes")
        return val

    # ── image helpers ──
    def _img_to_pil(self, img_t: torch.Tensor) -> PIL.Image.Image:
        """LeRobot video tensor [3,H,W] float[0,1] → RGB PIL at image_size (stretch/crop/pad).
        A PIL input was already converted by __getitem__ and is returned as is."""
        if isinstance(img_t, PIL.Image.Image):
            return img_t
        arr =(img_t.detach().float().clamp(0, 1) * 255.0).to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()           # HWC
        img = PIL.Image.fromarray(arr, mode="RGB")
        if self.image_size is None:
            return img
        W, H = self.image_size
        w, h = img.size
        if self.image_fit == "crop":
            if w * H > h * W:
                cw = round(h * W / H); x0 = (w - cw) // 2
                img = img.crop((x0, 0, x0 + cw, h))
            else:
                ch = round(w * H / W); y0 = (h - ch) // 2
                img = img.crop((0, y0, w, y0 + ch))
        elif self.image_fit == "pad":
            s = min(W / w, H / h)
            canvas = PIL.Image.new("RGB", (W, H))
            small = img.resize((round(w * s), round(h * s)), PIL.Image.LANCZOS)
            canvas.paste(small, ((W - small.size[0]) // 2, (H - small.size[1]) // 2))
            return canvas
        return img.resize(self.image_size, PIL.Image.LANCZOS)

    def collate_fn(self, batch: List[Dict]) -> Dict:
        B = len(batch)

        # ── actions + flow matching ──
        actions = np.stack([x["action_chunk"] for x in batch], axis=0)          # [B, chunk, D]
        norm_actions = torch.tensor(
            _normalize(actions, self.action_mask, self.action_min, self.action_max),
            dtype=torch.bfloat16)
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        time = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = (beta.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        # ── tactile (already at t+delay) ──
        norm_tacf6 = None
        tactile_f6_history_tensor = None
        if "f6_hist" in batch[0]:
            f6_hist = torch.stack([x["f6_hist"] for x in batch], dim=0)          # [B,W,10,6]
            if self.use_tactile_vqvae:
                tactile_f6_history_tensor = f6_hist                              # raw, model normalizes
            if self.use_tactile_vec:
                cur = f6_hist[:, -1].reshape(B, -1).numpy()                       # [B,60]
                norm_tacf6 = torch.tensor(
                    _normalize(cur, self.tacf6_mask, self.tacf6_min, self.tacf6_max).reshape(B, -1, 6),
                    dtype=torch.bfloat16)

        deforms_tensor = None
        if "deform" in batch[0]:
            deforms_tensor = torch.stack([x["deform"] for x in batch], dim=0).unsqueeze(2).float()  # [B,10,1,H,W]

        # ── robot state ──
        state_raw = None
        if self.use_robot_state:
            sl = []
            for x in batch:
                s = add_tracking_error_noise(x["state"], self.te_mean, self.te_std,
                                             self.action_dim, self.hand_dim)
                sl.append(torch.tensor(_normalize(s, self.state_mask, self.state_min, self.state_max),
                                       dtype=torch.bfloat16))
            state_raw = torch.stack(sl)

        # ── images → Qwen processor: [head (slow) | text | fast images] ──
        all_input_ids, all_pixel_values, all_grid_thw = [], [], []
        n_slow_images = 1
        for x in batch:
            pil_slow = [self._img_to_pil(x["head"][0])]
            pil_fast = [self._img_to_pil(f) for f in x["fast"]]
            content = [{"type": "image"} for _ in pil_slow]
            content.append({"type": "text", "text": x.get("task", "")})
            content += [{"type": "image"} for _ in pil_fast]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            inp = self.processor(text=text, images=pil_slow + pil_fast, return_tensors="pt", padding=False)
            all_input_ids.append(inp.input_ids[0])
            if "pixel_values" in inp and inp.pixel_values is not None:
                all_pixel_values.append(inp.pixel_values)
                all_grid_thw.append(inp.image_grid_thw)

        # ── FLARE future head frames ──
        flare_pixel_values = flare_grid_thw = None
        if self.n_flare_steps > 0:
            flare_pil = []
            for x in batch:
                head_seq, is_pad = x["head"], x.get("head_is_pad")
                for k in range(self.n_flare_steps):
                    fi = 1 + k
                    if is_pad is not None and bool(is_pad[fi]):
                        flare_pil.append(self._img_to_pil(head_seq[0]))   # fall back to current
                    else:
                        flare_pil.append(self._img_to_pil(head_seq[fi]))
            finp = self.processor.image_processor(flare_pil, return_tensors="pt")
            flare_pixel_values = finp.pixel_values.to(torch.bfloat16)
            flare_grid_thw = finp.image_grid_thw

        # ── pad input_ids (left-pad, like SftDataset) ──
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids, attn_ms = [], []
        for ids in all_input_ids:
            pad = max_len - ids.shape[0]
            padded_ids.append(F.pad(ids, (pad, 0), value=pad_id))
            a = torch.ones(max_len, dtype=torch.long)
            if pad > 0:
                a[:pad] = 0
            attn_ms.append(a)
        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attn_ms)
        pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else None
        image_grid_thw = torch.cat(all_grid_thw, dim=0) if all_grid_thw else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_slow_images": n_slow_images,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "norm_actions": norm_actions,
            "tactile_f6s": norm_tacf6,
            "tactile_deforms": deforms_tensor,
            "tactile_f6s_delayed": norm_tacf6,            # sampled at t + delay_k
            "tactile_deforms_delayed": deforms_tensor,
            "tactile_codes": None,                        # embedded VQ-VAE encodes the history
            "tactile_f6_history": tactile_f6_history_tensor,
            "delay_k": torch.tensor([x["delay_k"] for x in batch], dtype=torch.long),
            "time_r": time_r,
            "eps_r": eps_r,
            "state_raw": state_raw,
            "flare_pixel_values": flare_pixel_values,
            "flare_grid_thw": flare_grid_thw,
        }
