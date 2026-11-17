"""TRexTapticDataset — primitive-level Panda + XHand midtraining data for T-REX.

Drop-in for ``MidtrainTacFlareDataset`` (scripts/midtrain.py): same
``create_val_split`` / ``collate_fn`` / ``stats_data`` / ``_cum_frames`` /
``_num_episodes`` interface, and ``collate_fn`` returns the same keys, so the
cascaded training loop, FLARE loss and EpisodeGroupedSampler are unchanged.

Differences from the official loader (all dimensional / data-source):
  * reads the derived set written by ``python -m taptic.convert`` (one or more
    task directories); one item = one primitive-annotated sample, never a
    chunk crossing a primitive boundary; instruction = primitive instruction;
  * actions: [16, 21] delta-base chunks (precomputed, verified by tests);
  * tactile: XHand fingertip forces [5, 3] (current / delayed) and a native-rate
    history window [16, 5, 3] (raw; the embedded VQ-VAE normalizes it);
  * cameras: slow = ``observation.images.left`` (overhead scene, also the FLARE
    target stream), fast = ``observation.images.wrist`` only.  The official
    pipeline has two fast wrist views; the second one does not exist on this
    robot and is NOT synthesized;
  * no robot state input (``state_raw`` is None), no deformation maps.

Delayed tactile (official Paradigm C): k ~ U(tactile_delay_offsets) frames; the
tactile expert sees the observation available at the real observation time of
frame t+k (latest native sample <= observation_ns[t+k]) and the 16-sample
history ending there.  t+k <= t+12 stays inside the sample's primitive.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn.functional as F

from taptic.constants import ACTION_DIM, N_FINGERS, TAC_DIM, VQVAE_WINDOW
from taptic.source import TapticSource
from taptic.tactile import EpisodeTactile, TactileNorm
from taptic.video import VideoReader

SLOW_CAM = "observation.images.left"
FAST_CAMS = ("observation.images.wrist",)


def _normalize(values, mask, vmin, vmax):          # == midtrain.MidtrainTacFlareDataset._normalize
    return np.where(mask, np.clip(2 * (values - vmin) / (vmax - vmin + 1e-8) - 1, -1, 1), values)


class TRexTapticDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, split: str = "train"):
        self.config = config
        self.processor = processor
        self.accelerator = accelerator
        self.split = split
        dirs = list(getattr(config, "taptic_dirs", None) or [])
        if not dirs:
            raise ValueError("--taptic_dirs is required for --data_format taptic")
        self.dirs = [os.path.expanduser(d) for d in dirs]

        # ── normalization (train-split stats from taptic.convert / taptic.stats) ──
        stats_path = getattr(config, "taptic_stats", "") or ""
        if not stats_path:
            if len(self.dirs) > 1:
                raise ValueError("several --taptic_dirs need a joint --taptic_stats "
                                 "(python -m taptic.stats --dirs ... --out stats.json)")
            stats_path = os.path.join(self.dirs[0], "norm_stats.json")
        with open(stats_path) as f:
            blob = json.load(f)
        self.stats_key = next(iter(blob))
        block = blob[self.stats_key]
        self.action_min = np.asarray(block["action"]["q01"], np.float32)
        self.action_max = np.asarray(block["action"]["q99"], np.float32)
        self.action_mask = np.asarray(block["action"]["mask"], bool)
        if self.action_min.shape != (config.action_chunk, config.action_dim):
            raise ValueError(f"action stats {self.action_min.shape} != "
                             f"({config.action_chunk}, {config.action_dim})")
        self.tnorm = TactileNorm.from_dict(block["tactile_f6"])
        self.tacf6_min, self.tacf6_max, self.tacf6_mask = self.tnorm.q01, self.tnorm.q99, self.tnorm.mask
        self.stats_data = {self.stats_key: {
            "action": {"mask": self.action_mask.tolist(), "q01": self.action_min.tolist(),
                       "q99": self.action_max.tolist()},
            "tactile_f6": self.tnorm.to_dict(),
            "layout": {"action": "[dp_xyz(3) in chunk-start flange frame, rot6d(6), xhand_q(12) abs rad]",
                       "tactile_f6": "[5 fingertips thumb..little, (Fx,Fy,Fz) XHand calc_force raw counts]",
                       "state": "not used"},
        }}

        # ── samples (sorted by dir, episode, frame -> contiguous per episode) ──
        frames = []
        self._srcs = []
        for di, d in enumerate(self.dirs):
            with open(os.path.join(d, "manifest.json")) as f:
                man = json.load(f)
            self._srcs.append(TapticSource(man["source_root"], man["task"], man.get("source_revision")))
            s = pd.read_parquet(os.path.join(d, "samples.parquet"))
            s["dir_idx"] = di
            frames.append(s)
        self._all = pd.concat(frames, ignore_index=True).sort_values(
            ["dir_idx", "episode", "frame"]).reset_index(drop=True)
        self._set_rows(self._all[self._all.split == split])

        self.image_size = tuple(config.image_size) if getattr(config, "image_size", None) else None
        self.use_flare = bool(getattr(config, "use_flare", 0))
        self.n_flare_steps = int(getattr(config, "n_flare_steps", 0)) if self.use_flare else 0
        self.flare_frame_stride = int(getattr(config, "flare_frame_stride", 4))
        self.delay_offsets = tuple(getattr(config, "tactile_delay_offsets", (0, 4, 8, 12)))
        self.vqvae_window = int(getattr(config, "vqvae_window", VQVAE_WINDOW))
        if max(self.delay_offsets) > config.action_chunk - 1:
            raise ValueError("tactile delay beyond the action chunk would leave the primitive")
        self._video = VideoReader()
        self._cache_key = None
        accelerator.print(
            f"[Taptic] {split}: {self._num_episodes} episodes, {len(self)} samples, "
            f"dirs={[os.path.basename(d) for d in self.dirs]}, flare={self.n_flare_steps}x{self.flare_frame_stride}")

    def _set_rows(self, rows: pd.DataFrame):
        self._rows = rows.reset_index(drop=True)
        counts = self._rows.groupby(["dir_idx", "episode"], sort=False).size().to_numpy()
        self._cum_frames = np.cumsum(counts).astype(np.int64)
        self._num_episodes = int(len(counts))

    def __len__(self):
        return len(self._rows)

    def create_val_split(self, val_ratio=None, seed=None):
        """Uses the converter's deterministic episode-level split (val_ratio/seed
        are ignored: the split is fixed at conversion time to avoid leakage)."""
        val = copy.copy(self)
        val.split = "val"
        val._set_rows(self._all[self._all.split == "val"])
        val._video = VideoReader()
        val._cache_key = None
        self.accelerator.print(f"[Taptic] val split from split.json: {val._num_episodes} eps, {len(val)} samples "
                               f"(train {self._num_episodes} eps, {len(self)} samples)")
        return val

    # ── per-worker episode cache ──
    def _episode(self, di: int, ep: int):
        key = (di, ep)
        if self._cache_key != key:
            z = np.load(os.path.join(self.dirs[di], "episodes", f"episode_{ep:06d}.npz"))
            self._cache = {k: z[k] for k in z.files}
            self._cache_tac = EpisodeTactile(
                t_ns=self._cache["tac_t_ns"], force=self._cache["tac_force"].astype(np.float32),
                reset=self._cache["tac_reset"], start_idx=int(self._cache["tac_start_idx"]))
            src = self._srcs[di]
            self._cache_vid = {cam: src.video_ref(ep, cam) for cam in (SLOW_CAM,) + FAST_CAMS}
            self._cache_fps = src.fps
            self._cache_key = key
        return self._cache, self._cache_tac

    def _img(self, cam: str, frame: int) -> np.ndarray:
        path, t0, _ = self._cache_vid[cam]
        return self._video.frame(path, t0 + frame / self._cache_fps, self._cache_fps)

    def __getitem__(self, idx: int) -> Dict:
        r = self._rows.iloc[idx]
        ep, t = int(r.episode), int(r.frame)
        z, tac = self._episode(int(r.dir_idx), ep)
        n = len(z["obs_ns"])
        chunk = z["chunks21"][int(r.chunk_row)]
        assert int(z["chunk_anchor"][int(r.chunk_row)]) == t

        delay_k = int(np.random.default_rng().choice(self.delay_offsets))
        td = t + delay_k                                   # <= t+12 < seg_end (inside primitive)
        cur_i = tac.latest_index(int(z["obs_ns"][t]))
        del_i = tac.latest_index(int(z["obs_ns"][td]))
        hist = tac.force[tac.window_indices(del_i, self.vqvae_window)]       # [W,5,3] raw

        head = self._img(SLOW_CAM, t)
        fast = [self._img(c, t) for c in FAST_CAMS]
        flare, flare_clamped = [], 0
        for k in range(self.n_flare_steps):
            ft = t + (k + 1) * self.flare_frame_stride
            flare_clamped += int(ft > n - 1)
            flare.append(self._img(SLOW_CAM, min(ft, n - 1)))   # official: clamp to episode end
        return {
            "head": head, "fast": fast,
            "action_chunk": chunk,
            "tactile_f6": tac.force[cur_i],
            "tactile_f6_delayed": tac.force[del_i],
            "tactile_f6_history": hist,
            "delay_k": delay_k,
            "language": r.instruction,
            "flare_frames": flare,
            "meta": {"dir": int(r.dir_idx), "episode": ep, "frame": t, "primitive": r.primitive,
                     "sample_idx": int(r.sample_idx), "flare_clamped": flare_clamped,
                     "tactile_age_ms": float((z["obs_ns"][t] - tac.t_ns[cur_i]) / 1e6)},
        }

    def _pil(self, arr):
        img = PIL.Image.fromarray(arr)
        return img.resize(self.image_size, PIL.Image.LANCZOS) if self.image_size is not None else img

    def collate_fn(self, batch: List[Dict]) -> Dict:
        cfg = self.config
        B = len(batch)
        actions = np.stack([x["action_chunk"] for x in batch]).astype(np.float32)        # [B,16,21]
        norm_actions = torch.tensor(_normalize(actions, self.action_mask, self.action_min, self.action_max),
                                    dtype=torch.bfloat16)
        # ── flow matching (identical to MidtrainTacFlareDataset.collate_fn) ──
        d = torch.distributions.Beta(torch.tensor(1.5, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))
        time = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        t_ = time[:, None, None]
        noise = torch.randn_like(norm_actions)
        x_t = t_ * noise + (1 - t_) * norm_actions
        u_t = noise - norm_actions
        time_r = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        norm_tac = norm_tac_d = None
        if cfg.use_tactile_vec:
            norm_tac = torch.tensor(self.tnorm.normalize(np.stack([x["tactile_f6"] for x in batch])),
                                    dtype=torch.bfloat16)                                 # [B,5,3]
            norm_tac_d = torch.tensor(self.tnorm.normalize(np.stack([x["tactile_f6_delayed"] for x in batch])),
                                      dtype=torch.bfloat16)
        hist = None
        if getattr(cfg, "use_tactile_vqvae", 0):
            hist = torch.from_numpy(np.stack([x["tactile_f6_history"] for x in batch]).astype(np.float32))  # [B,16,5,3] raw

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
        return {
            "input_ids": torch.stack(ids), "attention_mask": torch.stack(attn),
            "pixel_values": torch.cat(all_pv), "image_grid_thw": torch.cat(all_thw),
            "n_slow_images": 1,
            "noisy_actions": x_t, "target": u_t, "timesteps": time, "norm_actions": norm_actions,
            "tactile_f6s": norm_tac, "tactile_deforms": None,
            "tactile_f6s_delayed": norm_tac_d, "tactile_deforms_delayed": None,
            "tactile_codes": None, "tactile_f6_history": hist,
            "delay_k": torch.tensor([x["delay_k"] for x in batch], dtype=torch.long),
            "time_r": time_r, "eps_r": eps_r, "state_raw": None,
            "flare_pixel_values": flare_pv, "flare_grid_thw": flare_thw,
            "meta": [x["meta"] for x in batch],
        }
