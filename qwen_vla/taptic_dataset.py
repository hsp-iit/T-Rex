"""Isolated TAPTIC adapter for the unchanged cascaded T-Rex loop.

Full taxels enter the existing vector embedder as five 360-D tokens. No future
observations, VQ codes, deformation images, contact labels or risk labels.
The historical `tactile_f6s` key now carries taxels; it does not imply six axes.
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
from taptic.data import (Source, CAMERAS, finite, normalize, denormalize,
                         tactile_at, tactile_stream_between)
from taptic.video import VideoReader


def validate_config(cfg):
    if (cfg.action_dim, cfg.action_chunk) != (21, 16):
        raise ValueError('TAPTIC requires action_dim=21, action_chunk=16')
    for k in ('use_tactile_deform', 'use_tactile_vqvae', 'use_tactile_code'):
        if getattr(cfg, k, 0):
            raise ValueError(f'TAPTIC raw taxels require {k}=0')
    if not getattr(cfg, 'use_tactile_vec', 0):
        raise ValueError('TAPTIC requires use_tactile_vec=1')
    if tuple(getattr(cfg, 'tactile_delay_offsets', (0,))) != (0,):
        raise ValueError('TAPTIC causal baseline requires tactile_delay_offsets=[0]')
    if (getattr(cfg, 'cascaded_total_steps', 10), getattr(cfg, 'cascaded_split_step', 6)) != (10, 6):
        raise ValueError('TAPTIC baseline requires fixed tau_split=0.4 (6/10 steps)')


class TRexTapticDataset(torch.utils.data.Dataset):
    def __init__(self, config, processor, accelerator, split='train', episode_ids=None, _stats=None):
        validate_config(config)
        self.config, self.processor, self.accelerator = config, processor, accelerator
        self.source = Source(config.lerobot_root)
        self.stats_data = _stats if _stats is not None else json.loads(Path(config.taptic_stats).read_text())
        block = self.stats_data['taptic']
        if block['source_identity'] != self.source.identity:
            raise ValueError('statistics belong to a different dataset')
        ids = block['train_episodes']
        if len(ids) != len(set(ids)) or any(self.source.episodes.loc[e, 'suggested_split'] != 'train' for e in ids):
            raise ValueError('statistics contain non-training episodes')
        st = block['action']
        lo, hi = np.asarray(st['min']), np.asarray(st['max'])
        if lo.shape != (21,) or hi.shape != (21,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError('invalid action normalization statistics')
        self.action_stats = st
        self.scope = self.source.episodes if episode_ids is None else self.source.episodes.loc[episode_ids]
        training = sorted(map(int, self.scope.index[self.scope.suggested_split == 'train']))
        if sorted(ids) != training:
            raise ValueError('statistics training episodes do not match dataset scope')
        self.image_size = tuple(config.image_size) if getattr(config, 'image_size', None) else None
        self.use_flare = bool(getattr(config, 'use_flare', 0))
        self.n_flare_steps = int(getattr(config, 'n_flare_steps', 0)) if self.use_flare else 0
        self.flare_frame_stride = int(getattr(config, 'flare_frame_stride', 1))
        if self.flare_frame_stride < 1:
            raise ValueError('FLARE stride must be positive')
        self._cache_key = None
        self._video = VideoReader()
        self._set_split(split)

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
            self._cache = (d, s, a, times, sm, available)
            self._cache_key = ep
        return self._cache

    def _img(self, ep, cam, t):
        row = self.source.episodes.loc[ep]
        path = self.source.info['video_path'].format(video_key=cam,
            chunk_index=int(row[f'videos/{cam}/chunk_index']), file_index=int(row[f'videos/{cam}/file_index']))
        ts = float(row[f'videos/{cam}/from_timestamp']) + t / 10
        return self._video.frame(str(self.source.root / path), ts, 10)

    def __getitem__(self, idx):
        ep, t = self._rows[idx]
        d, s, a, times, sm, available = self._episode(ep)
        native = not bool(self.source.episodes.loc[ep, 'tactile_interpolated'])
        taxels, force, valid, force_valid, sample_ns = tactile_at(sm, available, int(times[t]), int(times[0]), native)
        stream = tactile_stream_between(sm, available, int(times[t]),
                                        int(times[t-1]) if t else None, native)
        # Missing taxels use neutral zeros and a separate explicit validity mask.
        scaled = taxels / np.asarray([128., 128., 255.], np.float32)
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
        return {'head': self._img(ep, CAMERAS[0], t), 'fast': [self._img(ep, CAMERAS[1], t)],
                'language': self.source.episodes.loc[ep, 'caption'], 'action_chunk': chunk_21d(s, a, t),
                'tactile_f6': scaled.reshape(5, 360), 'tactile_taxels': taxels,
                'tactile_valid': np.full(5, valid), 'sdk_force': force, 'sdk_force_valid': force_valid,
                'tactile_stream': stream,
                'state_raw': state, 'recorded_haptics': haptics, 'flare_frames': flare,
                'meta': {'episode': ep, 'frame': t, 'observation_ns': int(times[t]),
                         'tactile_available_ns': sample_ns, 'clock_is_nominal': not native,
                         'action_timestamps_ns': times[t:t+16].copy()}, 'delay_k': 0}

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
        time_r = (d.sample((B,)) * 0.999 + 0.001).to(torch.bfloat16)
        eps_r = torch.randn_like(norm_actions)

        norm_tac = torch.from_numpy(np.stack([x["tactile_f6"] for x in batch]))
        norm_tac_d = norm_tac  # same inference tick; no future tactile leakage
        hist = None

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
            "recorded_haptics": {k: torch.from_numpy(np.stack([x["recorded_haptics"][k] for x in batch]))
                                 for k in batch[0]["recorded_haptics"]},
            "contact_features": None, "contact_risk": None,
        }
