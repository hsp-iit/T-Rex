"""Test D — XHand tactile VQ-VAE: shapes, real-window reconstruction, codebook use,
and parity between the standalone VQ-VAE and the copy embedded in the VLA."""
import json
import os

import numpy as np
import torch

from tactile_vqvae.data.taptic_windows import XHandWindowDataset, taptic_split
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig


def test_untrained_smoke_shapes():
    cfg = TactileVQVAEConfig(window=16, in_channels=15, n_fingers=5, per_finger_dim=3, codebook_size=64)
    m = TactileVQVAE(cfg)
    x = torch.randn(8, 16, 5, 3)
    out = m(x, torch.ones(8))
    assert out["recon"].shape == (8, 16, 5, 3) and out["indices"].shape == (8,)
    assert torch.isfinite(out["total_loss"])
    out["total_loss"].backward()
    assert m.encode(x).shape == (8,) and m.decode_indices(m.encode(x)).shape == (8, 16, 5, 3)


def test_trained_ckpt_reconstructs_real_windows(vq_ckpt, tdir):
    blob = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    cfg = TactileVQVAEConfig.from_dict(blob["config"])
    assert (cfg.window, cfg.in_channels, cfg.n_fingers, cfg.per_finger_dim, cfg.granularity) == (16, 15, 5, 3, "hand")
    m = TactileVQVAE(cfg); m.load_state_dict(blob["model_state"]); m.eval()
    from tactile_vqvae.data.stats import TacF6Stats
    stats = TacF6Stats.from_dict(blob["stats"])
    norm = json.load(open(os.path.join(tdir, "norm_stats.json")))
    assert np.allclose(stats.tacf6_min, norm[next(iter(norm))]["tactile_f6"]["q01"])     # same train-only stats
    ds = XHandWindowDataset(tdir, taptic_split(tdir)["val"], stats, 16, 8)
    x = torch.stack([ds[i]["f6"] for i in range(len(ds))])
    with torch.no_grad():
        idx = m.encode(x)
        r = m.decode_indices(idx)
    mse = float(((r - x) ** 2).mean())
    base = float(((x - x.mean((0, 1), keepdim=True)) ** 2).mean())
    assert mse < 0.5 * base, (mse, base)
    h = np.bincount(idx.numpy(), minlength=cfg.codebook_size)
    assert (h > 0).sum() >= 8                    # no collapse
    assert h.max() / h.sum() < 0.6               # largest code = free-space windows


def test_embedded_vqvae_matches_standalone(vq_ckpt, tiny_qwen, tdir):
    from transformers import AutoConfig
    from qwen_vla.modeling_vla import Qwen3VLVLAModel
    blob = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    tcfg = AutoConfig.from_pretrained(tiny_qwen).text_config
    vla = Qwen3VLVLAModel(tcfg, action_dim=21, action_chunk=16, tacf6_dim=3, use_tactile_vqvae=True,
                          vqvae_config=blob["config"], n_tactile_fingers=5)
    vla.load_tactile_vqvae(blob)
    assert vla.tacf6_vqvae_min.shape == (15,)
    from tactile_vqvae.data.stats import TacF6Stats
    stats = TacF6Stats.from_dict(blob["stats"])
    ds = XHandWindowDataset(tdir, taptic_split(tdir)["val"][:2], stats, 16, 16)
    raw = torch.from_numpy(np.stack([ds.raw_window(i) for i in range(len(ds))]))      # [B,16,5,3] raw
    codes = vla.encode_tactile_f6_history(raw)
    ref = vla.tactile_vqvae.encode(torch.stack([ds[i]["f6"] for i in range(len(ds))]))
    assert codes.shape == (len(ds), 1)                                    # hand granularity: 1 token
    assert torch.equal(codes[:, 0], ref)
    tok = vla._embed_tactile_observations(torch.zeros(len(ds), 5, 3), None, "cpu", torch.float32,
                                          tactile_f6_history=raw)
    assert tok.shape == (len(ds), 1 + 5, tcfg.hidden_size)               # 1 code token + 5 fingertip tokens
    try:
        vla.encode_tactile_f6_history(torch.zeros(2, 16, 10, 6))
        raise AssertionError("old [B,T,10,6] layout must be rejected")
    except ValueError:
        pass
