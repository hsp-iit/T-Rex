"""Evaluate an XHand tactile VQ-VAE on the held-out TAPTIC episodes.

    python -m tactile_vqvae.eval_taptic --checkpoint <run>/latest.pt \
        --taptic_dir <out>/<task> --out_dir <run>/eval

Reports (JSON + PNGs):
  * reconstruction MSE (normalized units) overall, per finger, per axis;
  * contact vs free-space windows (contact = any |F| > --contact_counts);
  * global code histogram over ALL val windows: active codes, perplexity,
    max code share (collapse check), train vs val usage;
  * temporal alignment: per-timestep MSE inside the window and the lag that
    maximises the recon/original cross-correlation (expected 0);
  * plots of original vs reconstructed forces for all five fingers.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from tactile_vqvae.data.taptic_windows import XHandWindowDataset, taptic_split
from tactile_vqvae.data.stats import TacF6Stats
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig
from taptic.constants import FINGERS


def load_vqvae(path: str, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TactileVQVAE(TactileVQVAEConfig.from_dict(blob["config"]))
    model.load_state_dict(blob["model_state"])
    return model.to(device).eval(), TacF6Stats.from_dict(blob["stats"]), blob


@torch.no_grad()
def run(model, ds, device, bs=1024):
    X, R, I, M = [], [], [], []
    for s in range(0, len(ds), bs):
        items = [ds[i] for i in range(s, min(len(ds), s + bs))]
        x = torch.stack([b["f6"] for b in items]).to(device)
        idx = model.encode(x)
        X.append(x.cpu().numpy()); R.append(model.decode_indices(idx).cpu().numpy())
        I.append(idx.cpu().numpy()); M.append(np.stack([ds.raw_window(i) for i in range(s, s + len(items))]))
    return np.concatenate(X), np.concatenate(R), np.concatenate(I), np.concatenate(M)


def code_stats(idx, K):
    h = np.bincount(idx.reshape(-1), minlength=K).astype(np.float64)
    p = h / h.sum()
    nz = p[p > 0]
    return {"active_codes": int((h > 0).sum()), "codebook_size": K,
            "perplexity": float(np.exp(-(nz * np.log(nz)).sum())),
            "max_code_share": float(p.max()), "top5_share": float(np.sort(p)[-5:].sum()),
            "histogram": h.astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--taptic_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--contact_counts", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, blob = load_vqvae(a.checkpoint, dev)
    cfg = model.cfg
    sp = taptic_split(a.taptic_dir)
    val = XHandWindowDataset(a.taptic_dir, sp["val"], stats, cfg.window, a.stride)
    tr = XHandWindowDataset(a.taptic_dir, sp["train"], stats, cfg.window, a.stride * 4)
    X, R, I, RAW = run(model, val, dev)
    _, _, I_tr, _ = run(model, tr, dev)
    err = (R - X) ** 2                                             # [N,T,5,3]
    contact = (np.abs(RAW).max(axis=(1, 2, 3)) > a.contact_counts)
    per_t = err.mean(axis=(0, 2, 3))
    # temporal alignment: cross-correlation lag on the most active channel
    ch = np.unravel_index(np.argmax(np.abs(RAW).mean(axis=(0, 1))), RAW.shape[2:])
    xs = X[:, :, ch[0], ch[1]] - X[:, :, ch[0], ch[1]].mean(1, keepdims=True)
    rs = R[:, :, ch[0], ch[1]] - R[:, :, ch[0], ch[1]].mean(1, keepdims=True)
    lags = list(range(-4, 5))
    xc = [float(np.mean([np.dot(np.roll(r, L), x) for r, x in zip(rs[contact], xs[contact])])) for L in lags]
    res = {
        "checkpoint": a.checkpoint, "config": cfg.to_dict(), "val_windows": int(len(X)),
        "val_episodes": sp["val"], "contact_threshold_counts": a.contact_counts,
        "recon_mse": float(err.mean()),
        "recon_mse_contact": float(err[contact].mean()) if contact.any() else None,
        "recon_mse_free": float(err[~contact].mean()) if (~contact).any() else None,
        "frac_contact_windows": float(contact.mean()),
        "recon_mse_per_finger": {f: float(err[:, :, i].mean()) for i, f in enumerate(FINGERS)},
        "recon_mse_per_finger_contact": {f: float(err[contact][:, :, i].mean()) for i, f in enumerate(FINGERS)},
        "recon_mse_per_axis": {ax: float(err[..., j].mean()) for j, ax in enumerate("xyz")},
        "recon_mse_per_timestep": per_t.tolist(),
        # trivial baseline: one constant (the val mean) for every window
        "baseline_mse_global_mean": float(((X - X.mean(axis=(0, 1), keepdims=True)) ** 2).mean()),
        "baseline_mse_global_mean_contact": float(((X[contact] - X.mean(axis=(0, 1), keepdims=True)) ** 2).mean()),
        # oracle: each window's own temporal mean (= within-window temporal detail only)
        "oracle_mse_window_mean": float(((X - X.mean(1, keepdims=True)) ** 2).mean()),
        "codes_val": code_stats(I, cfg.codebook_size),
        "codes_train": {k: v for k, v in code_stats(I_tr, cfg.codebook_size).items() if k != "histogram"},
        "codes_contact_windows_val": int(len(np.unique(I[contact]))),
        "codes_free_windows_val": int(len(np.unique(I[~contact]))),
        "alignment": {"channel": f"{FINGERS[ch[0]]}.{'xyz'[ch[1]]}", "lags": lags, "xcorr": xc,
                      "best_lag": lags[int(np.argmax(xc))]},
    }
    with open(os.path.join(a.out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # one val episode, non-overlapping windows -> continuous trace (denormalized counts)
    ep = sp["val"][int(np.argmax([np.abs(RAW[val.index[:, 2] == e]).sum() for e in sp["val"]]))]
    sel = np.flatnonzero(val.index[:, 2] == ep)[::max(1, cfg.window // a.stride)]
    xo = stats.denormalize(X[sel]).reshape(-1, 5, 3)
    xr = stats.denormalize(R[sel]).reshape(-1, 5, 3)
    t = np.arange(len(xo)) / 76.4
    fig, axs = plt.subplots(5, 3, figsize=(15, 12), sharex=True)
    for i, fname in enumerate(FINGERS):
        for j, ax in enumerate("xyz"):
            axs[i, j].plot(t, xo[:, i, j], lw=1, label="original")
            axs[i, j].plot(t, xr[:, i, j], lw=1, label="VQ-VAE recon")
            axs[i, j].set_title(f"{fname} F{ax}", fontsize=9)
    axs[0, 0].legend(fontsize=8)
    for j in range(3):
        axs[-1, j].set_xlabel("s (concatenated 16-sample windows)")
    fig.suptitle(f"episode {ep} (val), raw counts, normalization clipped to [q01,q99]")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "recon_all_fingers.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].bar(range(cfg.codebook_size), res["codes_val"]["histogram"]); ax[0].set_title("val code histogram")
    ax[1].plot(per_t, marker="o"); ax[1].set_title("recon MSE per window timestep"); ax[1].set_xlabel("sample (15 = newest)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "codes_and_timesteps.png"), dpi=110); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k not in ("codes_val", "config", "recon_mse_per_timestep")}, indent=1))
    print("codes_val:", {k: v for k, v in res["codes_val"].items() if k != "histogram"})


if __name__ == "__main__":
    main()
