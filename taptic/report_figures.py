"""Figures + facts for docs/xhand (action reconstruction, chunk accounting, timing, FLARE).

    python -m taptic.report_figures --taptic_dir <out>/<task> --out docs/xhand/figures
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .actions import reconstruct_chunk, rotation_angle_error
from .constants import STATE_EE_POS, STATE_EE_ROTVEC, XHAND_JOINTS

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def _style(ax):
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taptic_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flare_steps", type=int, default=8)
    ap.add_argument("--flare_stride", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_parquet(os.path.join(a.taptic_dir, "samples.parquet"))
    seg = pd.read_parquet(os.path.join(a.taptic_dir, "segments.parquet"))
    sp = json.load(open(os.path.join(a.taptic_dir, "split.json")))
    facts = {}

    # ── 1. reconstruction error over ALL samples + one val trajectory ──
    pe, re, flare_clamped, n_eps = [], [], 0, {}
    for ep, d in s.groupby("episode"):
        z = np.load(os.path.join(a.taptic_dir, "episodes", f"episode_{ep:06d}.npz"))
        S, A, C = z["state"], z["action"], z["chunks21"]
        n = len(S)
        n_eps[ep] = n
        for i, t in enumerate(d.frame.to_numpy()):
            rec = reconstruct_chunk(C[i], S[t, STATE_EE_POS].astype(np.float64), S[t, STATE_EE_ROTVEC])
            pe.append(np.linalg.norm(rec[:, :3] - A[t:t + 16, :3], axis=1).max())
            re.append(rotation_angle_error(rec[:, 3:6], A[t:t + 16, 3:6]).max())
            flare_clamped += int(t + a.flare_steps * a.flare_stride > n - 1)
    facts["reconstruction"] = {"pos_err_max_m": float(np.max(pe)), "pos_err_p99_m": float(np.percentile(pe, 99)),
                               "rot_err_max_rad": float(np.max(re)), "rot_err_p99_rad": float(np.percentile(re, 99)),
                               "n_chunks": len(pe)}
    facts["flare_future_clamped_frac"] = flare_clamped / len(s)

    ep = sp["val_merged"][int(np.argmax([n_eps.get(e, 0) for e in sp["val_merged"]]))]
    z = np.load(os.path.join(a.taptic_dir, "episodes", f"episode_{ep:06d}.npz"))
    S, A, C, anc = z["state"], z["action"], z["chunks21"], z["chunk_anchor"]
    rec_t, rec = [], []
    last = -16
    for i, t in enumerate(anc):           # non-overlapping chunks, each anchored on the measured pose
        if t >= last + 16:
            rec_t.append(np.arange(t, t + 16))
            rec.append(reconstruct_chunk(C[i], S[t, STATE_EE_POS].astype(np.float64), S[t, STATE_EE_ROTVEC]))
            last = t
    rec_t, rec = np.concatenate(rec_t), np.concatenate(rec)
    T = np.arange(len(A)) / 10
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for j, lab in enumerate("xyz"):
        axs[0, 0].plot(T, A[:, j], color=[C1, C2, C3][j], lw=1.5, label=f"recorded ee.{lab}")
        axs[0, 0].plot(rec_t / 10, rec[:, j], "o", ms=2.5, color=[C1, C2, C3][j], mfc="white", label=f"from 21-D chunks ({lab})")
        axs[1, 0].plot(T, A[:, 3 + j], color=[C1, C2, C3][j], lw=1.5, label=f"recorded ee.w{lab}")
        axs[1, 0].plot(rec_t / 10, rec[:, 3 + j], "o", ms=2.5, color=[C1, C2, C3][j], mfc="white")
    for j, k in enumerate([0, 3, 6]):
        axs[0, 1].plot(T, A[:, 6 + k], color=[C1, C2, C3][j], lw=1.5, label=XHAND_JOINTS[k].replace("right_hand_", ""))
        axs[0, 1].plot(rec_t / 10, rec[:, 9 + k], "o", ms=2.5, color=[C1, C2, C3][j], mfc="white")
    pos_e = np.linalg.norm(rec[:, :3] - A[rec_t, :3], axis=1) * 1e3
    rot_e = np.degrees(rotation_angle_error(rec[:, 3:6], A[rec_t, 3:6]))
    axs[1, 1].plot(rec_t / 10, pos_e * 1e3, color=C1, lw=1, label="position error (µm)")
    axs[1, 1].plot(rec_t / 10, rot_e * 1e3, color=C2, lw=1, label="rotation error (mdeg)")
    dp = C[:, :, :3]
    axs[2, 0].plot(anc / 10, np.linalg.norm(dp[:, -1], axis=1) * 100, color=C1, lw=1)
    axs[2, 0].set_ylabel("|dp_15| (cm)", color=MUTED, fontsize=8)
    axs[2, 1].hist(np.array(pe) * 1e9, bins=50, color=C1)
    axs[2, 1].set_xlabel("max position error per chunk, all samples (nm)", color=MUTED, fontsize=8)
    titles = ["commanded EE position (m)", "XHand targets (rad), 3 of 12",
              "commanded rotvec, gripper_down frame (rad)", "reconstruction error, this episode",
              "chunk displacement at k=15", "all 21,821 chunks"]
    for ax, ti in zip(axs.flat, titles):
        _style(ax)
        ax.set_title(ti, fontsize=9, color=INK, loc="left")
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]):
        ax.legend(fontsize=7, frameon=False, ncol=2)
    axs[2, 0].set_xlabel("time (s, frame/10)", color=MUTED, fontsize=8)
    fig.suptitle(f"18-D -> 21-D delta-base -> reconstructed absolute commands (val episode {ep}); "
                 "lines = recorded, circles = reconstructed", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "action_reconstruction.png"), dpi=110); plt.close(fig)

    # ── 2. chunk accounting per primitive ──
    g = seg.groupby("label").agg(frames=("length", "sum"), kept=("n_kept", "sum"),
                                 cross=("n_anchor_cross_allowed", "sum")).sort_values("frames")
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(g))
    ax.barh(y + 0.2, g.frames, height=0.38, color="#c9c8c2", label="annotated frames")
    ax.barh(y - 0.2, g.kept, height=0.38, color=C1, label="kept samples (chunk inside primitive)")
    for yi, (f, k) in enumerate(zip(g.frames, g.kept)):
        ax.text(k + 60, yi - 0.2, f"{k} ({100 * k / f:.0f}%)", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, g.index, fontsize=8)
    _style(ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("samples per primitive under strict 16-step boundary rule", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "chunks_per_primitive.png"), dpi=110); plt.close(fig)
    facts["per_primitive"] = g.astype(int).to_dict(orient="index")

    with open(os.path.join(a.out, "facts.json"), "w") as f:
        json.dump(facts, f, indent=1)
    print(json.dumps({k: v for k, v in facts.items() if k != "per_primitive"}, indent=1))


if __name__ == "__main__":
    main()
