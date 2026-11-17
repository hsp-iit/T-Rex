"""Joint TRAIN-split normalization stats for several converted TAPTIC tasks.

    python -m taptic.stats --dirs <out>/taskA <out>/taskB --out joint_stats.json

Recomputes from the per-episode arrays exactly like taptic.convert does for
one task (per-(step,dim) action q01/q99 over train chunks; per-channel
tactile q01/q99 over the native samples inside train episodes).  Use the
result with ``midtrain.py --taptic_stats`` and for VQ-VAE training on the mix.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .constants import ACTION_DIM, TAC_FLAT
from .tactile import TactileNorm


def joint_stats(dirs):
    chunks, tac = [], []
    for d in dirs:
        sp = json.load(open(os.path.join(d, "split.json")))
        for ep in sp["train_merged"]:
            z = np.load(os.path.join(d, "episodes", f"episode_{ep:06d}.npz"))
            if len(z["chunks21"]):
                chunks.append(z["chunks21"])
            last = int(np.searchsorted(z["tac_t_ns"], int(z["obs_ns"][-1]), side="right"))
            tac.append(z["tac_force"][int(z["tac_start_idx"]):last].reshape(-1, TAC_FLAT))
    A = np.concatenate(chunks)
    tn = TactileNorm.fit(np.concatenate(tac))
    return {"taptic_joint": {
        "action": {"mask": [True] * ACTION_DIM, "q01": np.quantile(A, 0.01, axis=0).tolist(),
                   "q99": np.quantile(A, 0.99, axis=0).tolist()},
        "tactile_f6": {**tn.to_dict(), "units": "XHand SDK calc_force raw counts"},
        "num_train_chunks": int(len(A)), "dirs": [os.path.abspath(d) for d in dirs]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = joint_stats(a.dirs)
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
    print(f"{s['taptic_joint']['num_train_chunks']} train chunks -> {a.out}")


if __name__ == "__main__":
    main()
