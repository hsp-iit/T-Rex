"""Build the primitive-level T-REX training set for one TAPTIC task.

    # standalone Hub download (videos/ + tactile_raw/ included)
    python -m taptic.convert --root ~/midtrain_merge/src/insert_credit_card \
        --out ~/midtrain_merge/derived/trex_xhand_v1
    # or the merged multi-task dataset
    python -m taptic.convert --root ~/midtrain_merge/TAPTIC/midtrain_dataset \
        --task insert_credit_card --out ~/midtrain_merge/derived/trex_xhand_v1

The source dataset is only read.  Output (``<out>/<task>/``):
  episodes/episode_XXXXXX.npz  per-episode arrays (state, action, frame times,
                               native tactile stream, 21-D chunks of valid samples)
  samples.parquet              one row per training sample (episode, frame,
                               primitive, instruction, split, provenance)
  segments.parquet             every annotated segment + chunk accounting
  split.json                   deterministic episode-level train/val split
  norm_stats.json              q01/q99 stats (TRAIN episodes only)
  conversion_report.json       counts, losses, class balance, timing checks
  manifest.json                source path/revision, parameters, code version

A sample anchored at frame t is valid iff
  1. frames t..t+15 lie in ONE annotated primitive segment (never crosses a
     primitive or episode boundary; incomplete chunks are dropped, not padded);
  2. no real-time inter-frame gap > --max_frame_gap_s inside t..t+15
     (teleop pauses; --gap_policy ignore disables this rule);
  3. a causal tactile sample exists at the observation time of frame t.
Tactile history is NOT limited to the primitive: it is the native stream up to
the anchor time, inside the episode.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .actions import chunk_21d
from .annotations import load_segments, segments_frame, validate
from .constants import ACTION_CHUNK, ACTION_DIM, MAX_FRAME_GAP_S, TAC_FLAT
from .instructions import instruction_for
from .source import TapticSource
from .tactile import EpisodeTactile, TactileNorm


def episode_split(local_eps, val_ratio: float, seed: int):
    eps = np.array(sorted(local_eps))
    perm = np.random.RandomState(seed).permutation(len(eps))
    n_val = max(1, int(round(len(eps) * val_ratio)))
    return sorted(eps[perm[n_val:]].tolist()), sorted(eps[perm[:n_val]].tolist())


def _git_rev() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def convert(root: str, task: Optional[str], out: str, val_ratio: float = 0.1, seed: int = 42,
            max_frame_gap_s: float = MAX_FRAME_GAP_S, gap_policy: str = "exclude",
            allow_preroll: bool = False, chunk: int = ACTION_CHUNK, delay_offsets=(0, 4, 8, 12),
            revision: Optional[str] = None) -> dict:
    src = TapticSource(root, task, revision)
    task = src.task.name
    if src.task.tactile_raw != "native":
        raise RuntimeError(
            f"{task}: tactile sidecar is {src.task.tactile_raw!r} (10 Hz, rebuilt from state). "
            "The XHand VQ-VAE windows are defined on the native ~76 Hz stream; mixing 10 Hz "
            "windows would change their temporal meaning. Not converted.")
    segs = load_segments(src)
    ann_report = validate(src, segs)
    if ann_report["errors"]:
        raise RuntimeError(f"annotation validation failed: {ann_report['errors'][:5]}")

    out_dir = os.path.join(os.path.expanduser(out), task)
    os.makedirs(os.path.join(out_dir, "episodes"), exist_ok=True)

    local_eps = [src.task.local_episode(e) for e in src.task.episodes]
    train_loc, val_loc = episode_split(local_eps, val_ratio, seed)
    split_of = {src.task.ep_start + l: "train" for l in train_loc}
    split_of.update({src.task.ep_start + l: "val" for l in val_loc})

    labels = sorted(src.task.subtask_index, key=lambda k: src.task.subtask_index[k])
    seg_rows, sample_rows = [], []
    lost = Counter()
    timing = defaultdict(list)
    train_tac, train_chunks = [], []

    by_ep = defaultdict(list)
    for s in segs:
        by_ep[s.episode].append(s)

    for ep in src.task.episodes:
        fr = src.episode_frames(ep)
        ft = src.frame_times(ep).sort_values("frame_index").reset_index(drop=True)
        assert np.array_equal(ft.frame_index.to_numpy(), fr.frame_index.to_numpy()), ep
        state = np.stack(fr["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(fr["action"].to_numpy()).astype(np.float32)
        obs_ns = ft.observation_ns.to_numpy(np.int64)
        n = len(fr)
        gaps = np.diff(obs_ns) / 1e9                                   # [n-1]
        big_gap = gaps > max_frame_gap_s
        tac = EpisodeTactile.from_frame(src.tactile_samples(ep), int(obs_ns[0]), allow_preroll)
        cur_idx = np.array([tac.latest_index(int(t)) for t in obs_ns], dtype=np.int64)
        timing["frame_interval_ms"].append(gaps * 1e3)
        ok_cur = cur_idx >= 0
        timing["current_age_ms"].append((obs_ns[ok_cur] - tac.t_ns[cur_idx[ok_cur]]) / 1e6)
        last_ok = int(np.searchsorted(tac.t_ns, obs_ns[-1], side="right"))
        in_ep = slice(tac.start_idx, last_ok)
        timing["sample_interval_ms"].append(np.diff(tac.t_ns[in_ep]) / 1e6)

        chunks, anchors = [], []
        for s in by_ep[ep]:
            n_valid_strict = max(0, s.length - chunk + 1)
            n_kept = 0
            for t in range(s.start, s.end - chunk + 1):
                if gap_policy == "exclude" and big_gap[t:t + chunk - 1].any():
                    lost["frame_gap"] += 1
                    continue
                if cur_idx[t] < 0:
                    lost["no_causal_tactile"] += 1
                    continue
                c = chunk_21d(state, action, t, chunk)
                chunks.append(c)
                anchors.append(t)
                n_kept += 1
                sample_rows.append(dict(
                    episode=ep, local_episode=s.local_episode, frame=t, seg_id=s.seg_id,
                    primitive=s.label, primitive_index=labels.index(s.label),
                    instruction=instruction_for(task, s.label),
                    seg_start=s.start, seg_end=s.end, review_required=s.review_required,
                    frac_frames_flagged=s.frac_frames_flagged, overlaps_ambiguous=s.overlaps_ambiguous,
                    split=split_of[ep], chunk_row=len(chunks) - 1,
                    max_delay_frame=t + max(delay_offsets),
                ))
            lost["short_segment_frames"] += s.length - n_valid_strict
            seg_rows.append(dict(
                episode=s.episode, seg_id=s.seg_id, n_valid_strict=n_valid_strict, n_kept=n_kept,
                n_anchor_cross_allowed=max(0, min(s.end, n - chunk + 1) - s.start),
                split=split_of[ep]))
        chunks_arr = np.asarray(chunks, np.float32).reshape(-1, chunk, ACTION_DIM)
        np.savez_compressed(
            os.path.join(out_dir, "episodes", f"episode_{ep:06d}.npz"),
            state=state, action=action, obs_ns=obs_ns, frame_gap=big_gap,
            tac_t_ns=tac.t_ns, tac_force=tac.force.astype(np.float32), tac_reset=tac.reset,
            tac_start_idx=np.int64(tac.start_idx), cur_tac_idx=cur_idx,
            chunks21=chunks_arr, chunk_anchor=np.asarray(anchors, np.int64))
        if split_of[ep] == "train":
            train_tac.append(tac.force[in_ep].reshape(-1, TAC_FLAT))
            if len(chunks_arr):
                train_chunks.append(chunks_arr)

    samples = pd.DataFrame(sample_rows)
    samples.insert(0, "sample_idx", np.arange(len(samples)))
    seg_df = segments_frame(segs).merge(pd.DataFrame(seg_rows), on=["episode", "seg_id"])
    samples.to_parquet(os.path.join(out_dir, "samples.parquet"), index=False)
    seg_df.to_parquet(os.path.join(out_dir, "segments.parquet"), index=False)

    # ── normalization: TRAIN episodes only ──
    A = np.concatenate(train_chunks)                                   # [M,16,21]
    tnorm = TactileNorm.fit(np.concatenate(train_tac))
    stats_key = f"taptic_{task}"
    norm = {stats_key: {
        "action": {"mask": [True] * ACTION_DIM,
                   "q01": np.quantile(A, 0.01, axis=0).tolist(),
                   "q99": np.quantile(A, 0.99, axis=0).tolist(),
                   "min": A.min(0).tolist(), "max": A.max(0).tolist()},
        "tactile_f6": {**tnorm.to_dict(), "units": "XHand SDK calc_force raw counts",
                       "source": "native sidecar samples inside TRAIN episodes"},
        "num_train_chunks": int(A.shape[0]),
    }}
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(norm, f, indent=1)
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"method": "episode-level, numpy RandomState(seed).permutation over sorted local episode ids",
                   "seed": seed, "val_ratio": val_ratio,
                   "train_local": train_loc, "val_local": val_loc,
                   "train_merged": sorted(e for e, s in split_of.items() if s == "train"),
                   "val_merged": sorted(e for e, s in split_of.items() if s == "val")}, f, indent=1)

    # ── report ──
    def pct(x, q):
        x = np.concatenate(x)
        return {f"p{p}": float(np.percentile(x, p)) for p in q} if len(x) else {}
    per_prim = {}
    for lab in labels:
        sd = seg_df[seg_df.label == lab]
        per_prim[lab] = {
            "segments": int(len(sd)), "frames": int(sd.length.sum()),
            "segments_with_>=1_chunk": int((sd.n_valid_strict > 0).sum()),
            "valid_chunks_strict": int(sd.n_valid_strict.sum()),
            "kept_samples": int(sd.n_kept.sum()),
            "kept_train": int(sd[sd.split == "train"].n_kept.sum()),
            "kept_val": int(sd[sd.split == "val"].n_kept.sum()),
            "frames_lost_pct": float(100 * (1 - sd.n_kept.sum() / max(1, sd.length.sum()))),
            "if_chunks_may_cross_boundary": int(sd.n_anchor_cross_allowed.sum()),
        }
    kept = samples.groupby("primitive").size()
    report = {
        "task": task, "source_revision": src.task.revision, "episodes": len(src.task.episodes),
        "samples": int(len(samples)), "train_samples": int((samples.split == "train").sum()),
        "val_samples": int((samples.split == "val").sum()),
        "total_frames": int(seg_df.length.sum()),
        "lost": dict(lost), "per_primitive": per_prim,
        "imbalance_max_over_min": float(kept.max() / max(1, kept.min())),
        "timing": {"frame_interval_ms": pct(timing["frame_interval_ms"], [1, 50, 99, 99.9]),
                   "tactile_sample_interval_ms": pct(timing["sample_interval_ms"], [1, 50, 99, 100]),
                   "current_tactile_age_ms": pct(timing["current_age_ms"], [50, 99, 100])},
        "tactile_degenerate_channels_train": [i for i, d in enumerate(tnorm.degenerate) if d],
        "annotation_validation": {"errors": ann_report["errors"], "warnings": ann_report["warnings"]},
        "params": dict(val_ratio=val_ratio, seed=seed, max_frame_gap_s=max_frame_gap_s,
                       gap_policy=gap_policy, allow_preroll=allow_preroll, chunk=chunk,
                       delay_offsets=list(delay_offsets)),
    }
    with open(os.path.join(out_dir, "conversion_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"source_root": src.root, "source_layout": src.layout, "task": task, "repo_id": src.task.repo_id,
                   "source_revision": src.task.revision, "merged_episodes": [src.task.ep_start, src.task.ep_end],
                   "video_keys": {"slow_head": "observation.images.left", "fast": ["observation.images.wrist"]},
                   "stats_key": stats_key, "code_git_rev": _git_rev(),
                   "created": _dt.datetime.now().isoformat(timespec="seconds"),
                   "params": report["params"]}, f, indent=1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--task", default=None,
                   help="task name; required for the merged dataset, defaults to the folder name "
                        "for a standalone Hub download")
    p.add_argument("--revision", default=None, help="override the recorded source revision")
    p.add_argument("--out", required=True)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frame_gap_s", type=float, default=MAX_FRAME_GAP_S)
    p.add_argument("--gap_policy", choices=["exclude", "ignore"], default="exclude")
    p.add_argument("--allow_preroll", type=int, default=0)
    a = p.parse_args()
    r = convert(a.root, a.task, a.out, a.val_ratio, a.seed, a.max_frame_gap_s, a.gap_policy, bool(a.allow_preroll),
                revision=a.revision)
    print(json.dumps({k: r[k] for k in ("samples", "train_samples", "val_samples", "lost", "imbalance_max_over_min")}, indent=1))
    for k, v in r["per_primitive"].items():
        print(f"  {k:9s} segs={v['segments']:4d} frames={v['frames']:5d} kept={v['kept_samples']:5d} "
              f"(train {v['kept_train']}, val {v['kept_val']}) lost={v['frames_lost_pct']:.1f}%")


if __name__ == "__main__":
    main()
