"""Build stage of ``taptic.trex_style``: write the re-cut LeRobot v3.0 dataset.

* data/      frames copied from the source, re-indexed (episode_index,
             frame_index, index, timestamp, task_index); ``language_persistent``
             rebuilt from the per-frame ``subtask_index`` of the piece.
* videos/    the source mp4 files, hard-linked (no re-encoding); each new episode
             points into them with shifted from/to timestamps.
* tactile_raw/ per new episode: the source sidecar, sliced for cut pieces
             (2 s of earlier samples kept as history, never past the source
             recording start) and copied as-is for whole recordings.
* meta/      info.json, tasks.parquet (one task per caption), episodes with
             caption / motor_primitive / object / target / provenance columns,
             stats.json recomputed on the kept frames (image stats: source).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .trex_style import CARVE_CAPTIONS, CARVE_OBJECT, TASKS

FPS = 10
HISTORY_NS = 2_000_000_000
# LeRobot-Annotate columns with the non-standard "language" dtype (stock LeRobot
# cannot load them); the same information is in the per-frame subtask_index.
DROP_COLUMNS = ("language_persistent", "language_events")
NATIVE_PERIOD_NS = 13_090_000          # median native XHand sidecar interval (76.4 Hz)


def densify_legacy(sm: pd.DataFrame, fr: pd.DataFrame):
    """Rebuild a ~76.4 Hz sidecar from a legacy one-sample-per-frame sidecar.

    Nominal clock: frame k is observed at k * 100 ms. Between frames k-1 and k,
    samples are placed at t_k - m * 13.09 ms (m = 1..7) with force and joints
    linearly interpolated between the two frame samples. Such a sample uses the
    value of frame k, so it only becomes available at t_k (``available_ns``);
    any reader that takes samples with read_end_ns <= the observation time of
    a frame never sees future data. ``interpolated`` marks synthetic samples.
    """
    fs = sm[sm.is_frame_sample].sort_values("frame_index").reset_index(drop=True)
    k = fs.frame_index.to_numpy()
    t = k.astype(np.int64) * (1_000_000_000 // FPS)
    F = np.stack(fs.force.to_numpy()).astype(np.float64)
    J = np.stack(fs.joints.to_numpy()).astype(np.float64)
    zeros_values = fs["values"].iloc[0]
    rows = []
    for i in range(len(fs)):
        if i > 0:
            ms = np.arange(int((t[i] - t[i - 1] - 1) // NATIVE_PERIOD_NS), 0, -1)
            for m in ms:
                tau = t[i] - m * NATIVE_PERIOD_NS
                w = (tau - t[i - 1]) / (t[i] - t[i - 1])
                rows.append(dict(read_end_ns=tau, available_ns=t[i], frame_index=k[i - 1],
                                 is_frame_sample=False, interpolated=True,
                                 force=(1 - w) * F[i - 1] + w * F[i], joints=(1 - w) * J[i - 1] + w * J[i]))
        rows.append(dict(read_end_ns=t[i], available_ns=t[i], frame_index=k[i], is_frame_sample=True,
                         interpolated=False, force=F[i], joints=J[i]))
    d = pd.DataFrame(rows)
    out = pd.DataFrame({
        "t_rel": d.read_end_ns / 1e9, "t_mono": d.read_end_ns / 1e9, "frame_index": d.frame_index,
        "frame_t_rel": d.frame_index / FPS, "is_frame_sample": d.is_frame_sample,
        "sequence": np.arange(len(d)), "read_start_ns": d.read_end_ns, "read_end_ns": d.read_end_ns,
        "reset_sequence": 0, "read_errors": 0, "dropped": 0,
        "joints": [np.asarray(x, np.float32) for x in d.joints],
        "force": [np.asarray(x, np.float32) for x in d.force],
        "values": [zeros_values] * len(d),
        "command_sequence": np.nan, "command_ns": np.nan, "command_q": None, "command_target": None,
        "interpolated": d.interpolated, "available_ns": d.available_ns})
    fr = fr.copy()
    fr["observation_ns"] = (fr.frame_index.astype(np.int64) * (1_000_000_000 // FPS))
    fr["t_rel"] = fr.frame_index / FPS
    return out, fr
STATS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _link(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _num_stats(x: np.ndarray) -> Dict[str, list]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    q = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return dict(min=x.min(0).tolist(), max=x.max(0).tolist(), mean=x.mean(0).tolist(),
                std=x.std(0).tolist(), count=[int(len(x))], q01=q[0].tolist(), q10=q[1].tolist(),
                q50=q[2].tolist(), q90=q[3].tolist(), q99=q[4].tolist())


def _column_matrix(col: pa.ChunkedArray) -> np.ndarray:
    v = col.to_pylist()
    return np.asarray([np.asarray(r, dtype=np.float64).ravel() if r is not None else np.nan for r in v])


def _suggested_split(p: pd.DataFrame, val_ratio: float, seed: int) -> Dict[int, str]:
    """Per task, a deterministic 10 % of source recordings goes to val."""
    out = {}
    for task, g in p.groupby("task"):
        recs = np.array(sorted(g.source_episode.unique()))
        perm = np.random.RandomState(seed).permutation(len(recs))
        n_val = max(1, int(round(val_ratio * len(recs))))
        val = set(recs[perm[:n_val]].tolist())
        out.update({int(r): ("val" if r in val else "train") for r in recs})
    return out


def build(root: str, plan_dir: str, out: str, val_ratio: float = 0.1, seed: int = 42):
    root, out = os.path.expanduser(root), os.path.expanduser(out)
    if os.path.abspath(out).startswith(os.path.abspath(root)):
        raise ValueError("output must not be inside the source dataset")
    plan = pd.read_csv(os.path.join(plan_dir, "pieces.csv"))
    objects = pd.read_csv(os.path.join(plan_dir, "objects.csv")).set_index("piece_id")
    pieces = plan[plan.keep].sort_values("piece_id").reset_index(drop=True)
    info = json.load(open(os.path.join(root, "meta/info.json")))
    sub = {int(k): v for k, v in json.load(open(os.path.join(root, "meta/subtasks.json")))
           ["index_to_subtask"].items()}
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]

    # ── captions ─────────────────────────────────────────────────────────
    meta_rows: List[dict] = []
    for _, r in pieces.iterrows():
        t = TASKS[r.task]
        obj = t["object"]
        review = False
        if "{obj}" in obj:
            obj = objects.loc[r.piece_id, "object"]
            review = bool(objects.loc[r.piece_id, "needs_review"])
        prim = r.get("primitive", "")
        if isinstance(prim, str) and prim:   # carved reach / grasp_and_lifting piece (version 2)
            name = CARVE_OBJECT.get(r.task, obj)
            meta_rows.append(dict(caption=CARVE_CAPTIONS[prim].format(obj=name), motor_primitive=prim,
                                  object=obj, target=None, object_needs_review=review))
            continue
        meta_rows.append(dict(caption=t["caption"].format(obj=obj), motor_primitive=t["primitive"],
                              object=obj, target=t["target"] or None, object_needs_review=review))
    meta = pd.DataFrame(meta_rows)
    captions = sorted(meta.caption.unique())
    task_index = {c: i for i, c in enumerate(captions)}
    split = _suggested_split(pieces, val_ratio, seed)

    # ── frames ───────────────────────────────────────────────────────────
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(root, "data")) for f in fs
                   if f.endswith(".parquet"))
    src = pa.concat_tables([pq.read_table(f) for f in files])
    ep_col = src.column("episode_index").to_numpy()
    fi_col = src.column("frame_index").to_numpy()
    order = np.lexsort((fi_col, ep_col))
    ep_sorted, fi_sorted = ep_col[order], fi_col[order]
    starts = {int(e): int(i) for e, i in zip(*np.unique(ep_sorted, return_index=True))}

    take, new_ep, new_fi, lp = [], [], [], []
    lp_type = src.schema.field("language_persistent").type
    sub_col = src.column("subtask_index").to_numpy()
    for k, r in pieces.iterrows():
        s0 = starts[int(r.source_episode)]
        idx = order[s0 + r.start: s0 + r.end]
        assert (fi_col[idx] == np.arange(r.start, r.end)).all() and (ep_col[idx] == r.source_episode).all()
        take.append(idx)
        new_ep.append(np.full(len(idx), k))
        new_fi.append(np.arange(len(idx)))
        labels = sub_col[idx]
        ch = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        timeline = [dict(role="assistant", content=sub[int(labels[c])], style="subtask",
                         timestamp=float(np.float32(c / FPS)), camera=None, tool_calls=None) for c in ch]
        lp.extend([timeline] * len(idx))
    take = np.concatenate(take)
    tbl = src.take(pa.array(take))
    n = len(take)
    ep = np.concatenate(new_ep)
    fi = np.concatenate(new_fi)
    caption_of_ep = meta.caption.map(task_index).to_numpy()

    def put(name, arr):
        i = tbl.schema.get_field_index(name)
        return tbl.set_column(i, tbl.schema.field(name), arr)

    tbl = put("episode_index", pa.array(ep, pa.int64()))
    tbl = put("frame_index", pa.array(fi, pa.int64()))
    tbl = put("index", pa.array(np.arange(n), pa.int64()))
    tbl = put("timestamp", pa.array((fi / FPS).astype(np.float32), pa.float32()))
    tbl = put("task_index", pa.array(caption_of_ep[ep], pa.int64()))
    tbl = put("language_persistent", pa.array(lp, lp_type))

    for col in DROP_COLUMNS:
        if col in tbl.schema.names:
            tbl = tbl.drop([col])
    os.makedirs(os.path.join(out, "data/chunk-000"), exist_ok=True)
    pq.write_table(tbl, os.path.join(out, "data/chunk-000/file-000.parquet"))

    # ── videos: hard links + shifted ranges ──────────────────────────────
    src_eps = pd.concat([pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in
                         os.walk(os.path.join(root, "meta/episodes")) for f in fs if f.endswith(".parquet")])
    src_eps = src_eps.set_index("episode_index")
    for cam in cams:
        for (c, f) in src_eps[[f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"]].drop_duplicates().values:
            rel = info["video_path"].format(video_key=cam, chunk_index=int(c), file_index=int(f))
            _link(os.path.join(root, rel), os.path.join(out, rel))

    # ── episodes metadata ────────────────────────────────────────────────
    ep_rows = []
    stat_feats = [c[len("stats/"):-len("/min")] for c in src_eps.columns
                  if c.startswith("stats/") and c.endswith("/min")]
    offset = 0
    for k, r in pieces.iterrows():
        se = src_eps.loc[int(r.source_episode)]
        L = int(r.length)
        row = dict(episode_index=k, tasks=[meta.caption[k]], length=L,
                   **{"data/chunk_index": 0, "data/file_index": 0,
                      "dataset_from_index": offset, "dataset_to_index": offset + L})
        for cam in cams:
            f0 = float(se[f"videos/{cam}/from_timestamp"])
            row[f"videos/{cam}/chunk_index"] = int(se[f"videos/{cam}/chunk_index"])
            row[f"videos/{cam}/file_index"] = int(se[f"videos/{cam}/file_index"])
            row[f"videos/{cam}/from_timestamp"] = f0 + r.start / FPS
            row[f"videos/{cam}/to_timestamp"] = f0 + r.end / FPS
        seg = tbl.slice(offset, L)
        for feat in stat_feats:
            if feat in cams:
                for s in STATS:
                    row[f"stats/{feat}/{s}"] = se[f"stats/{feat}/{s}"]
            else:
                st = _num_stats(_column_matrix(seg.column(feat)))
                for s in STATS:
                    row[f"stats/{feat}/{s}"] = st[s]
        labels = seg.column("subtask_index").to_numpy()
        row["subtasks"] = list(dict.fromkeys(sub[int(x)] for x in labels))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        row.update(caption=meta.caption[k], motor_primitive=meta.motor_primitive[k], object=meta.object[k],
                   target=meta.target[k], object_needs_review=bool(meta.object_needs_review[k]),
                   source_task=r.task, source_repo=f"TAPTIC/{r.task}",
                   source_episode_index=int(r.source_episode), source_local_episode=int(r.local_episode),
                   source_frame_start=int(r.start), source_frame_end=int(r.end),
                   object_in_recording=int(r.cycle), tactile_raw=r.tactile_raw, tactile_interpolated=r.tactile_raw != "native",
                   suggested_split=split[int(r.source_episode)])
        ep_rows.append(row)
        offset += L
    episodes = pd.DataFrame(ep_rows)
    os.makedirs(os.path.join(out, "meta/episodes/chunk-000"), exist_ok=True)
    episodes.to_parquet(os.path.join(out, "meta/episodes/chunk-000/file-000.parquet"), index=False)

    pd.DataFrame({"task_index": range(len(captions))}, index=pd.Index(captions, name="task")) \
        .to_parquet(os.path.join(out, "meta/tasks.parquet"))
    os.makedirs(os.path.join(out, "meta/trex_style"), exist_ok=True)
    for f in ("pieces.csv", "objects.csv"):
        shutil.copy2(os.path.join(plan_dir, f), os.path.join(out, "meta/trex_style", f))
    for f in ("subtasks.json", "subtasks.parquet"):
        shutil.copy2(os.path.join(root, "meta", f), os.path.join(out, "meta", f))

    # ── global stats ─────────────────────────────────────────────────────
    stats = json.load(open(os.path.join(root, "meta/stats.json")))
    for feat in list(stats):
        if feat in cams or feat not in tbl.schema.names:
            continue
        st = _num_stats(_column_matrix(tbl.column(feat)))
        shape = np.asarray(stats[feat]["min"]).shape
        stats[feat] = {s: (np.asarray(v).reshape(shape).tolist() if s != "count" else v) for s, v in st.items()}
    json.dump(stats, open(os.path.join(out, "meta/stats.json"), "w"), indent=2)

    info["features"] = {k: v for k, v in info["features"].items() if k not in DROP_COLUMNS}
    info = dict(info, total_episodes=len(pieces), total_frames=int(n), total_tasks=len(captions),
                splits={"train": f"0:{len(pieces)}"})
    json.dump(info, open(os.path.join(out, "meta/info.json"), "w"), indent=4)

    # ── tactile sidecar ──────────────────────────────────────────────────
    traw_in, traw_out = os.path.join(root, "tactile_raw"), os.path.join(out, "tactile_raw")
    os.makedirs(traw_out, exist_ok=True)
    tinfo = json.load(open(os.path.join(traw_in, "info.json")))
    tinfo["trex_style_note"] = (
        "Files are per re-cut episode. For episodes cut out of a longer recording, samples start "
        "2 s before the episode's first frame (or at the recording start) and stop at the observation "
        "of its last frame; frame_index is renumbered to the new episode (-1 = before its first frame). "
        "t_rel / frame_t_rel / timestamps stay relative to the SOURCE recording. "
        "Episodes with tactile_interpolated=True (insert_game_object_cylinder, fruit_in_the_basket, "
        "unscrew_the_bottle) were recorded with one XHand sample per 10 Hz frame and no clock: their "
        "sidecar uses a nominal clock (frame k at k*100 ms) and adds 7 samples between frames "
        "(13.09 ms spacing, 76.4 Hz) with force and joints linearly interpolated between the two "
        "frame samples. Those rows have interpolated=True and available_ns = time of the next frame: "
        "they are synthetic and only usable once that frame is observed. 'values' (taxels) are "
        "zeros in these episodes (not recorded).")
    json.dump(tinfo, open(os.path.join(traw_out, "info.json"), "w"), indent=2)
    side = tinfo.get("side", "right")
    for k, r in pieces.iterrows():
        s_src = os.path.join(traw_in, f"episode_{int(r.source_episode):06d}_{side}.parquet")
        f_src = os.path.join(traw_in, f"episode_{int(r.source_episode):06d}_frames.parquet")
        s_dst = os.path.join(traw_out, f"episode_{k:06d}_{side}.parquet")
        f_dst = os.path.join(traw_out, f"episode_{k:06d}_frames.parquet")
        legacy = r.tactile_raw != "native"
        if r.start == 0 and r.end == r.source_length and not legacy:
            _link(s_src, s_dst); _link(f_src, f_dst)
            continue
        fr = pd.read_parquet(f_src)
        sm = pd.read_parquet(s_src)
        if legacy:
            sm, fr = densify_legacy(sm, fr)
        obs = fr.set_index("frame_index").observation_ns
        lo = int(obs.loc[r.start]) - HISTORY_NS
        hi = int(obs.loc[r.end - 1])   # nothing after the piece's last observation
        keep = (sm.read_end_ns >= lo) & (sm.read_end_ns <= hi)
        fr = fr[(fr.frame_index >= r.start) & (fr.frame_index < r.end)].copy()
        fr["frame_index"] -= r.start
        fr["timestamp"] = (fr.frame_index / FPS).astype(fr.timestamp.dtype)
        fr.to_parquet(f_dst, index=False)
        sm = sm[keep].copy()
        fi_new = sm.frame_index - r.start
        before = (sm.frame_index < r.start)
        sm["frame_index"] = np.where(before, -1, fi_new)
        sm.loc[before, "frame_t_rel"] = np.nan
        sm.loc[before, "is_frame_sample"] = False
        sm.to_parquet(s_dst, index=False)

    print(f"wrote {len(pieces)} episodes, {n} frames, {len(captions)} captions -> {out}")
