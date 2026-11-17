"""Primitive annotation loading + validation for one TAPTIC task.

Sources (all read-only):
  * per-frame ``subtask_index`` / ``annotation_review_required`` in the merged
    LeRobot data -> AUTHORITATIVE (it is what the dataset exposes to loaders);
  * ``meta/sources/<task>/meta/segments.csv``          -> cross-check (frames);
  * ``meta/sources/<task>/meta/reviewed_segments.csv`` -> per-segment review flag + notes;
  * ``meta/sources/<task>/meta/lerobot_annotations.json`` -> cross-check (seconds);
  * ``meta/sources/<task>/meta/ambiguous_intervals.csv`` / ``boundary_review.csv``;
  * merged per-frame ``language_persistent`` (subtask timeline) -> cross-check.

Any disagreement is reported (and makes ``validate`` return errors) instead of
being silently resolved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .source import TapticSource


@dataclass
class Segment:
    episode: int          # merged episode index
    local_episode: int    # episode index in the source repo
    seg_id: int           # 0.. within the episode
    start: int            # first frame (inclusive)
    end: int              # last frame + 1
    label: str
    review_required: bool          # reviewed_segments.csv flag
    frac_frames_flagged: float     # mean of per-frame annotation_review_required
    boundary_uncertainty_start: float   # frames, boundary_review.csv (nan if none)
    boundary_uncertainty_end: float
    overlaps_ambiguous: bool       # intersects an ambiguous_intervals.csv interval

    @property
    def length(self) -> int:
        return self.end - self.start


def _runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Run-length encode -> [(start, end_exclusive, value)]."""
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(labels)]
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def load_segments(src: TapticSource) -> List[Segment]:
    inv = {v: k for k, v in src.task.subtask_index.items()}
    frames = src.frames_table()
    rev = pd.read_csv(src.source_file("reviewed_segments.csv"))
    bnd = pd.read_csv(src.source_file("boundary_review.csv"))
    amb = pd.read_csv(src.source_file("ambiguous_intervals.csv"))
    segs: List[Segment] = []
    for ep, df in frames.groupby("episode_index"):
        loc = src.task.local_episode(int(ep))
        flags = df.annotation_review_required.to_numpy(bool)
        r_ep = rev[rev.episode_index == loc]
        b_ep = bnd[bnd.episode_index == loc].set_index("frame")["estimated_uncertainty_frames"]
        a_ep = amb[amb.episode_index == loc]
        for k, (s, e, v) in enumerate(_runs(df.subtask_index.to_numpy())):
            r = r_ep[(r_ep.start_frame == s) & (r_ep.end_frame_exclusive == e)]
            segs.append(Segment(
                episode=int(ep), local_episode=loc, seg_id=k, start=s, end=e, label=inv[v],
                review_required=bool(r.review_required.iloc[0]) if len(r) else True,
                frac_frames_flagged=float(flags[s:e].mean()),
                boundary_uncertainty_start=float(b_ep.get(s, np.nan)),
                boundary_uncertainty_end=float(b_ep.get(e, np.nan)),
                overlaps_ambiguous=bool(((a_ep.start_frame < e) & (a_ep.end_frame_exclusive > s)).any()),
            ))
    return segs


def validate(src: TapticSource, segs: List[Segment]) -> Dict:
    """Cross-check every annotation source; returns a JSON-able report."""
    errors: List[str] = []
    warnings: List[str] = []
    fps = src.fps
    frames = src.frames_table()
    lengths = frames.groupby("episode_index").size().to_dict()
    csv = pd.read_csv(src.source_file("segments.csv"))
    rev = pd.read_csv(src.source_file("reviewed_segments.csv"))
    with open(src.source_file("lerobot_annotations.json")) as f:
        ann = json.load(f)["episodes"]
    with open(src.source_file("subtasks.json")) as f:
        src_names = json.load(f)["subtask_to_index"]

    if set(src_names) != set(src.task.subtask_index):
        errors.append(f"label set differs: source {sorted(src_names)} vs merged {sorted(src.task.subtask_index)}")

    by_ep: Dict[int, List[Segment]] = {}
    for s in segs:
        by_ep.setdefault(s.episode, []).append(s)

    for ep, ss in by_ep.items():
        loc = ss[0].local_episode
        n = lengths[ep]
        # coverage / ordering / overlap (by construction runs partition the episode;
        # we still check explicitly so the report states it)
        if ss[0].start != 0 or ss[-1].end != n:
            errors.append(f"ep {ep}: segments do not cover [0,{n})")
        for a, b in zip(ss[:-1], ss[1:]):
            if b.start != a.end:
                errors.append(f"ep {ep}: gap/overlap between seg {a.seg_id} and {b.seg_id}")
        # segments.csv (frames)
        c = csv[csv.episode_index == loc].sort_values("start_frame")
        mine = [(s.start, s.end, s.label) for s in ss]
        theirs = list(zip(c.start_frame, c.end_frame_exclusive, c.label))
        if mine != theirs:
            # identical consecutive labels are merged by run-length encoding; re-merge csv to compare
            merged: List[Tuple[int, int, str]] = []
            for s0, e0, l0 in theirs:
                if merged and merged[-1][2] == l0 and merged[-1][1] == s0:
                    merged[-1] = (merged[-1][0], e0, l0)
                else:
                    merged.append((s0, e0, l0))
            if mine == merged:
                warnings.append(f"ep {ep}: segments.csv has adjacent identical labels (merged)")
            else:
                errors.append(f"ep {ep}: per-frame subtask_index disagrees with segments.csv")
        if len(rev[rev.episode_index == loc]) != len(c):
            errors.append(f"ep {ep}: reviewed_segments.csv row count != segments.csv")
        # lerobot_annotations.json (seconds) -> frames
        a = ann.get(str(loc), {}).get("subtasks", [])
        a_fr = [(int(round(x["start"] * fps)), int(round(x["end"] * fps)), x["label"]) for x in a]
        if a_fr != theirs:
            errors.append(f"ep {ep}: lerobot_annotations.json disagrees with segments.csv")
        # language_persistent timeline (first frame of the episode carries the whole timeline)
        lp = frames[frames.episode_index == ep].language_persistent.iloc[0]
        lp_starts = [(int(round(float(x["timestamp"]) * fps)), x["content"]) for x in lp if x.get("style") == "subtask"]
        if lp_starts != [(s0, l0) for s0, _, l0 in theirs]:
            errors.append(f"ep {ep}: language_persistent timeline disagrees with segments.csv")

    labels = sorted({s.label for s in segs})
    per_label = {}
    for lab in labels:
        L = np.array([s.length for s in segs if s.label == lab])
        per_label[lab] = {
            "segments": int(len(L)), "frames": int(L.sum()),
            "len_min": int(L.min()), "len_p25": float(np.percentile(L, 25)),
            "len_median": float(np.median(L)), "len_max": int(L.max()),
            "review_required": int(sum(s.review_required for s in segs if s.label == lab)),
            "overlaps_ambiguous": int(sum(s.overlaps_ambiguous for s in segs if s.label == lab)),
        }
    # transitions
    trans: Dict[str, int] = {}
    for ss in by_ep.values():
        for a, b in zip(ss[:-1], ss[1:]):
            k = f"{a.label}->{b.label}"
            trans[k] = trans.get(k, 0) + 1
    return {
        "task": src.task.name, "episodes": len(by_ep), "segments": len(segs),
        "authoritative_source": "per-frame subtask_index in merged LeRobot data",
        "errors": errors, "warnings": warnings, "per_label": per_label,
        "transitions": dict(sorted(trans.items(), key=lambda kv: -kv[1])),
        "episodes_single_segment": sorted(ep for ep, ss in by_ep.items() if len(ss) == 1),
    }


def segments_frame(segs: List[Segment]) -> pd.DataFrame:
    return pd.DataFrame([{**asdict(s), "length": s.length} for s in segs])
