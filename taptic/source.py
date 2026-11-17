"""Read-only access to a TAPTIC task inside the merged LeRobot v3.0 dataset.

The merged dataset (``TAPTIC/midtrain_dataset``) is the only local copy that
carries the native XHand sidecar (``tactile_raw/``, ~76 Hz) next to the 10 Hz
LeRobot frames and videos.  A task is selected by its source repo name (e.g.
``insert_credit_card``) through ``meta/merged_from.json``.

Nothing in this module writes to the dataset.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CAMERAS = ("observation.images.left", "observation.images.wrist")


@dataclass
class TaskInfo:
    name: str                    # e.g. "insert_credit_card"
    repo_id: str                 # e.g. "TAPTIC/insert_credit_card"
    revision: str
    ep_start: int                # merged episode range [ep_start, ep_end)
    ep_end: int
    tactile_raw: str             # "native" | "legacy_inline_10hz"
    sample_rate_hz: Optional[float]
    subtask_index: Dict[str, int]   # label -> merged subtask_index

    @property
    def episodes(self) -> List[int]:
        return list(range(self.ep_start, self.ep_end))

    def local_episode(self, merged_ep: int) -> int:
        return merged_ep - self.ep_start


def _local_revision(root: str) -> str:
    """Commit hash recorded by ``huggingface_hub.snapshot_download(local_dir=...)``,
    or the snapshot directory name for a hub-cache path; "unknown" otherwise."""
    meta = os.path.join(root, ".cache", "huggingface", "download", "README.md.metadata")
    if os.path.isfile(meta):
        with open(meta) as f:
            return f.readline().strip()
    parts = os.path.normpath(root).split(os.sep)
    if len(parts) > 2 and parts[-2] == "snapshots":
        return parts[-1]
    return "unknown"


class TapticSource:
    """One TAPTIC task, from either layout (detected automatically):

    * **merged**: ``TAPTIC/midtrain_dataset`` (has ``meta/merged_from.json``);
      ``task`` selects the source repo, annotation files live under
      ``meta/sources/<task>/meta``.
    * **standalone**: a full download of one task repo, e.g.
      ``snapshot_download("TAPTIC/insert_credit_card", repo_type="dataset",
      local_dir=...)`` including ``videos/`` and ``tactile_raw/``; annotation
      files live directly in ``meta/``.  ``task`` defaults to the folder name.
    """

    def __init__(self, root: str, task: Optional[str] = None, revision: Optional[str] = None):
        self.root = os.path.abspath(os.path.expanduser(root))
        with open(os.path.join(self.root, "meta", "info.json")) as f:
            self.info = json.load(f)
        tinfo = os.path.join(self.root, "tactile_raw", "info.json")
        if not os.path.isfile(tinfo):
            raise FileNotFoundError(
                f"{self.root}: no tactile_raw/info.json. Download the full repo "
                "(videos/ and tactile_raw/ included), not only meta/ + data/.")
        with open(tinfo) as f:
            self.tactile_info = json.load(f)
        self.fps = int(self.info["fps"])
        merged_path = os.path.join(self.root, "meta", "merged_from.json")
        if os.path.isfile(merged_path):
            self.layout = "merged"
            if task is None:
                raise ValueError("a merged dataset needs --task")
            with open(merged_path) as f:
                merged = json.load(f)
            entry = next((m for m in merged if m["repo_id"].split("/")[-1] == task), None)
            if entry is None:
                names = [m["repo_id"] for m in merged]
                raise KeyError(f"task {task!r} not in merged dataset (have {names})")
            src = self.tactile_info["sources"][entry["repo_id"]]
            self.task = TaskInfo(
                name=task, repo_id=entry["repo_id"], revision=revision or entry["revision"],
                ep_start=int(entry["episodes"][0]), ep_end=int(entry["episodes"][1]),
                tactile_raw=entry["tactile_raw"], sample_rate_hz=src.get("sample_rate_hz"),
                subtask_index=dict(entry["subtask_index"]),
            )
            self.source_meta_dir = os.path.join(self.root, "meta", "sources", task, "meta")
        else:
            self.layout = "standalone"
            task = task or os.path.basename(self.root.rstrip("/"))
            with open(os.path.join(self.root, "meta", "subtasks.json")) as f:
                sub = json.load(f)["subtask_to_index"]
            legacy = bool(self.tactile_info.get("legacy_inline_10hz"))
            self.task = TaskInfo(
                name=task, repo_id=f"TAPTIC/{task}", revision=revision or _local_revision(self.root),
                ep_start=0, ep_end=int(self.info["total_episodes"]),
                tactile_raw="legacy_inline_10hz" if legacy else "native",
                sample_rate_hz=self.tactile_info.get("sample_rate_hz"),
                subtask_index={k: int(v) for k, v in sub.items()},
            )
            self.source_meta_dir = os.path.join(self.root, "meta")

    # ── tables ────────────────────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def episodes_table(self) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.root, "meta", "episodes", "*", "*.parquet")))
        e = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        e = e[(e.episode_index >= self.task.ep_start) & (e.episode_index < self.task.ep_end)]
        return e.sort_values("episode_index").reset_index(drop=True)

    @lru_cache(maxsize=1)
    def frames_table(self) -> pd.DataFrame:
        e = self.episodes_table()
        files = sorted({int(x) for x in e["data/file_index"]})
        chunks = sorted({int(x) for x in e["data/chunk_index"]})
        parts = []
        for c in chunks:
            for fi in files:
                p = os.path.join(self.root, self.info["data_path"].format(chunk_index=c, file_index=fi))
                if os.path.isfile(p):
                    parts.append(pd.read_parquet(p))
        d = pd.concat(parts, ignore_index=True)
        d = d[(d.episode_index >= self.task.ep_start) & (d.episode_index < self.task.ep_end)]
        return d.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    def episode_frames(self, ep: int) -> pd.DataFrame:
        d = self.frames_table()
        return d[d.episode_index == ep].reset_index(drop=True)

    # ── sidecar ───────────────────────────────────────────────────────────
    def frame_times(self, ep: int) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.root, "tactile_raw", f"episode_{ep:06d}_frames.parquet"))

    def tactile_samples(self, ep: int, side: str = "right") -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.root, "tactile_raw", f"episode_{ep:06d}_{side}.parquet"),
                               columns=["t_rel", "frame_index", "is_frame_sample", "sequence",
                                        "read_start_ns", "read_end_ns", "reset_sequence",
                                        "read_errors", "dropped", "force"])

    # ── video ─────────────────────────────────────────────────────────────
    def video_ref(self, ep: int, cam: str):
        """(mp4 path, from_timestamp_s, to_timestamp_s) for one episode/camera."""
        row = self.episodes_table().set_index("episode_index").loc[ep]
        path = os.path.join(self.root, self.info["video_path"].format(
            video_key=cam, chunk_index=int(row[f"videos/{cam}/chunk_index"]),
            file_index=int(row[f"videos/{cam}/file_index"])))
        return path, float(row[f"videos/{cam}/from_timestamp"]), float(row[f"videos/{cam}/to_timestamp"])

    # ── source annotation files ──────────────────────────────────────────
    def source_file(self, name: str) -> str:
        return os.path.join(self.source_meta_dir, name)
