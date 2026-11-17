"""Test A2 — a standalone Hub download converts to exactly the same data as the merged copy."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from taptic.convert import convert
from taptic.source import TapticSource


def test_standalone_layout_detected(hub_root):
    s = TapticSource(hub_root)
    assert s.layout == "standalone" and s.task.name == "insert_credit_card"
    assert (s.task.ep_start, s.task.ep_end, s.task.tactile_raw) == (0, 100, "native")
    assert len(s.task.revision) == 40


def test_hub_conversion_equals_merged(tmp_path, hub_root, tdir, train_cfg, processor, accel):
    convert(hub_root, None, str(tmp_path))
    h = str(tmp_path / "insert_credit_card")
    a, b = pd.read_parquet(os.path.join(tdir, "samples.parquet")), pd.read_parquet(os.path.join(h, "samples.parquet"))
    cols = ["local_episode", "frame", "primitive", "instruction", "seg_start", "seg_end", "split", "chunk_row"]
    assert a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))
    na, nb = json.load(open(os.path.join(tdir, "norm_stats.json"))), json.load(open(os.path.join(h, "norm_stats.json")))
    assert na == nb
    off = int(a.episode.iloc[0] - a.local_episode.iloc[0])
    for loc in range(0, 100, 7):
        za = np.load(os.path.join(tdir, "episodes", f"episode_{loc + off:06d}.npz"))
        zb = np.load(os.path.join(h, "episodes", f"episode_{loc:06d}.npz"))
        assert all(np.array_equal(za[k], zb[k]) for k in za.files)
    # the loader reads the Hub copy's videos
    from qwen_vla.taptic_dataset import TRexTapticDataset
    cfg = SimpleNamespace(**{**vars(train_cfg), "taptic_dirs": [h]})
    it = TRexTapticDataset(cfg, processor, accel)[123]
    assert it["head"].shape == (480, 640, 3) and len(it["flare_frames"]) == 8
