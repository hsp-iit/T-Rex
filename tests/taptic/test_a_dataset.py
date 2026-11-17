"""Test A — dataset integrity: episodes, primitive annotations, valid chunks, split."""
import json
import os

import numpy as np
import pandas as pd

from taptic.annotations import load_segments, validate
from taptic.convert import episode_split


def test_task_and_episodes(src):
    t = src.task
    assert t.tactile_raw == "native"
    eps = src.episodes_table()
    assert eps.episode_index.tolist() == t.episodes
    f = src.frames_table()
    for ep, d in f.groupby("episode_index"):
        assert d.frame_index.tolist() == list(range(len(d)))           # contiguous, no duplicates
        assert np.allclose(d.timestamp.to_numpy(), np.arange(len(d)) / src.fps, atol=1e-4)
    assert len(f) == int(eps.length.sum())


def test_annotations_consistent_across_all_sources(src):
    segs = load_segments(src)
    rep = validate(src, segs)
    assert rep["errors"] == [], rep["errors"][:5]
    assert rep["segments"] == 896                                      # documented in ANNOTATION_README


def test_samples_never_cross_primitive_or_episode(samples, src):
    lengths = src.frames_table().groupby("episode_index").size()
    assert (samples.frame >= samples.seg_start).all()
    assert (samples.frame + 16 <= samples.seg_end).all()
    assert (samples.seg_end <= samples.episode.map(lengths)).all()
    assert (samples.max_delay_frame < samples.seg_end).all()            # delayed tactile stays in primitive
    f = src.frames_table().set_index(["episode_index", "frame_index"]).subtask_index
    inv = src.task.subtask_index
    for r in samples.sample(500, random_state=0).itertuples():
        labs = {int(f.loc[(r.episode, r.frame + k)]) for k in range(16)}
        assert labs == {inv[r.primitive]}


def test_chunk_accounting(tdir, samples):
    seg = pd.read_parquet(os.path.join(tdir, "segments.parquet"))
    assert (seg.n_valid_strict == np.maximum(0, seg.length - 15)).all()
    assert (seg.n_kept <= seg.n_valid_strict).all()
    assert int(seg.n_kept.sum()) == len(samples)


def test_frame_gaps_excluded(tdir, samples):
    for ep, d in samples.groupby("episode"):
        z = np.load(os.path.join(tdir, "episodes", f"episode_{ep:06d}.npz"))
        gap = z["frame_gap"]
        for t in d.frame.to_numpy():
            assert not gap[t:t + 15].any()


def test_split_episode_level_and_deterministic(tdir, samples, src):
    sp = json.load(open(os.path.join(tdir, "split.json")))
    assert not set(sp["train_merged"]) & set(sp["val_merged"])
    assert sorted(sp["train_merged"] + sp["val_merged"]) == src.task.episodes
    tr, va = episode_split([src.task.local_episode(e) for e in src.task.episodes], sp["val_ratio"], sp["seed"])
    assert tr == sp["train_local"] and va == sp["val_local"]
    per_ep = samples.groupby("episode").split.nunique()
    assert (per_ep == 1).all()                                          # no episode in both splits
    assert set(samples[samples.split == "val"].episode) <= set(sp["val_merged"])
