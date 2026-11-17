"""Test C — XHand [5,3] tactile: source, units, causality, history windows, normalization."""
import json
import os

import numpy as np

from taptic.constants import STATE_XHAND_FORCE
from taptic.tactile import EpisodeTactile, TactileNorm


def _ep(tdir, ep):
    z = np.load(os.path.join(tdir, "episodes", f"episode_{ep:06d}.npz"))
    return z, EpisodeTactile(z["tac_t_ns"], z["tac_force"], z["tac_reset"], int(z["tac_start_idx"]))


def test_sidecar_force_is_state_force(src):
    """The frame's is_frame_sample force == observation.state[19:34] (same SDK calc_force)."""
    for ep in src.task.episodes[::9]:
        s = src.tactile_samples(ep)
        st = np.stack(src.episode_frames(ep)["observation.state"].to_numpy())
        fs = s[s.is_frame_sample]
        F = np.stack(fs.force.to_numpy()).astype(np.float32)
        assert np.array_equal(F, st[fs.frame_index.to_numpy(), STATE_XHAND_FORCE])
        assert np.all(np.mod(F, 1) == 0)                                   # integer SDK counts
        assert F[:, 2::3].min() >= 0                                        # Fz is uint8 (>= 0)


def test_current_obs_shape_and_causality(tdir, samples):
    for ep in samples.episode.unique()[::5]:
        z, tac = _ep(tdir, ep)
        obs = z["obs_ns"]
        for t in range(0, len(obs), 13):
            i = tac.latest_index(int(obs[t]))
            assert i >= tac.start_idx
            assert tac.t_ns[i] <= obs[t]                                    # never the future
            assert i + 1 >= len(tac.t_ns) or tac.t_ns[i + 1] > obs[t]        # the freshest one
            assert tac.current(int(obs[t])).shape == (5, 3)


def test_history_windows(tdir, samples):
    for ep in samples.episode.unique()[::4]:
        z, tac = _ep(tdir, ep)
        obs = z["obs_ns"]
        for t in list(range(0, min(len(obs), 5))) + list(range(5, len(obs), 17)):
            end = tac.latest_index(int(obs[t]))
            idx = tac.window_indices(end, 16)
            assert idx.shape == (16,) and idx[-1] == end
            assert np.all(np.diff(idx) >= 0)                                # oldest -> newest
            assert idx.min() >= tac.start_idx                                # no pre-roll
            assert (tac.reset[idx] == tac.reset[end]).all()                 # no re-zero crossing
            assert (tac.t_ns[idx] <= obs[t]).all()                           # causal
            assert tac.history(int(obs[t])).shape == (16, 5, 3)
            if idx[0] != idx[1]:                                             # unpadded window: native rate
                span_ms = (tac.t_ns[idx[-1]] - tac.t_ns[idx[0]]) / 1e6
                assert 150 < span_ms < 260 or (np.diff(tac.t_ns[idx]) > 30e6).any()


def test_reset_episode_handled(tdir):
    """Episode 606 (local 4) has one tactile re-zero; windows after it start after it."""
    z, tac = _ep(tdir, 606)
    cut = int(np.flatnonzero(np.diff(tac.reset))[0]) + 1
    idx = tac.window_indices(cut + 3, 16)
    assert idx.min() == cut and (idx[:13] == cut).all()


def test_normalization_train_only_and_range(tdir, samples):
    blob = json.load(open(os.path.join(tdir, "norm_stats.json")))
    norm = TactileNorm.from_dict(blob[next(iter(blob))]["tactile_f6"])
    sp = json.load(open(os.path.join(tdir, "split.json")))
    flat = []
    for ep in sp["train_merged"]:
        z, tac = _ep(tdir, ep)
        last = int(np.searchsorted(tac.t_ns, int(z["obs_ns"][-1]), side="right"))
        flat.append(tac.force[tac.start_idx:last].reshape(-1, 15))
    ref = TactileNorm.fit(np.concatenate(flat))
    np.testing.assert_allclose(ref.q01, norm.q01)
    np.testing.assert_allclose(ref.q99, norm.q99)
    z, tac = _ep(tdir, sp["val_merged"][0])
    x = norm.normalize(tac.force)
    assert x.shape == tac.force.shape and x.min() >= -1 and x.max() <= 1
    assert norm.degenerate[6:].all() and not norm.degenerate[:6].any()    # only thumb/index see contact
