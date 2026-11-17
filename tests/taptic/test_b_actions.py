"""Test B — 18-D -> 21-D delta-base conversion, inverse reconstruction, conventions."""
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from taptic.actions import (chunk_21d, reconstruct_chunk, rot6d_to_matrix, rotation_angle_error,
                            rotvec_gd_to_matrix)
from taptic.constants import STATE_EE_POS, STATE_EE_ROTVEC, STATE_PANDA_JOINTS


def _episodes(tdir, samples):
    for ep, d in samples.groupby("episode"):
        z = np.load(os.path.join(tdir, "episodes", f"episode_{ep:06d}.npz"))
        yield ep, d, z


def test_all_chunks_shape_and_recompute(tdir, samples):
    n = 0
    for ep, d, z in _episodes(tdir, samples):
        C = z["chunks21"]
        assert C.shape[1:] == (16, 21) and np.isfinite(C).all()
        assert np.array_equal(z["chunk_anchor"], d.frame.to_numpy())
        for i in range(0, len(d), 7):
            np.testing.assert_allclose(chunk_21d(z["state"], z["action"], int(d.frame.iloc[i])), C[i], atol=1e-6)
        n += len(C)
    assert n == len(samples)


def test_rot6d_valid(tdir, samples):
    for ep, d, z in _episodes(tdir, samples):
        r6 = z["chunks21"][..., 3:9].astype(np.float64)
        a, b = r6[..., :3], r6[..., 3:]
        assert np.abs(np.linalg.norm(a, axis=-1) - 1).max() < 1e-5
        assert np.abs(np.linalg.norm(b, axis=-1) - 1).max() < 1e-5
        assert np.abs((a * b).sum(-1)).max() < 1e-5
        assert np.abs(np.linalg.det(rot6d_to_matrix(r6)) - 1).max() < 1e-5


def test_inverse_reconstruction_matches_recorded_commands(tdir, samples):
    pe, re = [], []
    for ep, d, z in _episodes(tdir, samples):
        S, A, C = z["state"], z["action"], z["chunks21"]
        for i, t in enumerate(d.frame.to_numpy()):
            rec = reconstruct_chunk(C[i], S[t, STATE_EE_POS].astype(np.float64), S[t, STATE_EE_ROTVEC])
            tgt = A[t:t + 16].astype(np.float64)
            pe.append(np.abs(rec[:, :3] - tgt[:, :3]).max())
            re.append(rotation_angle_error(rec[:, 3:6], tgt[:, 3:6]).max())
            assert np.array_equal(C[i][:, 9:], A[t:t + 16, 6:18])        # absolute hand targets, same order
    assert max(pe) < 1e-5, max(pe)        # metres (float32 storage)
    assert max(re) < 1e-5, max(re)        # rad


def test_delta_base_reference_is_measured_pose(tdir, samples):
    """k=0 is NOT identity: it is action[t] seen from the measured pose state[t]."""
    ep, d, z = next(_episodes(tdir, samples))
    t = int(d.frame.iloc[10])
    S, A = z["state"].astype(np.float64), z["action"].astype(np.float64)
    R0 = rotvec_gd_to_matrix(S[t, STATE_EE_ROTVEC])
    dp = R0.T @ (A[t, :3] - S[t, STATE_EE_POS])
    np.testing.assert_allclose(z["chunks21"][10][0, :3], dp, atol=1e-6)


def test_orientation_convention_by_forward_kinematics(src):
    """ee.w* == rotvec(Rx(pi)^T R_base) of panda_link8 (FK on joints in degrees)."""
    try:
        from lerobot.model.kinematics import RobotKinematics
    except Exception:
        pytest.skip("lerobot kinematics (placo) unavailable")
    urdf = os.path.expanduser("~/lerobot_plugins/lerobot_robot_panda/lerobot_robot_panda/urdf/panda.urdf")
    if not os.path.isfile(urdf):
        pytest.skip("panda urdf missing")
    k = RobotKinematics(urdf_path=urdf, target_frame_name="panda_link8",
                        joint_names=[f"panda_joint{i}" for i in range(1, 8)])
    st = np.stack(src.episode_frames(src.task.ep_start)["observation.state"].to_numpy())
    for i in range(0, len(st), 40):
        T = k.forward_kinematics(st[i, STATE_PANDA_JOINTS].astype(float))
        assert np.abs(T[:3, 3] - st[i, STATE_EE_POS]).max() < 1e-5
        err = (Rot.from_matrix(T[:3, :3]).inv() * Rot.from_matrix(rotvec_gd_to_matrix(st[i, STATE_EE_ROTVEC]))).magnitude()
        assert err < 1e-5


def test_synthetic_roundtrip():
    rng = np.random.default_rng(0)
    N = 40
    state = np.zeros((N, 40), np.float32)
    action = np.zeros((N, 18), np.float32)
    state[:, 34:37] = rng.normal(0.4, 0.1, (N, 3))
    state[:, 37:40] = Rot.random(N, random_state=1).as_rotvec()
    action[:, :3] = rng.normal(0.4, 0.1, (N, 3))
    action[:, 3:6] = Rot.random(N, random_state=2).as_rotvec()
    action[:, 6:] = rng.uniform(0, 1.5, (N, 12))
    for t in (0, 5, 24):
        c = chunk_21d(state, action, t)
        rec = reconstruct_chunk(c, state[t, 34:37].astype(np.float64), state[t, 37:40])
        assert np.abs(rec[:, :3] - action[t:t + 16, :3]).max() < 1e-5
        assert rotation_angle_error(rec[:, 3:6], action[t:t + 16, 3:6]).max() < 1e-5
    with pytest.raises(IndexError):
        chunk_21d(state, action, N - 10)                                  # never pads


def test_hand_targets_within_recorded_range(tdir, samples):
    lo, hi = np.full(12, np.inf), np.full(12, -np.inf)
    for ep, d, z in _episodes(tdir, samples):
        h = z["chunks21"][..., 9:].reshape(-1, 12)
        lo, hi = np.minimum(lo, h.min(0)), np.maximum(hi, h.max(0))
    assert lo.min() >= 0.0 and hi.max() <= 1.9                            # rad, XHand command range seen in data
