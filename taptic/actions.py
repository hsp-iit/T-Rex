"""18-D TAPTIC action -> 21-D T-REX delta-base action chunks (and back).

Verified conventions (docs/xhand/DATASET_AUDIT.md, section "Pose conventions"):
  * ee.x/y/z      panda_link8 (flange) position in the URDF base frame, metres.
  * ee.wx/wy/wz   rotation vector (axis-angle, rad) of Rx(pi)^T @ R_base, i.e.
                  the recorder's "gripper_down" frame.  Hence
                  R_base = Rx(pi) @ Rot(rotvec).  Checked by forward kinematics
                  of the recorded joints: error < 2e-7 rad / 7e-8 m.
  * action        commanded target pose (closer to state[t+1] than state[t]),
                  same frames as the state.
  * XHand action  absolute joint targets in rad (no deltas).

T-REX delta-base convention (utils/gen_json_tac_deltabase_eef_bimanual_parallel.py,
`compute_chunk_delta_pose`), reproduced exactly:
  reference      = MEASURED pose at the chunk start  (state[t])
  target k       = COMMANDED pose at t+k             (action[t+k]), k = 0..15
  dp_k           = R_t^T (p_{t+k} - p_t)
  dR_k           = R_t^T R_{t+k}
  rot6d          = [dR[:,0], dR[:,1]]  (first two columns)
  hand           = absolute target joints at t+k

Note on frames: dR is invariant to the constant Rx(pi) pre-rotation
((Rx R_a)^T (Rx R_b) = R_a^T R_b), but dp is not, so positions are expressed
with the true base-frame flange rotation R_base.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .constants import (ACTION_CHUNK, ACTION_DIM, ACTION_EE_POS, ACTION_EE_ROTVEC,
                        ACTION_XHAND, GRIPPER_DOWN_ROTVEC, STATE_EE_POS, STATE_EE_ROTVEC)

_RX_PI = Rot.from_rotvec(GRIPPER_DOWN_ROTVEC).as_matrix()


def rotvec_gd_to_matrix(rotvec_gd: np.ndarray) -> np.ndarray:
    """Recorder rotvec ("gripper_down" frame) -> flange rotation in the URDF base frame."""
    return _RX_PI @ Rot.from_rotvec(np.asarray(rotvec_gd, dtype=np.float64)).as_matrix()


def matrix_to_rotvec_gd(R_base: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(_RX_PI.T @ R_base).as_rotvec()


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 6-D -> SO(3) (columns a1, a2, a1 x a2).  For exact rotation
    columns this equals T-REX's ``get_rot_mat``; it is also robust to network
    outputs that are not exactly orthonormal."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def chunk_21d(state: np.ndarray, action: np.ndarray, t: int, chunk: int = ACTION_CHUNK) -> np.ndarray:
    """Build the [chunk, 21] delta-base chunk anchored at frame ``t``.

    state  : [N, 40] observation.state of one episode
    action : [N, 18] action of one episode
    The caller guarantees t + chunk <= N (no clamping: chunks never cross a
    primitive/episode boundary in this pipeline).
    """
    if t + chunk > len(action):
        raise IndexError(f"chunk [{t},{t + chunk}) exceeds episode length {len(action)}")
    p0 = state[t, STATE_EE_POS].astype(np.float64)
    R0 = rotvec_gd_to_matrix(state[t, STATE_EE_ROTVEC])
    a = action[t:t + chunk].astype(np.float64)
    Rk = _RX_PI[None] @ Rot.from_rotvec(a[:, ACTION_EE_ROTVEC]).as_matrix()     # [C,3,3]
    dp = (a[:, ACTION_EE_POS] - p0) @ R0                                        # == R0^T (p_k - p0)
    dR = np.einsum("ji,kjl->kil", R0, Rk)                                       # R0^T @ R_k
    out = np.concatenate([dp, dR[:, :, 0], dR[:, :, 1], a[:, ACTION_XHAND]], axis=1)
    assert out.shape == (chunk, ACTION_DIM)
    return out.astype(np.float32)


def reconstruct_chunk(chunk21: np.ndarray, ref_pos: np.ndarray, ref_rotvec_gd: np.ndarray) -> np.ndarray:
    """Inverse: [C,21] delta-base chunk + measured reference pose -> [C,18] absolute
    commands in the dataset's native convention (ee xyz, gripper_down rotvec, 12 joints).
    This is exactly what a deployment wrapper must do with the policy output."""
    chunk21 = np.asarray(chunk21, dtype=np.float64)
    R0 = rotvec_gd_to_matrix(ref_rotvec_gd)
    p = ref_pos[None] + chunk21[:, 0:3] @ R0.T                                  # p0 + R0 dp
    Rk = R0[None] @ rot6d_to_matrix(chunk21[:, 3:9])
    rv = Rot.from_matrix(_RX_PI.T[None] @ Rk).as_rotvec()
    return np.concatenate([p, rv, chunk21[:, 9:21]], axis=1)


def rotation_angle_error(rotvec_a: np.ndarray, rotvec_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between two batches of rotvecs (same frame)."""
    return (Rot.from_rotvec(rotvec_a).inv() * Rot.from_rotvec(rotvec_b)).magnitude()
