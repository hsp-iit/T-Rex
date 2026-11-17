"""Panda + RobotEra XHand constants for the TAPTIC -> T-REX adaptation.

Everything here was verified against the recorded data (see
docs/xhand/DATASET_AUDIT.md), not inferred from feature names.
"""
from __future__ import annotations

# ── Source dataset layout (TAPTIC LeRobot v3.0, robot_type "panda_ee") ────────
ACTION_SOURCE_DIM = 18          # ee.x,y,z | ee.wx,wy,wz | 12 XHand joint targets
STATE_SOURCE_DIM = 40

# observation.state slices (names in meta/info.json)
STATE_PANDA_JOINTS = slice(0, 7)      # degrees (verified: FK(deg) == recorded ee pose)
STATE_XHAND_JOINTS = slice(7, 19)     # rad, measured
STATE_XHAND_FORCE = slice(19, 34)     # SDK calc_force, raw counts, 5 tips x (x,y,z)
STATE_EE_POS = slice(34, 37)          # m, panda_link8 (flange) in URDF base frame
STATE_EE_ROTVEC = slice(37, 40)       # rad, rotvec(Rx(pi)^T R_base)  ("gripper_down")

# action slices
ACTION_EE_POS = slice(0, 3)           # commanded flange position, m, URDF base frame
ACTION_EE_ROTVEC = slice(3, 6)        # commanded rotvec, same "gripper_down" frame as state
ACTION_XHAND = slice(6, 18)           # absolute XHand joint targets, rad

XHAND_JOINTS = (
    "right_hand_thumb_bend_link", "right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2",
    "right_hand_index_bend_link", "right_hand_index_rota_link1", "right_hand_index_rota_link2",
    "right_hand_mid_link1", "right_hand_mid_link2",
    "right_hand_ring_link1", "right_hand_ring_link2",
    "right_hand_pinky_link1", "right_hand_pinky_link2",
)

# ── T-REX adapted dims ───────────────────────────────────────────────────────
ACTION_DIM = 21                 # [dp(3), rot6d(6), xhand q(12)]
ACTION_CHUNK = 16
N_FINGERS = 5
TAC_DIM = 3                     # (Fx, Fy, Fz) per fingertip, SDK calc_force
TAC_FLAT = N_FINGERS * TAC_DIM  # 15
FINGERS = ("thumb", "index", "middle", "ring", "little")   # sidecar `force` order
VQVAE_WINDOW = 16               # tactile samples (~209 ms at the measured 76.4 Hz)

# Orientation reference used by the recorder (lerobot_robot_panda ee_safety.py):
# recorded rotvec r satisfies R_base = Rx(pi) @ Rot(r).
GRIPPER_DOWN_ROTVEC = (3.141592653589793, 0.0, 0.0)

# Delta-base reference: T-REX gen_json uses the *measured* pose at the chunk
# start (`*_arm_current_pose[i]`) and the *commanded* targets at i+k.
NOMINAL_FPS = 10
MAX_FRAME_GAP_S = 0.15          # real-time gap treated as a temporal discontinuity
