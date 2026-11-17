"""Shared fixtures for the Panda + XHand T-REX adaptation tests (A-G).

Paths can be overridden with env vars:
  TAPTIC_ROOT   merged LeRobot dataset with tactile_raw/ sidecar
  TAPTIC_DIR    derived dir from `python -m taptic.convert` (<out>/<task>)
  VQVAE_CKPT    trained XHand VQ-VAE checkpoint (latest.pt)
  TINY_QWEN     directory for the tiny Qwen3-VL fixture (built on demand)
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TASK = os.environ.get("TAPTIC_TASK", "insert_credit_card")
ROOT = os.path.expanduser(os.environ.get("TAPTIC_ROOT", "~/midtrain_merge/TAPTIC/midtrain_dataset"))
TDIR = os.path.expanduser(os.environ.get("TAPTIC_DIR", f"~/midtrain_merge/derived/trex_xhand_v1/{TASK}"))
VQ = os.path.expanduser(os.environ.get(
    "VQVAE_CKPT", "~/midtrain_merge/derived/vqvae_xhand/vqvae_xhand_f3_w16_k64_hand/latest.pt"))
TINY = os.path.expanduser(os.environ.get("TINY_QWEN", "~/midtrain_merge/derived/tiny_qwen3vl_test"))


def _need(path):
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    return path


@pytest.fixture(scope="session")
def src():
    from taptic.source import TapticSource
    return TapticSource(_need(ROOT), TASK)


@pytest.fixture(scope="session")
def tdir():
    return _need(TDIR)


@pytest.fixture(scope="session")
def samples(tdir):
    import pandas as pd
    return pd.read_parquet(os.path.join(tdir, "samples.parquet"))


@pytest.fixture(scope="session")
def vq_ckpt():
    return _need(VQ)


@pytest.fixture(scope="session")
def tiny_qwen():
    from tests.taptic.tiny_qwen3vl import TREX_META, build
    _need(os.path.join(TREX_META, "processor"))
    return build(TINY)


@pytest.fixture(scope="session")
def processor(tiny_qwen):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(tiny_qwen)


class _Acc:
    is_main_process = True

    @staticmethod
    def print(*a, **k):
        pass


@pytest.fixture(scope="session")
def train_cfg(tdir):
    return SimpleNamespace(
        taptic_dirs=[tdir], taptic_stats="", action_dim=21, action_chunk=16,
        image_size=[384, 288], use_flare=1, n_flare_steps=8, flare_frame_stride=4,
        tactile_delay_offsets=[0, 4, 8, 12], vqvae_window=16, use_tactile_vec=1,
        use_tactile_vqvae=1, use_tactile_deform=0, use_robot_state=0)


@pytest.fixture(scope="session")
def accel():
    return _Acc()

HUB = os.path.expanduser(os.environ.get("TAPTIC_HUB_ROOT", f"~/midtrain_merge/src/{TASK}"))


@pytest.fixture(scope="session")
def hub_root():
    return _need(os.path.join(HUB, "tactile_raw", "info.json")) and HUB
