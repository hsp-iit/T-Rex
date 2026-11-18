"""Small real-data CPU smoke suite; no training or model downloads.

TAPTIC_ROOT points to the verified LeRobot export. TAPTIC_PROCESSOR points to
local official Qwen3-VL processor files (no weights required).
"""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor

from qwen_vla.taptic_dataset import TRexTapticDataset, validate_config
from taptic.actions import chunk_21d, reconstruct_chunk, rotation_angle_error, rotvec_gd_to_matrix
from taptic.data import Source, fit_stats, tactile_at, tactile_stream_between

ROOT = os.environ.get('TAPTIC_ROOT')
PROCESSOR = os.environ.get('TAPTIC_PROCESSOR')
pytestmark = pytest.mark.skipif(not ROOT or not PROCESSOR, reason='set TAPTIC_ROOT and TAPTIC_PROCESSOR for real-data tests')


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(2)
    src = Source(ROOT)
    e = src.episodes
    # One native train episode, one missing-taxel train episode, one held-out episode.
    ids = [int(e[(e.suggested_split == 'train') & (~e.tactile_interpolated)].index[0]),
           int(e[(e.suggested_split == 'train') & e.tactile_interpolated].index[0]),
           int(e[e.suggested_split == 'val'].index[0])]
    stats = fit_stats(src, ids[:2])
    cfg = SimpleNamespace(lerobot_root=ROOT, action_dim=21, action_chunk=16,
        use_tactile_vec=1, use_tactile_deform=0, use_tactile_vqvae=0, use_tactile_code=0,
        use_robot_state=1, use_flare=1, n_flare_steps=2, flare_frame_stride=2,
        image_size=(56, 56), tactile_delay_offsets=[0], cascaded_total_steps=10, cascaded_split_step=6)
    proc = AutoProcessor.from_pretrained(PROCESSOR, local_files_only=True)
    ds = TRexTapticDataset(cfg, proc, SimpleNamespace(print=print), episode_ids=ids, _stats=stats)
    return src, ds, ids


def test_load_episode_and_causal_windows(setup):
    src, ds, ids = setup
    for ep in ids[:2]:
        idx = next(i for i, (e, t) in enumerate(ds._rows) if e == ep and t >= 1)
        x = ds[idx]
        m = x['meta']
        assert x['action_chunk'].shape == (16, 21)
        assert x['tactile_taxels'].shape == (5, 120, 3)
        assert m['tactile_available_ns'] <= m['observation_ns']
        assert np.all((np.diff(m['action_timestamps_ns']) > 0) & (np.diff(m['action_timestamps_ns']) <= 150_000_000))
        assert x['tactile_valid'].all() == (ep == ids[0])
        if ep == ids[1]:
            assert not x['tactile_taxels'].any()
        _, _, _, times, sm, available = ds._episode(ep)
        assert 'interpolated' not in sm or not sm.interpolated.any()
        assert x['state_raw'].shape == (21,)
        assert x['recorded_haptics']['force'].shape == (3,)
    val = ds.create_val_split()
    assert {e for e, _ in ds._rows}.isdisjoint({e for e, _ in val._rows})
    assert {e for e, _ in val._rows} == {ids[2]}
    assert ds.stats_data['taptic']['train_episodes'] == sorted(ids[:2])


def _check_finite(x):
    if torch.is_tensor(x):
        assert torch.isfinite(x).all()
    elif isinstance(x, dict):
        for v in x.values():
            _check_finite(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _check_finite(v)
    elif isinstance(x, np.ndarray):
        assert np.isfinite(x).all()


def test_batch_and_roundtrip(setup):
    src, ds, ids = setup
    other = next(i for i, (ep, _) in enumerate(ds._rows) if ep == ids[1])
    b = next(iter(DataLoader(torch.utils.data.Subset(ds, [0, other]), batch_size=2,
                            collate_fn=ds.collate_fn, num_workers=0)))
    required = ['input_ids', 'pixel_values', 'image_grid_thw', 'attention_mask', 'n_slow_images',
                'norm_actions', 'noisy_actions', 'target', 'timesteps', 'eps_r', 'time_r',
                'tactile_f6s_delayed', 'tactile_taxels', 'tactile_valid', 'recorded_haptics']
    assert all(k in b for k in required)
    for k in ['norm_actions', 'noisy_actions', 'target', 'eps_r']:
        assert b[k].shape == (2, 16, 21)
    assert b['tactile_f6s'].shape == (2, 5, 360)
    assert b['tactile_taxels'].shape == (2, 5, 120, 3)
    assert b['tactile_valid'][0].all() and not b['tactile_valid'][1].any()
    assert b['flare_grid_thw'].shape == (4, 3)
    assert b['contact_risk'] is b['contact_features'] is b['tactile_f6_history'] is None
    _check_finite(b)
    a = ds[1]['action_chunk']
    np.testing.assert_allclose(ds.denormalize_actions(ds.normalize_actions(a)), a, atol=3e-7)
    # Held-out/out-of-range values also round trip: normalization does not clip.
    np.testing.assert_allclose(ds.denormalize_actions(ds.normalize_actions(a + 4)), a + 4, atol=1e-6)
    s = np.stack(src.frames(ids[0])['observation.state'])
    raw = np.stack(src.frames(ids[0]).action)
    c = chunk_21d(s, raw, 2)
    rec = reconstruct_chunk(c, s[2, 34:37], s[2, 37:40])
    np.testing.assert_allclose(rec[:, :3], raw[2:18, :3], atol=1e-6)
    assert rotation_angle_error(rec[:, 3:6], raw[2:18, 3:6]).max() < 1e-6
    np.testing.assert_array_equal(c[:, 9:], raw[2:18, 6:])
    with pytest.raises(IndexError):
        chunk_21d(s, raw, len(raw) - 15)


def test_future_samples_and_missing_are_explicit():
    import pandas as pd
    sm = pd.DataFrame({'read_end_ns': [1, 3, 5], 'force': [np.ones(15)*i for i in [1, 2, 3]],
                       'values': [np.ones(1800)*i for i in [1, 2, 3]]})
    # Sample 3 was read at time 3, but unavailable until 7.
    avail = np.array([1, 7, 5])
    raw, force, valid, fv, ts = tactile_at(sm, avail, 4, 1, True)
    assert valid and fv and ts == 1 and np.all(raw == 1)
    raw, force, valid, fv, ts = tactile_at(sm, avail, 0, 0, True)
    assert not valid and not fv and not raw.any()


def test_native_stream_preserves_every_recorded_sample(setup):
    _, ds, ids = setup
    i = next(i for i, (ep, t) in enumerate(ds._rows) if ep == ids[0] and t == 1)
    item = ds[i]
    stream = item['tactile_stream']
    _, _, _, times, sm, available = ds._episode(ids[0])
    expected = sm[(available > times[0]) & (available <= times[1])]
    n = len(expected)
    assert n > 1  # native-rate observations were not reduced to one 10 Hz frame
    np.testing.assert_array_equal(stream['read_end_ns'], expected.read_end_ns)
    np.testing.assert_array_equal(stream['sdk_force'], np.stack(expected.force).reshape(n, 5, 3))
    np.testing.assert_array_equal(stream['taxels'], np.stack(expected['values']).reshape(n, 5, 120, 3))
    assert stream['taxel_valid'].all() and not stream['clock_is_nominal'].any()
    assert np.all(stream['available_ns'] <= item['meta']['observation_ns'])
    other = next(i for i, (ep, _) in enumerate(ds._rows) if ep == ids[1])
    legacy = ds[other]
    assert not legacy['tactile_stream']['taxel_valid'].any()
    assert legacy['tactile_stream']['clock_is_nominal'].all()
    b = ds.collate_fn([item, legacy])['tactile_stream']
    assert b['taxels'].shape == (2, n, 5, 120, 3)
    assert b['sdk_force'].shape == (2, n, 5, 3)
    assert b['lengths'].tolist() == [n, 1]
    assert b['sample_mask'].sum(1).tolist() == [n, 1]
    assert not b['taxel_valid'][1].any()
    _check_finite(b)
    # Empty intervals are represented explicitly, with zero valid measurements.
    empty = copy.copy(item)
    empty['tactile_stream'] = {k: v[:0] for k, v in stream.items()}
    batch = ds.collate_fn([empty, empty])['tactile_stream']
    assert batch['taxels'].shape == (2, 0, 5, 120, 3)
    assert batch['sample_mask'].shape == (2, 0)
    assert batch['lengths'].tolist() == [0, 0]


def test_native_stream_intervals_do_not_leak_or_resample():
    import pandas as pd
    sm = pd.DataFrame({'read_end_ns': [10, 20, 25, 30],
        'force': [np.full(15, i) for i in range(4)],
        'values': [np.full(1800, i) for i in range(4)],
        'sequence': [0, 1, 2, 3], 'reset_sequence': [0, 0, 1, 1],
        'read_errors': [0, 1, 1, 1], 'dropped': [0, 0, 0, 1],
        'interpolated': [False, False, True, False]})
    avail = np.array([10, 28, 25, 30], dtype=np.int64)
    empty = tactile_stream_between(sm, avail, 26, 10, True)
    assert len(empty['read_end_ns']) == 0  # 20 unavailable; 25 synthetic
    initial = tactile_stream_between(sm, avail, 10, None, True)
    assert initial['sequence'].tolist() == [0]  # only latest initial observation
    next_interval = tactile_stream_between(sm, avail, 30, 26, True)
    assert next_interval['sequence'].tolist() == [1, 3]
    assert next_interval['available_ns'].tolist() == [28, 30]
    assert next_interval['read_end_ns'].tolist() == [20, 30]
    assert next_interval['reset_sequence'].tolist() == [0, 1]
    assert next_interval['read_errors'].tolist() == [1, 1]
    assert next_interval['dropped'].tolist() == [0, 1]
    assert tactile_stream_between(sm, avail, 40, 30, True)['taxels'].shape == (0, 5, 120, 3)
    legacy = tactile_stream_between(sm, avail, 30, 26, False)
    assert not legacy['taxel_valid'].any() and not legacy['taxels'].any()
    np.testing.assert_array_equal(legacy['sdk_force'], next_interval['sdk_force'])
    with pytest.raises(ValueError):
        tactile_stream_between(sm, avail, 30, 30, True)


def test_reject_leaking_statistics_and_bad_configuration(setup):
    _, ds, ids = setup
    bad = copy.deepcopy(ds.stats_data)
    bad['taptic']['train_episodes'].append(ids[2])
    with pytest.raises(ValueError, match='non-training'):
        TRexTapticDataset(ds.config, ds.processor, ds.accelerator, episode_ids=ids, _stats=bad)
    for key, value in [('action_dim', 18), ('tactile_delay_offsets', [4]), ('use_tactile_vqvae', 1),
                       ('use_tactile_deform', 1), ('cascaded_split_step', 8)]:
        cfg = copy.copy(ds.config)
        setattr(cfg, key, value)
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_frame_convention_against_urdf(setup):
    src, _, ids = setup
    path = os.environ.get('TAPTIC_PANDA_URDF')
    if not path:
        pytest.skip('set TAPTIC_PANDA_URDF for independent frame verification')
    joints = {j.attrib['name']: j for j in ET.parse(path).getroot().findall('joint')}
    # URDF FK implemented directly to avoid requiring optional placo bindings.
    for ep in src.episodes.groupby('source_task').head(1).index:
        state = np.stack(src.frames(ep)['observation.state'])
        for s in state[::max(1, len(state)//10)]:
            T = np.eye(4)
            for i in range(1, 9):
                j = joints[f'panda_joint{i}']
                o = j.find('origin')
                O = np.eye(4)
                O[:3, 3] = np.fromstring(o.attrib['xyz'], sep=' ')
                O[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.attrib['rpy'], sep=' ')).as_matrix()
                Q = np.eye(4)
                if i <= 7:
                    axis = np.fromstring(j.find('axis').attrib['xyz'], sep=' ')
                    Q[:3, :3] = Rotation.from_rotvec(axis * np.deg2rad(s[i-1])).as_matrix()
                T = T @ O @ Q
            np.testing.assert_allclose(T[:3, 3], s[34:37], atol=1e-6)
            np.testing.assert_allclose(T[:3, :3], rotvec_gd_to_matrix(s[37:40]), atol=1e-6)


def test_original_loader_contract_unchanged():
    import subprocess
    for f in ['qwen_vla/lerobot_dataset.py', 'utils/lerobot_common.py',
              'qwen_vla/modeling_vla.py', 'qwen_vla/modeling_qwen3vl_mot.py']:
        assert Path(f).read_bytes() == subprocess.check_output(['git', 'show', f'HEAD:{f}'])
    from utils.lerobot_common import ACTION_DIM, ACTION_CHUNK, build_trex_features
    assert (ACTION_DIM, ACTION_CHUNK) == (62, 16)
    assert build_trex_features((3, 56, 56), include_tactile=False)['action']['shape'] == (16, 62)


@torch.no_grad()
@torch.autocast('cpu', dtype=torch.bfloat16)
def test_original_action_and_tactile_forwards(setup):
    from qwen_vla import Qwen3VLVLAModel, split_slow_fast_embeds
    from transformers import Qwen3VLForConditionalGeneration
    _, ds, _ = setup
    b = ds.collate_fn([ds[1]])
    cfg = AutoConfig.from_pretrained(PROCESSOR, local_files_only=True)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 1
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.rope_scaling = {'rope_type': 'default', 'mrope_section': [6, 5, 5], 'mrope_interleaved': True}
    v.hidden_size, v.intermediate_size, v.depth, v.num_heads = 32, 64, 1, 2
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0]
    cfg._attn_implementation = 'eager'
    base = Qwen3VLForConditionalGeneration(cfg).float().eval()
    m = Qwen3VLVLAModel(t, action_dim=21, action_chunk=16, tacf6_dim=360,
                        use_robot_state=True, tactile_intermediate_size=64).float().eval()
    m.initialize_vla_weights()
    m.visual = base.model.visual
    object.__setattr__(m, '_rope_index_fn', base.model.get_rope_index)
    emb = m.prepare_inputs_embeds(b['input_ids'], b['pixel_values'].float(), b['image_grid_thw'])
    g = b['image_grid_thw']
    nslow = int(g[0].prod() // 4)
    slow, fast = split_slow_fast_embeds(emb, b['input_ids'], m.image_token_id, nslow)
    pos, _ = m.get_rope_index(b['input_ids'], g, b['attention_mask'])
    pos = pos[:, :, :slow.shape[1]]
    state = m.state_embedder(b['state_raw'].float()).unsqueeze(1)
    x, cache, nact, tau = m.forward_flow_action_partial(slow, pos, b['eps_r'].float(),
        attention_mask=b['attention_mask'], fast_embeds=fast, state_embeds=state,
        num_steps_total=10, split_step=6, refresh_clean_kv=True)
    assert abs(tau - .4) < 1e-2 and x.shape == (1, 16, 21) and torch.isfinite(x).all()
    # Training-time tactile pass with full-resolution data through existing embedder.
    ts = b['time_r'].float() * .4
    xt = (1-ts[:, None, None])*b['norm_actions'].float() + ts[:, None, None]*b['eps_r'].float()
    velocity = m.tactile_flow_train_step(cache, pos, nact, xt, ts, tactile_f6=b['tactile_f6s_delayed'])
    assert velocity.shape == (1, 16, 21) and torch.isfinite(velocity).all()
    # Check training velocity on the original Action Expert path as well.
    full = torch.cat([slow, fast, state, m.t_embedder(b['timesteps'].float()).unsqueeze(1),
                      m.x_embedder(b['noisy_actions'].float())], 1)
    out = m.model(inputs_embeds=full, position_ids=pos, attention_mask=b['attention_mask'],
        use_cache=False, latent_indexes=torch.arange(slow.shape[1]),
        action_indexes=torch.arange(slow.shape[1], full.shape[1]), tactile_indexes=torch.arange(0))
    vel = m.final_layer(out.last_hidden_state[:, -16:])
    assert vel.shape == (1, 16, 21) and torch.isfinite(vel).all()


def test_official_loader_still_collates(setup, tmp_path):
    import json
    from qwen_vla.lerobot_dataset import TRexLeRobotDataset
    from utils.lerobot_common import build_trex_features, KEY_HEAD, KEY_ACTION, KEY_STATE
    _, taptic, _ = setup
    feats = build_trex_features((3, 56, 56), include_wrist=False, include_tactile=False)
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({'fps': 30, 'features': feats}))
    cfg = copy.copy(taptic.config)
    cfg.lerobot_root, cfg.action_dim, cfg.use_flare, cfg.use_tactile_vec = str(tmp_path), 62, 0, 0
    stat = {'mask': [True]*62, 'q01': [-1.]*62, 'q99': [1.]*62}
    original = TRexLeRobotDataset(cfg, taptic.processor, taptic.accelerator, _ds=[],
                                 _stats={'rlbench': {'action': stat, 'state': stat}})
    item = {KEY_HEAD: torch.zeros(1, 3, 56, 56), KEY_ACTION: torch.zeros(16, 62),
            KEY_STATE: torch.zeros(62), 'task': 'Move the hand.'}
    b = original.collate_fn([item, item])
    assert b['norm_actions'].shape == (2, 16, 62)
    assert b['tactile_f6s'] is None
    _check_finite(b)
