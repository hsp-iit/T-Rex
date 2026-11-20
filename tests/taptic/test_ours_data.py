"""Real TAPTIC episodes: delayed fast-tick alignment, no future leakage, slow/fast
risk timestamps, and the actual trainer step (cascaded_losses) on real batches.

Needs TAPTIC_ROOT (verified LeRobot export) and TAPTIC_PROCESSOR (local
Qwen3-VL processor/config; random tiny weights are built, nothing downloaded).
Set TAPTIC_OURS_EVIDENCE=<path.json> to also write the alignment evidence.
"""
import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
ROOT, PROCESSOR = os.environ.get('TAPTIC_ROOT'), os.environ.get('TAPTIC_PROCESSOR')
pytestmark = pytest.mark.skipif(not ROOT or not PROCESSOR, reason='set TAPTIC_ROOT and TAPTIC_PROCESSOR')
BASE = '05c932853ac5984ac7128561de2583c34b568262'
MS = 1_000_000
DELAYS = (0, 50, 100, 150)


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    from transformers import AutoProcessor
    from taptic.data import Source, fit_stats
    from taptic.risk import RiskConfig, fit_risk_stats
    from tactile_vqvae.data.taptic import fit_force_stats, load_tracks
    from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig
    from qwen_vla.taptic_dataset import TRexTapticDataset
    torch.set_num_threads(2)
    tmp = tmp_path_factory.mktemp('ours_real')
    src = Source(ROOT)
    e = src.episodes
    nat = ~e.tactile_interpolated.astype(bool)
    by_task = e[(e.suggested_split == 'train') & nat].groupby('source_task')
    natives = [int(g.index[0]) for _, g in by_task][:3]
    legacy = int(e[(e.suggested_split == 'train') & ~nat].index[0])
    val = int(e[(e.suggested_split == 'val') & nat].index[0])
    ids = natives + [legacy, val]
    train = natives + [legacy]
    stats = fit_stats(src, train)
    vq_cfg = TactileVQVAEConfig(window=16, per_finger_dim=3, n_hands=1, in_channels=15, codebook_size=16,
                                hidden_channels=16, bottleneck_channels=32, embed_dim=16, granularity='finger',
                                input_signal='taptic_sdk_force')
    fstats = fit_force_stats(load_tracks(src, natives, log=None))
    torch.manual_seed(0)
    torch.save({'config': vq_cfg.to_dict(), 'stats': fstats.to_dict(), 'model_state': TactileVQVAE(vq_cfg).state_dict(),
                'data': {'input_signal': 'taptic_sdk_force', 'source_identity': src.identity,
                         'train_episodes': natives, 'window': 16}}, tmp / 'vq.pt')
    rstats = fit_risk_stats(src, train, RiskConfig(), log=None)
    (tmp / 'risk.json').write_text(json.dumps(rstats.to_dict()))
    cfg = SimpleNamespace(lerobot_root=ROOT, action_dim=21, action_chunk=16, use_tactile_vec=1, use_tactile_deform=0,
                          use_tactile_vqvae=1, use_tactile_code=1, vqvae_window=16, vqvae_ckpt=str(tmp / 'vq.pt'),
                          taptic_tactile_input='sdk_force', use_robot_state=0, use_flare=1, n_flare_steps=2,
                          flare_frame_stride=2, image_size=(56, 56), tactile_delay_offsets=[0],
                          tactile_delay_ms=list(DELAYS), risk_stats=str(tmp / 'risk.json'), use_risk_tokens=1,
                          cascaded_total_steps=10, cascaded_split_step=6, cascaded_split_mode='multi',
                          cascaded_splits=[.2, .4, .6], flow_time_precision='fp32', tactile_stale_ms=50.,
                          tactile_gap_ms=50., cascaded_tactile_dropout=0.)
    proc = AutoProcessor.from_pretrained(PROCESSOR, local_files_only=True)
    ds = TRexTapticDataset(cfg, proc, SimpleNamespace(print=print), episode_ids=ids, _stats=stats)
    return SimpleNamespace(src=src, ds=ds, cfg=cfg, natives=natives, legacy=legacy, val=val, tmp=tmp,
                           rstats=rstats, vq_cfg=vq_cfg, proc=proc, stats=stats, ids=ids)


def _item(ds, idx, delay_ms):
    old = ds.delays_ns
    ds.delays_ns = (delay_ms * MS,)
    try:
        return ds[idx]
    finally:
        ds.delays_ns = old


def test_delayed_alignment_and_no_future_leakage_on_real_episodes(env):
    from taptic.force_history import history_at
    from taptic.risk import risk_features, features_at
    ds = env.ds
    ds._img = lambda *a: np.zeros((56, 56, 3), np.uint8)        # images are irrelevant here
    evidence = {'episodes': {}, 'delays_ms': list(DELAYS)}
    rows = {}
    for i, (ep, t) in enumerate(ds._rows):
        rows.setdefault(ep, []).append(i)
    for ep in env.natives + [env.legacy]:
        idxs = rows[ep][::max(1, len(rows[ep]) // 12)][:12]
        native = ep != env.legacy
        stats_ep = {d: {'n': 0, 'tactile_valid': 0, 'supervised_slots': [], 'age_ms': [], 'new_samples': [],
                        'risk_F_valid': 0, 'risk_S_valid': 0} for d in DELAYS}
        for idx in idxs:
            _, _, _, times, _, _ = ds._episode(ep)
            track = ds._cache_force
            items = {d: _item(ds, idx, d) for d in DELAYS}
            for d, x in items.items():
                m = x['meta']
                S, Fns = m['slow_ns'], m['fast_ns']
                t = m['frame']
                assert S == int(times[t]) and Fns == S + d * MS and m['fast_tick_aligned']
                assert np.array_equal(x['tactile_slot_mask'], times[t:t + 16] >= Fns)
                assert x['tactile_slot_mask'].all() == (d == 0) or times[t] >= Fns
                assert np.array_equal(m['action_timestamps_ns'], times[t:t + 16])   # targets not shifted
                st = stats_ep[d]
                st['n'] += 1
                st['supervised_slots'].append(int(x['tactile_slot_mask'].sum()))
                st['risk_S_valid'] += x['risk_S']['risk_valid']
                st['risk_F_valid'] += x['risk_F']['risk_valid']
                if not native:
                    assert not x['force_valid'] and not x['force_history'].any()
                    assert not x['risk_S']['risk_valid'] and not x['risk_F']['risk_valid']
                    continue
                ref, valid, n_real, age = history_at(track, Fns, 16, 50 * MS)
                assert np.array_equal(x['force_history'], ref) and x['force_valid'] == valid
                if valid:
                    end = int(np.flatnonzero((track['read_ns'] <= Fns) & (track['available_ns'] <= Fns))[-1])
                    assert np.array_equal(x['force_history'][-1], track['force'][end])
                    assert track['read_ns'][end] <= Fns and track['available_ns'][end] <= Fns
                    np.testing.assert_allclose(x['tactile_f6'], ds.force_stats.normalize(track['force'][end]), atol=0)
                    st['tactile_valid'] += 1
                    st['age_ms'].append(age / 1e6)
                    st['new_samples'].append(int(((track['available_ns'] > S) & (track['available_ns'] <= Fns)).sum()))
                feats = risk_features(track, np.array([S, Fns]), ds.risk_stats, ds.risk_cfg)
                assert x['risk_S'] == features_at(feats, 0) and x['risk_F'] == features_at(feats, 1)
            if native:
                assert items[0]['risk_S'] == items[0]['risk_F']        # F = S
                assert all(items[d]['risk_S'] == items[0]['risk_S'] for d in DELAYS)   # r_S fixed by S
                if items[150]['force_valid'] and items[0]['force_valid']:
                    # The window ends at a strictly later native read at F than at S
                    # (values may coincide: repeated/zero payloads are valid readings).
                    latest = lambda tick: int(np.flatnonzero((track['read_ns'] <= tick)
                                                             & (track['available_ns'] <= tick))[-1])
                    sF, sS = latest(items[150]['meta']['fast_ns']), latest(items[0]['meta']['slow_ns'])
                    assert sF > sS and track['read_ns'][sF] > items[0]['meta']['slow_ns']
                    assert items[150]['risk_F']['force_age_ns'] == items[150]['meta']['fast_ns'] - track['read_ns'][sF]
                # Leakage: perturb every sample available AFTER F -> nothing at F changes;
                # perturb samples in (S, F] -> r_S unchanged.
                x150 = items[150]
                Fns, S = x150['meta']['fast_ns'], x150['meta']['slow_ns']
                saved = copy.deepcopy({k: track[k] for k in ('force', 'joints')})
                try:
                    later = track['available_ns'] > Fns
                    track['force'][later] = 4321.
                    track['joints'][later] = 9.
                    track.pop('_risk_derived', None)
                    y = _item(ds, idxs[idxs.index(idx)], 150)
                    assert np.array_equal(y['force_history'], x150['force_history'])
                    assert y['risk_F'] == x150['risk_F'] and y['risk_S'] == x150['risk_S']
                    between = (track['available_ns'] > S) & (track['available_ns'] <= Fns)
                    track['force'][between] = 1234.
                    track.pop('_risk_derived', None)
                    z = _item(ds, idxs[idxs.index(idx)], 150)
                    assert z['risk_S'] == x150['risk_S']
                finally:
                    track['force'][:], track['joints'][:] = saved['force'], saved['joints']
                    track.pop('_risk_derived', None)
        evidence['episodes'][str(ep)] = {
            'task': str(env.src.episodes.loc[ep, 'source_task']), 'native': native,
            'per_delay': {str(d): {'samples': v['n'], 'tactile_valid': v['tactile_valid'],
                                   'risk_S_valid': int(v['risk_S_valid']), 'risk_F_valid': int(v['risk_F_valid']),
                                   'supervised_slots_mean': float(np.mean(v['supervised_slots'])),
                                   'tactile_age_ms_max': float(np.max(v['age_ms'])) if v['age_ms'] else None,
                                   'new_native_reads_after_S_mean': float(np.mean(v['new_samples']))
                                   if v['new_samples'] else None} for d, v in stats_ep.items()}}
        for d in DELAYS:
            if native:
                assert stats_ep[d]['tactile_valid'] == stats_ep[d]['n']    # native data stay fresh at F
                assert max(stats_ep[d]['age_ms']) <= 50
                if d >= 50:
                    assert min(stats_ep[d]['new_samples']) >= 2         # ~76 Hz: >= 2 reads per 50 ms
    out = os.environ.get('TAPTIC_OURS_EVIDENCE')
    if out:
        Path(out).write_text(json.dumps(evidence, indent=1))
    print('ALIGNMENT_EVIDENCE', json.dumps(evidence)[:2000])


def test_collate_contract_and_deterministic_validation_delays(env):
    ds = env.ds
    ds._img = lambda *a: np.zeros((56, 56, 3), np.uint8)
    idx = [i for i, (e, _) in enumerate(ds._rows) if e == env.natives[0]][:3] + \
          [next(i for i, (e, _) in enumerate(ds._rows) if e == env.legacy)]
    b = ds.collate_fn([ds[i] for i in idx])
    assert b['risk_S'].shape == b['risk_F'].shape == (4,) and b['risk_S'].dtype == torch.float32
    assert b['risk_S_valid'].dtype == torch.bool and not b['risk_S_valid'][-1] and not b['risk_F_valid'][-1]
    assert b['tactile_slot_mask'].shape == (4, 16) and b['tactile_slot_mask'].dtype == torch.bool
    assert set(b['risk_features_F']) >= {'M', 'R', 'V', 'M_valid', 'R_valid', 'V_valid', 'force_age_ns', 'magnitude'}
    assert b['tactile_delay_ns'].shape == (4,) and set(b['tactile_delay_ns'].tolist()) <= {d * MS for d in DELAYS}
    assert b['tactile_f6_history'].shape == (4, 16, 5, 3) and not b['tactile_loss_mask'][-1]
    val = ds.create_val_split()
    val._img = ds._img
    d1 = [val[i]['meta']['tactile_delay_ns'] for i in range(8)]
    d2 = [val[i]['meta']['tactile_delay_ns'] for i in range(8)]
    assert d1 == d2 == [DELAYS[i % 4] * MS for i in range(8)]
    # Baseline configuration (no risk stats, delays [0]) keeps the original batch keys/values.
    cfg0 = SimpleNamespace(**{**vars(env.cfg), 'tactile_delay_ms': [0], 'risk_stats': '', 'use_risk_tokens': 0})
    from qwen_vla.taptic_dataset import TRexTapticDataset
    ds0 = TRexTapticDataset(cfg0, env.proc, SimpleNamespace(print=print), episode_ids=env.ids, _stats=env.stats)
    ds0._img = ds._img
    b0 = ds0.collate_fn([ds0[i] for i in idx])
    assert b0['tactile_slot_mask'] is None and b0['risk_S'] is None and b0['risk_F'] is None
    x0, x = ds0[idx[0]], _item(ds, idx[0], 0)
    assert np.array_equal(x0['force_history'], x['force_history']) and np.array_equal(x0['tactile_f6'], x['tactile_f6'])


def make_model(env, risk=True, seed=7):
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from qwen_vla import Qwen3VLVLAModel
    from midtrain import reinit_tactile_expert
    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(PROCESSOR, local_files_only=True)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.rope_scaling = {'rope_type': 'default', 'mrope_section': [6, 5, 5], 'mrope_interleaved': True}
    v.hidden_size, v.intermediate_size, v.depth, v.num_heads = 32, 64, 1, 2
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0]
    cfg._attn_implementation = 'eager'
    base = Qwen3VLForConditionalGeneration(cfg).float().eval()
    blob = torch.load(env.tmp / 'vq.pt', weights_only=False)
    m = Qwen3VLVLAModel(t, action_dim=21, action_chunk=16, tacf6_dim=3, tactile_intermediate_size=64,
                        n_flare_tokens_per_frame=2, n_flare_steps=2, use_tactile_code=True, vqvae_codebook_size=16,
                        use_tactile_vqvae=True, vqvae_config=blob['config'], use_risk_tokens=risk).float()
    m.initialize_vla_weights()
    m.load_tactile_vqvae(blob)
    reinit_tactile_expert(m)
    torch.nn.init.normal_(m.final_layer.mlp.fc2.weight, std=.05)
    m.visual = base.model.visual
    m.visual.requires_grad_(False)
    object.__setattr__(m, '_rope_index_fn', base.model.get_rope_index)
    return m.eval()


def _batch(env, delays=DELAYS, n=2):
    ds = env.ds
    ds.__dict__.pop('_img', None)
    i0 = next(i for i, (e, t) in enumerate(ds._rows) if e == env.natives[0] and t == 20)
    i1 = next(i for i, (e, t) in enumerate(ds._rows) if e == env.natives[1] and t == 20)
    items = [_item(ds, i0, delays[-1]), _item(ds, i1, delays[1 if len(delays) > 1 else 0])][:n]
    return ds.collate_fn(items)


@pytest.mark.parametrize('tau', [.2, .4, .6])
def test_actual_trainer_step_on_real_batches(env, tau):
    import midtrain as trainer
    from qwen_vla.flow_matching import Handover
    m = make_model(env)
    b = _batch(env)
    assert b['tactile_slot_mask'][0].sum() < 16 and b['tactile_loss_mask'].all()
    args = SimpleNamespace(action_chunk=16, use_robot_state=0, use_tactile_vec=1, use_tactile_deform=0,
                           cascaded_total_steps=10, flow_time_precision='fp32', cascaded_tactile_dropout=0.)
    h = Handover(10 - round(tau * 10))
    m.train()
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
        torch.manual_seed(5)
        res = trainer.cascaded_losses(m, m, b, args, h, is_stage1=False, use_flare=True, K=4, training=True)
    assert res['n_risk'] == 1 and torch.isfinite(res['loss_act']) and torch.isfinite(res['loss_tac'])
    (res['loss_act'] + res['loss_tac']).backward()
    g = {n: p.grad.abs().sum().item() for n, p in m.named_parameters() if p.grad is not None}
    for name in ('risk_embedder_action.mlp.0.weight', 'risk_embedder_tactile.mlp.0.weight',
                 'tactile_code_embedder.weight', 'tacf6_embedder.mlp.fc1.weight', 'final_layer.mlp.fc2.weight',
                 'final_layer_tactile.mlp.fc2.weight', 'flare_queries'):
        assert g.get(name, 0) > 0, name
    assert not any(n.startswith(('visual.', 'tactile_vqvae.')) for n in g)
    # The slot mask removes exactly the elapsed slots from the tactile loss.
    v = torch.randn(2, 16, 21, requires_grad=True)
    tgt = torch.randn(2, 16, 21)
    loss = trainer.tactile_flow_loss(v, tgt, b['tactile_loss_mask'], b['tactile_slot_mask'])
    loss.backward()
    assert (v.grad.abs().sum(-1) > 0).eq(b['tactile_slot_mask']).all()
    ref = ((v - tgt) ** 2).mean(-1)[b['tactile_slot_mask']].mean()
    assert torch.allclose(loss, ref)
    print('TRAIN_STEP_RESULT', json.dumps({'tau': tau, 'loss_act': float(res['loss_act'].detach()),
                                           'loss_tac': float(res['loss_tac'].detach()),
                                           'supervised_slots': b['tactile_slot_mask'].sum(1).tolist(),
                                           'risk_S': b['risk_S'].tolist(), 'risk_F': b['risk_F'].tolist()}))


def _base_trainer():
    src = subprocess.check_output(['git', 'show', f'{BASE}:scripts/midtrain.py'], text=True, cwd=REPO)
    mod = types.ModuleType('base_midtrain')
    mod.__file__ = str(REPO / 'scripts' / 'midtrain.py')
    exec(compile(src, '<base:midtrain>', 'exec'), mod.__dict__)
    return mod


@torch.no_grad()
@pytest.mark.parametrize('mode', ['fixed', 'multi'])
def test_vanilla_validation_loop_bitwise_equal_to_base_commit(env, mode, monkeypatch):
    """Risk-free model + delays [0]: the refactored loop == the base commit's loop."""
    import midtrain as trainer
    base = _base_trainer()
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    cfg0 = SimpleNamespace(**{**vars(env.cfg), 'tactile_delay_ms': [0], 'risk_stats': '', 'use_risk_tokens': 0})
    from qwen_vla.taptic_dataset import TRexTapticDataset
    ds0 = TRexTapticDataset(cfg0, env.proc, SimpleNamespace(print=print), episode_ids=env.ids, _stats=env.stats)
    i0 = next(i for i, (e, t) in enumerate(ds0._rows) if e == env.natives[0] and t == 20)
    rows2 = [i for i, (e, t) in enumerate(ds0._rows) if e == env.natives[2]]
    i1 = rows2[len(rows2) // 2]
    b = ds0.collate_fn([ds0[i0], ds0[i1]])
    m = make_model(env, risk=False)
    args = SimpleNamespace(cascaded_split_mode=mode, cascaded_total_steps=10, cascaded_split_step=6,
                           cascaded_splits=[.2, .4, .6], flow_time_precision='fp32', max_val_batches=1,
                           use_robot_state=0, use_tactile_vec=1, use_tactile_deform=0, action_chunk=16)
    accel = SimpleNamespace(unwrap_model=lambda x: x)
    out = []
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
        for fn in (base.run_validation, trainer.run_validation):
            torch.manual_seed(11)
            out.append(fn(m, [b], accel, args, False, True, 4, 2, -1))
    assert out[0] == out[1] and all(np.isfinite(v) for v in out[0].values())
    print('VANILLA_REGRESSION', mode, json.dumps(out[0]))
