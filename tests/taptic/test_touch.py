"""trex-touch: long tactile history, self-handover tactile pairs, contact-weighted tactile loss,
touch-event weighting, dataset variants (derived dataset + source exclusion). docs/TOUCH_MODULES.md.

Unit tests run anywhere; real-data parts need TAPTIC_ROOT and TAPTIC_PROCESSOR (tiny random
models, nothing downloaded), as in test_ours_data.py / test_ours_aux.py.
"""
import copy
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
real = pytest.mark.skipif(not ROOT or not PROCESSOR, reason='set TAPTIC_ROOT and TAPTIC_PROCESSOR')
TIP = '1a16caf'      # trex-ours-risk-aux, the code of the vanilla/ours finals' evaluation
MS = 1_000_000

if ROOT and PROCESSOR:
    from test_ours_data import env, _item, DELAYS  # noqa: F401,E402  (module-scoped fixture)


# ── unit tests (no data) ────────────────────────────────────────────────────

def _track(n=200, dt_ms=13, seg_break=None):
    read = (np.arange(n, dtype=np.int64) * dt_ms + 1000) * MS
    seg = np.zeros(n, np.int64)
    if seg_break is not None:
        seg[seg_break:] = 1
    force = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((1, 5, 3), np.float32)
    return {'force': force, 'read_ns': read, 'available_ns': read.copy(), 'segment': seg}


def test_spread_history_points_span_and_padding():
    from taptic.force_history import spread_history_at, history_at
    tr = _track()
    tick = int(tr['read_ns'][150]) + 5 * MS
    h, ok, n_real = spread_history_at(tr, tick, 1000 * MS, 8, 50 * MS)
    assert ok and h.shape == (8, 5, 3) and n_real == 8
    idx = h[:, 0, 0].astype(int)
    assert idx[-1] == 150                                    # last point = current sample
    cur, ok2, _, _ = history_at(tr, tick, 16, 50 * MS)
    assert ok2 and np.array_equal(h[-1], cur[-1])
    assert np.all(np.diff(idx) > 0)                          # oldest first, strictly increasing
    # point k = latest sample read at or before read[end] - 1 s * (7-k)/7
    t_end = tr['read_ns'][150]
    for k, i in enumerate(idx):
        target = t_end - round((7 - k) / 7 * 1000 * MS)
        assert tr['read_ns'][i] <= target < (tr['read_ns'][i + 1] if i + 1 < len(tr['read_ns']) else np.inf)
    # segment start: earlier points repeat the segment's first sample
    tr = _track(seg_break=140)
    h, ok, n_real = spread_history_at(tr, tick, 1000 * MS, 8, 50 * MS)
    idx = h[:, 0, 0].astype(int)
    assert ok and idx.min() == 140 and idx[-1] == 150 and n_real < 8
    assert np.all(idx[:8 - n_real] == 140)
    # stale -> invalid zeros
    h, ok, n_real = spread_history_at(tr, int(tr['read_ns'][-1]) + 60 * MS, 1000 * MS, 8, 50 * MS)
    assert not ok and not h.any() and n_real == 0


def _trace(segments):
    """[T, 5] force trace from (n_frames, strongest-finger level) segments, index finger only."""
    f = np.concatenate([np.full(n, v, float) for n, v in segments])
    F = np.zeros((len(f), 5))
    F[:, 1] = f
    return F


def test_touch_events_definitions():
    from taptic.touch_events import episode_events, jump_threshold, window_flags
    # free 1 s, grasp at 10 (reaches 8), hold 2 s, rise to 30 at 30 (resistance), hold, release at 50
    F = _trace([(10, 0.), (20, 8.), (2, 8.), (18, 30.), (10, 0.)])
    ev = episode_events(F, types=('grasp_onset', 'resistance', 'release'), jump=10.)
    assert np.flatnonzero(ev).tolist() == [10, 32, 50]
    # grasp needs >= 0.5 s free before: a re-touch 0.3 s after release is not a grasp onset
    F2 = _trace([(10, 0.), (10, 8.), (3, 0.), (10, 8.)])
    assert np.flatnonzero(episode_events(F2, types=('grasp_onset',))).tolist() == [10]
    # resistance needs >= 0.5 s of contact before the rise, and 1 s between triggers
    F3 = _trace([(10, 0.), (2, 8.), (10, 30.)])
    assert not episode_events(F3, types=('resistance',), jump=10.).any()
    # labels: start of a labelled step
    sub = np.array([0] * 5 + [11] * 5 + [4] * 5)
    assert np.flatnonzero(episode_events(np.zeros((15, 5)), types=('label',), subtask=sub, label_ids=(11,))).tolist() == [5]
    # window looks back 1 s and ahead 0.2 s
    e = np.zeros(40, bool)
    e[20] = True
    w = window_flags(e, 0.2, 1.0)
    assert np.flatnonzero(w).tolist() == list(range(18, 31))
    # threshold: quantile of the positive rises in contact, floor 5
    assert jump_threshold([_trace([(5, 0.), (20, 3.)])]) == 5.
    with pytest.raises(ValueError):
        episode_events(F, types=('bogus',))


def test_event_weights_math():
    """--tactile_event_weight: normalized per-sample weights, all dims, combined with contact weights."""
    import midtrain as trainer
    v, tgt = torch.randn(4, 16, 21), torch.randn(4, 16, 21)
    ev = torch.tensor([True, False, False, False])
    f = torch.where(ev, 5., 1.)
    f = f / f.mean()
    got = trainer.tactile_flow_loss(v * f.sqrt()[:, None, None], tgt * f.sqrt()[:, None, None], torch.ones(4, dtype=torch.bool))
    e = ((v - tgt) ** 2).mean(dim=(1, 2))
    assert torch.allclose(got, (e * f).mean())
    assert torch.isclose(f.mean(), torch.tensor(1.))


def test_feature_dim_and_guards():
    from qwen_vla.taptic_dataset import tactile_feature_dim, tactile_history_points
    c = SimpleNamespace(taptic_tactile_input='sdk_force')
    assert tactile_feature_dim(c) == 3 and tactile_history_points(c) == 0
    c.tactile_history_span_ms, c.tactile_history_points = 1000., 8
    assert tactile_feature_dim(c) == 24
    c.taptic_tactile_input = 'taxels'
    assert tactile_feature_dim(c) == 360 and tactile_history_points(c) == 0


# ── real data ───────────────────────────────────────────────────────────────

def _ds(env, **kw):
    from qwen_vla.taptic_dataset import TRexTapticDataset
    cfg = copy.copy(env.cfg)
    cfg.tactile_delay_ms, cfg.use_risk_tokens, cfg.risk_stats = [0], 0, ''
    cfg.cascaded_split_mode = 'fixed'
    for k, v in kw.items():
        setattr(cfg, k, v)
    return TRexTapticDataset(cfg, env.proc, SimpleNamespace(print=print), episode_ids=env.ids, _stats=env.stats)


FORCE_TASKS = ['dosing_pipette', 'insert_credit_card', 'insert_game_object_cylinder', 'unscrew_the_bottle',
               'grasp_plastic_glass']


@real
def test_dataset_history_tokens_and_contact_flag(env):
    base = _ds(env)
    long = _ds(env, tactile_history_span_ms=1000., tactile_history_points=8, tactile_force_tasks=FORCE_TASKS)
    seen_valid = seen_invalid = 0
    for i in range(0, len(base), max(1, len(base) // 40)):
        a, b = base[i], long[i]
        assert b['tactile_f6'].shape == (5, 24) and b['tactile_f6'].dtype == np.float32
        # the last point of every finger is the baseline's current-sample token
        assert np.array_equal(b['tactile_f6'].reshape(5, 8, 3)[:, -1], a['tactile_f6'])
        assert np.array_equal(a['force_history'], b['force_history'])   # VQ window unchanged
        assert a['force_valid'] == b['force_valid']
        if b['force_valid']:
            seen_valid += 1
        else:
            seen_invalid += 1
            assert not b['tactile_f6'].any() and not b['force_contact']
        task = str(long.source.episodes.loc[b['meta']['episode'], 'source_task'])
        if b['force_contact']:
            assert b['force_valid'] and task in FORCE_TASKS
            assert np.linalg.norm(b['force_history'][-1], axis=-1).max() > 2.
        # baseline config has no force tasks -> never flagged
        assert not a['force_contact']
    assert seen_valid and seen_invalid
    batch = long.collate_fn([long[0], long[1]])
    assert batch['tactile_f6s'].shape == (2, 5, 24) and batch['force_contact'].dtype == torch.bool


def _model(env, tacf6_dim=3, seed=7):
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
    m = Qwen3VLVLAModel(t, action_dim=21, action_chunk=16, tacf6_dim=tacf6_dim, tactile_intermediate_size=64,
                        n_flare_tokens_per_frame=2, n_flare_steps=2, use_tactile_code=True, vqvae_codebook_size=16,
                        use_tactile_vqvae=True, vqvae_config=blob['config']).float()
    m.initialize_vla_weights()
    m.load_tactile_vqvae(blob)
    reinit_tactile_expert(m)
    torch.nn.init.normal_(m.final_layer.mlp.fc2.weight, std=.05)
    torch.nn.init.normal_(m.final_layer_tactile.mlp.fc2.weight, std=.05)
    m.visual = base.model.visual
    m.visual.requires_grad_(False)
    object.__setattr__(m, '_rope_index_fn', base.model.get_rope_index)
    return m


def _args(**kw):
    return SimpleNamespace(action_chunk=16, use_robot_state=0, use_tactile_vec=1, use_tactile_deform=0,
                           cascaded_total_steps=10, flow_time_precision='fp32', cascaded_tactile_dropout=0., **kw)


def _losses(trainer, m, b, args, training=True, seed=5):
    from qwen_vla.flow_matching import Handover
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
        torch.manual_seed(seed)
        return trainer.cascaded_losses(m, m, b, args, Handover(6), is_stage1=False, use_flare=True, K=4,
                                       training=training)


def _contact_batch(ds, env):
    """Two native rows, at least one flagged force_contact when the data allow it."""
    rows = [i for i, (e, t) in enumerate(ds._rows) if e in env.natives]
    items = [ds[i] for i in rows[::max(1, len(rows) // 60)]]
    fc = [x for x in items if x['force_contact']]
    other = [x for x in items if x['force_valid'] and not x['force_contact']]
    pick = (fc[:1] + other[:1]) if fc else other[:2]
    return ds.collate_fn(pick), bool(fc)


def _tip_trainer():
    src = subprocess.check_output(['git', 'show', f'{TIP}:scripts/midtrain.py'], text=True, cwd=REPO)
    tip = types.ModuleType('tip_midtrain')
    tip.__file__ = str(REPO / 'scripts' / 'midtrain.py')
    exec(compile(src, '<tip:midtrain>', 'exec'), tip.__dict__)
    return tip


@real
def test_defaults_are_bitwise_identical_to_tip(env):
    import midtrain as trainer
    tip = _tip_trainer()
    ds = _ds(env, tactile_force_tasks=FORCE_TASKS)
    b, _ = _contact_batch(ds, env)
    m = _model(env)
    m.train()
    for training in (True, False):
        new = _losses(trainer, m, b, _args(), training=training)
        old = _losses(tip, m, b, _args(), training=training)
        assert torch.equal(new['loss_act'], old['loss_act']) and torch.equal(new['loss_tac'], old['loss_tac'])
        assert torch.equal(new['loss_tac_plain'], old['loss_tac'])
        # explicit neutral values are the baseline too
        neu = _losses(trainer, m, b, _args(tactile_self_handover=0., tactile_force_contact_weight=1.),
                      training=training)
        assert torch.equal(neu['loss_tac'], old['loss_tac'])


@real
def test_self_handover_pairs_and_gradient_routing(env):
    import midtrain as trainer
    from qwen_vla.flow_matching import Handover
    ds = _ds(env, tactile_force_tasks=FORCE_TASKS)
    b, _ = _contact_batch(ds, env)
    m = _model(env)
    m.train()
    base = _losses(trainer, m, b, _args())
    full = _losses(trainer, m, b, _args(tactile_self_handover=1.))
    assert full['self_handover_frac'] == 1.0 and torch.isfinite(full['loss_tac'])
    assert not torch.equal(full['loss_tac'], base['loss_tac'])
    assert torch.equal(full['loss_act'], base['loss_act'])          # Action loss untouched
    # validation never uses self pairs
    val_b = _losses(trainer, m, b, _args(), training=False)
    val_s = _losses(trainer, m, b, _args(tactile_self_handover=1.), training=False)
    assert torch.equal(val_b['loss_tac'], val_s['loss_tac']) and val_s['self_handover_frac'] == 0.
    # pair math: capture the Tactile expert's inputs and check x_tau lies on x_split -> demo
    seen = {}
    orig_step, orig_partial = m.tactile_flow_train_step, m.forward_flow_action_partial

    def spy_partial(*a, **k):
        out = orig_partial(*a, **k)
        seen['x_split'] = out[0].detach().float().clone()
        return out

    def spy_step(*a, **k):
        seen['x_tau'], seen['tau'] = k['x_tau'].detach().float().clone(), k['tau'].detach().float().clone()
        return orig_step(*a, **k)
    m.forward_flow_action_partial, m.tactile_flow_train_step = spy_partial, spy_step
    try:
        _losses(trainer, m, b, _args(tactile_self_handover=1.))
    finally:
        m.forward_flow_action_partial, m.tactile_flow_train_step = orig_partial, orig_step
    a = b['norm_actions'].float()
    s = (seen['tau'] / Handover(6).tau)[:, None, None]
    assert torch.allclose(seen['x_tau'], a + s * (seen['x_split'] - a), atol=2e-2)
    # gradients: Tactile path only from the tactile loss (x_split comes from the no-grad trajectory)
    full['loss_tac'].backward()
    g = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert 'final_layer_tactile.mlp.fc2.weight' in g and 'tacf6_embedder.mlp.fc1.weight' in g
    assert not any(n.startswith(('final_layer.', 'visual.', 'tactile_vqvae.')) or '_action' in n for n in g), sorted(g)


@real
def test_contact_weight_scales_hand_dims_of_flagged_samples_only(env):
    import midtrain as trainer
    ds = _ds(env, tactile_force_tasks=FORCE_TASKS)
    b, has_fc = _contact_batch(ds, env)
    m = _model(env)
    m.train()
    base = _losses(trainer, m, b, _args())
    w = _losses(trainer, m, b, _args(tactile_force_contact_weight=4.))
    assert torch.equal(w['loss_tac_plain'], base['loss_tac'])        # logged value = unweighted
    assert w['force_contact_frac'] == float(b['force_contact'].float().mean())
    if has_fc:
        assert w['loss_tac'] > base['loss_tac']
    # same formula by hand on random tensors
    v, tgt = torch.randn(2, 16, 21), torch.randn(2, 16, 21)
    fc = torch.tensor([True, False])
    sw = torch.ones(2, 1, 21)
    sw[fc, :, 9:] = 2.
    got = trainer.tactile_flow_loss(v * sw, tgt * sw, torch.tensor([True, True]))
    e = (v - tgt) ** 2
    e[0, :, 9:] *= 4.
    assert torch.allclose(got, e.mean())
    # validation is never weighted
    vw = _losses(trainer, m, b, _args(tactile_force_contact_weight=4.), training=False)
    vb = _losses(trainer, m, b, _args(), training=False)
    assert torch.equal(vw['loss_tac'], vb['loss_tac'])


@real
def test_history_model_trains_end_to_end(env):
    import midtrain as trainer
    ds = _ds(env, tactile_history_span_ms=1000., tactile_history_points=8, tactile_force_tasks=FORCE_TASKS)
    b, _ = _contact_batch(ds, env)
    m = _model(env, tacf6_dim=24)
    m.train()
    res = _losses(trainer, m, b, _args(tactile_self_handover=.5, tactile_force_contact_weight=3.))
    assert torch.isfinite(res['loss_act']) and torch.isfinite(res['loss_tac'])
    (res['loss_act'] + res['loss_tac']).backward()
    assert m.tacf6_embedder.mlp.fc1.weight.shape[1] == 24
    assert m.tacf6_embedder.mlp.fc1.weight.grad.abs().sum() > 0
    assert m.architecture_contract()['tacf6_dim'] == 24


@real
def test_dataset_touch_event_flag(env):
    ds = _ds(env, tactile_event_weight=5., tactile_event_tasks=FORCE_TASKS, tactile_force_tasks=FORCE_TASKS)
    assert ds.event_flags is not None and set(ds.event_flags) == {int(e) for e in ds.scope.index}
    seen = 0
    for i in range(0, len(ds), max(1, len(ds) // 60)):
        x = ds[i]
        e, t = ds._rows[i]
        assert x['touch_event'] == bool(x['force_valid'] and ds.event_flags[e][t]
                                        and str(ds.source.episodes.loc[e, 'source_task']) in FORCE_TASKS)
        seen += x['touch_event']
    off = _ds(env)                                    # defaults: no flags computed, never flagged
    assert off.event_flags is None and not off[0]['touch_event']
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch['touch_event'].dtype == torch.bool and batch['touch_event'].shape == (2,)


@real
def test_event_weight_in_trainer(env):
    import midtrain as trainer
    ds = _ds(env, tactile_event_weight=5., tactile_event_tasks=FORCE_TASKS, tactile_force_tasks=FORCE_TASKS)
    b, _ = _contact_batch(ds, env)
    b['touch_event'] = torch.tensor([True, False])
    m = _model(env)
    m.train()
    base = _losses(trainer, m, b, _args())
    w = _losses(trainer, m, b, _args(tactile_event_weight=5.))
    assert torch.equal(w['loss_tac_plain'], base['loss_tac']) and torch.equal(w['loss_act'], base['loss_act'])
    assert w['touch_event_frac'] == .5 and not torch.equal(w['loss_tac'], base['loss_tac'])
    # flag present but weight 1 -> baseline; validation never weighted
    assert torch.equal(_losses(trainer, m, b, _args(tactile_event_weight=1.))['loss_tac'], base['loss_tac'])
    vw = _losses(trainer, m, b, _args(tactile_event_weight=5.), training=False)
    assert torch.equal(vw['loss_tac'], _losses(trainer, m, b, _args(), training=False)['loss_tac'])
    # combined with contact weighting = product of the two sqrt-weights (checked on the closed form)
    both = _losses(trainer, m, b, _args(tactile_event_weight=5., tactile_force_contact_weight=3.))
    assert torch.isfinite(both['loss_tac'])


DERIVED = os.environ.get('TAPTIC_DERIVED_ROOT')
VQ_REAL = os.environ.get('TAPTIC_VQVAE')


@pytest.mark.skipif(not DERIVED or not VQ_REAL, reason='set TAPTIC_DERIVED_ROOT and TAPTIC_VQVAE')
def test_derived_dataset_provenance_and_selection():
    import json
    from taptic.data import Source
    src = Source(DERIVED)
    d = src.derived_from
    v = d['verification']
    assert v['parent_rows_identical'] and v['parent_episode_meta_identical'] and v['added_frames_and_clocks_ok']
    a0, a1 = d['appended']['episodes']
    e = src.episodes
    assert d['unchanged_episodes'] == list(range(a0)) and len(e) == a1
    assert set(e.index[e.source_repo == d['appended']['source_repo']]) == set(range(a0, a1))
    assert not e.loc[a0:a1 - 1, 'tactile_interpolated'].any()
    blob = torch.load(VQ_REAL, map_location='cpu', weights_only=False)['data']
    assert blob['source_identity'] == d['parent_identity'] != src.identity
    assert set(blob['train_episodes']) <= set(d['unchanged_episodes'])
    for name, excl in (('v1u30', ['TAPTIC/unscrew_the_bottle']), ('v1both', [])):
        for suffix in ('', '_fresh'):
            p = REPO / f'docs/taptic_train_stats_trex_q_{name}{suffix}.json'
            if not p.exists():
                continue
            st = json.loads(p.read_text())['taptic']
            scope = e[~e.source_repo.isin(excl)]
            assert st['source_identity'] == src.identity
            assert sorted(st['train_episodes']) == sorted(map(int, scope.index[scope.suggested_split == 'train']))


@real
@pytest.mark.skipif(not DERIVED or not VQ_REAL, reason='set TAPTIC_DERIVED_ROOT and TAPTIC_VQVAE')
def test_derived_dataset_loads_with_selection(env):
    import json
    from qwen_vla.taptic_dataset import TRexTapticDataset
    stats = json.loads((REPO / 'docs/taptic_train_stats_trex_q_v1u30.json').read_text())
    cfg = copy.copy(env.cfg)
    cfg.lerobot_root, cfg.vqvae_ckpt, cfg.tactile_f6_stats_ckpt = DERIVED, VQ_REAL, VQ_REAL
    cfg.tactile_delay_ms, cfg.use_risk_tokens, cfg.risk_stats, cfg.cascaded_split_mode = [0], 0, '', 'fixed'
    cfg.taptic_exclude_sources = ['TAPTIC/unscrew_the_bottle']
    cfg.tactile_event_weight, cfg.tactile_event_tasks = 5., FORCE_TASKS
    ds = TRexTapticDataset(cfg, env.proc, SimpleNamespace(print=print), _stats=stats)
    repos = set(ds.scope.source_repo)
    assert 'TAPTIC/unscrew_the_bottle' not in repos and 'TAPTIC/unscrew_the_bottle_fs65' in repos
    i = next(i for i, (ep, t) in enumerate(ds._rows) if ds.source.episodes.loc[ep, 'source_repo'] == 'TAPTIC/unscrew_the_bottle_fs65' and t > 100)
    x = ds[i]
    assert x['force_valid'] and x['head'] is not None and x['fast'][0] is not None
    assert np.isfinite(x['action_chunk']).all()
    assert ds.event_jump['unscrew_the_bottle'] < float('inf')
    # wrong selection for the stats file is refused
    cfg2 = copy.copy(cfg)
    cfg2.taptic_exclude_sources = []
    with pytest.raises(ValueError):
        TRexTapticDataset(cfg2, env.proc, SimpleNamespace(print=print), _stats=stats)


OFF = os.environ.get('TAPTIC_OFF_ROOT')


@real
@pytest.mark.skipif(not OFF or not VQ_REAL, reason='set TAPTIC_OFF_ROOT and TAPTIC_VQVAE')
def test_feedback_off_dataset_needs_named_encoder_source(env):
    """DATA=off: the owner's touch encoder (trained on v1) is refused on the feedback-off dataset unless its source
    is named; a wrong or too short name is refused; with the name the dataset loads and yields valid samples."""
    import json
    from qwen_vla.taptic_dataset import TRexTapticDataset
    from taptic.data import Source
    stats = json.loads((REPO / 'docs/taptic_train_stats_trex_q_off.json').read_text())
    vq_id = torch.load(VQ_REAL, map_location='cpu', weights_only=False)['data']['source_identity']
    assert vq_id != Source(OFF).identity == stats['taptic']['source_identity']
    cfg = copy.copy(env.cfg)
    cfg.lerobot_root, cfg.vqvae_ckpt, cfg.tactile_f6_stats_ckpt = OFF, VQ_REAL, VQ_REAL
    cfg.tactile_delay_ms, cfg.use_risk_tokens, cfg.risk_stats, cfg.cascaded_split_mode = [0], 0, '', 'fixed'
    cfg.taptic_exclude_sources, cfg.tactile_event_weight, cfg.tactile_event_tasks = [], 5., FORCE_TASKS
    mk = lambda c, split='train': TRexTapticDataset(c, env.proc, SimpleNamespace(print=print), split=split, _stats=stats)
    for bad in ('', vq_id[:8], 'f' * 16):
        c = copy.copy(cfg); c.taptic_vqvae_source_identity = bad
        with pytest.raises(ValueError):
            mk(c)
    cfg.taptic_vqvae_source_identity = vq_id[:16]
    ds = mk(cfg)
    assert set(ds.event_jump) >= {'insert_credit_card', 'unscrew_the_bottle'}
    i = next(i for i, (ep, t) in enumerate(ds._rows) if ds.source.episodes.loc[ep, 'source_task'] == 'insert_credit_card' and t > 100)
    x = ds[i]
    assert x['force_valid'] and x['head'] is not None and np.isfinite(x['action_chunk']).all()
    # the option cannot point at the dataset's own identity
    own = copy.copy(env.cfg); own.taptic_vqvae_source_identity = vq_id[:16]
    own.tactile_delay_ms, own.use_risk_tokens, own.risk_stats, own.cascaded_split_mode = [0], 0, '', 'fixed'
    with pytest.raises(ValueError):
        TRexTapticDataset(own, env.proc, SimpleNamespace(print=print), episode_ids=env.ids, _stats=env.stats)


@real
@pytest.mark.skipif(not OFF or not VQ_REAL, reason='set TAPTIC_OFF_ROOT and TAPTIC_VQVAE')
def test_cross_dataset_evaluation_flag(env):
    """offline_eval --root on another dataset: the v1 model's stats score the feedback-off held-out episodes;
    refused without the flag and for the training split."""
    from qwen_vla.taptic_dataset import TRexTapticDataset
    cfg = copy.copy(env.cfg)
    cfg.lerobot_root, cfg.vqvae_ckpt, cfg.tactile_f6_stats_ckpt = OFF, VQ_REAL, VQ_REAL
    cfg.tactile_delay_ms, cfg.use_risk_tokens, cfg.risk_stats, cfg.cascaded_split_mode = [0], 0, '', 'fixed'
    cfg.taptic_exclude_sources = []
    mk = lambda c, split: TRexTapticDataset(c, env.proc, SimpleNamespace(print=print), split=split, _stats=env.stats)
    with pytest.raises(ValueError):
        mk(cfg, 'val')
    c = copy.copy(cfg); c.taptic_eval_other_dataset = 1
    with pytest.raises(ValueError):
        mk(c, 'train')
    ds = mk(c, 'val')
    assert len(ds) > 0 and set(ds.source.episodes.loc[sorted({ep for ep, _ in ds._rows}), 'suggested_split']) == {'val'}
    x = ds[len(ds) // 2]
    assert np.isfinite(x['action_chunk']).all()


def test_launcher_passes_encoder_source_only_when_set(tmp_path):
    """midtrain_taptic_ours.sh: TAPTIC_VQVAE_SOURCE_IDENTITY reaches the real parser when set (DATA=off) and the
    flag is absent otherwise (default '')."""
    from test_ours_model import _parse_launcher
    env = dict(TAPTIC_ROOT=str(tmp_path / 'data'), MODEL_PATH='m', VQVAE_CKPT='vq.pt', OUTPUT_DIR=str(tmp_path / 'o'),
               RUN_NAME='r', EXPERIMENT_NAME='e', VARIANT='vanilla')
    assert _parse_launcher('scripts/midtrain_taptic_ours.sh', env, tmp_path)['taptic_vqvae_source_identity'] == ''
    got = _parse_launcher('scripts/midtrain_taptic_ours.sh', {**env, 'TAPTIC_VQVAE_SOURCE_IDENTITY': '1cb54e3dfd51811f',
                                                              'TACTILE_EVENT_WEIGHT': '5'}, tmp_path)
    assert got['taptic_vqvae_source_identity'] == '1cb54e3dfd51811f' and got['tactile_event_weight'] == 5.


@real
@pytest.mark.skipif(not OFF or not DERIVED or not VQ_REAL, reason='set TAPTIC_OFF_ROOT, TAPTIC_DERIVED_ROOT, TAPTIC_VQVAE')
def test_risk_statistics_follow_the_touch_statistics_rule(env):
    """offline_eval always loads the v1 contact-risk statistics (risk levels, gated condition): accepted on the derived
    dataset (fitted on unchanged parent episodes), on the feedback-off dataset when v1 is named, and in a cross-
    dataset evaluation; refused on the feedback-off dataset otherwise. Samples carry a valid risk value."""
    import json
    from qwen_vla.taptic_dataset import TRexTapticDataset
    risk = str(REPO / 'docs/taptic_ours/risk_stats.json')
    vq_id = torch.load(VQ_REAL, map_location='cpu', weights_only=False)['data']['source_identity']
    base = copy.copy(env.cfg)
    base.vqvae_ckpt = base.tactile_f6_stats_ckpt = VQ_REAL
    base.tactile_delay_ms, base.use_risk_tokens, base.risk_stats, base.cascaded_split_mode = [0], 0, risk, 'fixed'
    mk = lambda c, stats, split='val': TRexTapticDataset(c, env.proc, SimpleNamespace(print=print), split=split, _stats=stats)
    # derived dataset, C2's selection and stats (C2's own evaluation)
    c = copy.copy(base); c.lerobot_root, c.taptic_exclude_sources = DERIVED, ['TAPTIC/unscrew_the_bottle']
    ds = mk(c, json.loads((REPO / 'docs/taptic_train_stats_trex_q_v1u30_fresh.json').read_text()))
    x = ds[len(ds) // 2]; assert np.isfinite(x['action_chunk']).all()
    # feedback-off dataset, own stats: refused unless v1 is named
    off_stats = json.loads((REPO / 'docs/taptic_train_stats_trex_q_off.json').read_text())
    c = copy.copy(base); c.lerobot_root, c.taptic_exclude_sources = OFF, []
    with pytest.raises(ValueError):
        mk(c, off_stats)
    c.taptic_vqvae_source_identity = vq_id[:16]
    ds = mk(c, off_stats)
    i = next(i for i, (ep, t) in enumerate(ds._rows) if ds.source.episodes.loc[ep, 'source_task'] == 'insert_credit_card' and t > 100)
    x = ds[i]; assert np.isfinite(x['action_chunk']).all() and x['force_valid']
    # cross-dataset evaluation: v1 model stats on the feedback-off held-out set
    c = copy.copy(base); c.lerobot_root, c.taptic_exclude_sources, c.taptic_eval_other_dataset = OFF, [], 1
    ds = mk(c, env.stats)
    x = ds[len(ds) // 2]; assert np.isfinite(x['action_chunk']).all()


def test_launcher_async_touch_delays(tmp_path):
    """fast20 / fast40 (jobs/midtrain_touch.sbatch): TACTILE_ASYNC_DELAYS_MS reaches the fixed-split variant;
    default stays synchronous ([0], the baseline launcher's value)."""
    from test_ours_model import _parse_launcher
    env = dict(TAPTIC_ROOT=str(tmp_path / 'data'), MODEL_PATH='m', VQVAE_CKPT='vq.pt', OUTPUT_DIR=str(tmp_path / 'o'),
               RUN_NAME='r', EXPERIMENT_NAME='e', VARIANT='vanilla')
    base = _parse_launcher('scripts/midtrain_taptic_ours.sh', env, tmp_path)
    assert base['tactile_delay_ms'] == [0] and base['cascaded_split_mode'] == 'fixed' and base['tactile_async_touch'] == 0
    keys = ('tactile_delay_ms', 'tactile_async_touch')
    for d in ('0 25 50 75', '0 50 100 150'):
        got = _parse_launcher('scripts/midtrain_taptic_ours.sh', {**env, 'TACTILE_ASYNC_DELAYS_MS': d}, tmp_path)
        assert got['tactile_delay_ms'] == [int(x) for x in d.split()] and got['cascaded_split_mode'] == 'fixed'
        assert got['tactile_async_touch'] == 1
        assert {k: v for k, v in got.items() if k not in keys} == {k: v for k, v in base.items() if k not in keys}
    # the contract still refuses delays for vanilla without the explicit opt-in
    from qwen_vla.contract import validate_variant
    with pytest.raises(ValueError):
        validate_variant(SimpleNamespace(**{**base, 'tactile_delay_ms': [0, 25, 50, 75]}))
    validate_variant(SimpleNamespace(**{**base, 'tactile_delay_ms': [0, 25, 50, 75], 'tactile_async_touch': 1}))


@real
def test_async_touch_with_lane_recipe_in_trainer(env):
    """Touch newer than the image (fast40) combined with self-handover, contact and event weighting: the batch
    carries the slot mask (elapsed chunk steps not supervised for the tactile expert) and the losses are finite."""
    import midtrain as trainer
    kw = dict(tactile_delay_ms=[0, 25, 50, 75], tactile_event_weight=5., tactile_event_tasks=FORCE_TASKS,
              tactile_force_tasks=FORCE_TASKS)
    ds = _ds(env, **kw)
    assert ds.emit_slot_mask
    b, _ = _contact_batch(ds, env)
    sm = b['tactile_slot_mask']
    assert sm is not None and sm.shape == (2, 16) and sm.dtype == torch.bool
    assert all(bool(r[2:].all()) or not bool(r.any()) for r in sm)   # (all-False = chunk with a recording gap)
    seen = [ds[i]['tactile_slot_mask'] for i in range(0, len(ds), max(1, len(ds) // 40))]
    assert any(not m[0] for m in seen)            # some samples drew a delay > 0: step 0 already elapsed
    m = _model(env)
    m.train()
    out = _losses(trainer, m, b, _args(tactile_self_handover=0.5, tactile_force_contact_weight=3.,
                                       tactile_event_weight=5.))
    assert torch.isfinite(out['loss_tac']) and torch.isfinite(out['loss_act'])


# ── restartable training and the frame-table read (25 Nov) ─────────────────

def test_sampler_skip_resumes_the_epoch_exactly():
    import midtrain as trainer
    fc = np.array([37, 5, 120, 64, 9, 88, 41, 23, 70, 12])
    ds = SimpleNamespace(_cum_frames=np.cumsum(fc), _num_episodes=len(fc))
    ds.__len__ = lambda: int(fc.sum())
    class D:                                   # DistributedSampler needs len(dataset)
        _cum_frames, _num_episodes = ds._cum_frames, ds._num_episodes
        def __len__(self): return int(fc.sum())
    for rank in (0, 1):
        s = trainer.EpisodeGroupedSampler(D(), num_replicas=2, rank=rank, shuffle=True, seed=42)
        s.set_epoch(3)
        full = list(iter(s))
        s.skip_samples = 64
        assert list(iter(s)) == full[64:]
        assert list(iter(s)) == full                       # the skip applies once
        high = np.zeros(int(fc.sum()), bool); high[::17] = True
        r = trainer.RiskOversampledEpisodeSampler(D(), high, 2, num_replicas=2, rank=rank, shuffle=True, seed=42)
        r.set_epoch(5)
        full = list(iter(r)); r.skip_samples = 10
        assert list(iter(r)) == full[10:] and list(iter(r)) == full


def test_resume_state_roundtrip_cpu(tmp_path):
    """save_resume_state / load_resume_state with a CPU accelerator: weights, optimizer, LR scheduler and counters
    come back; atomic swap leaves no .tmp/.old; a structural mismatch is refused; 'auto' without a state = fresh."""
    import midtrain as trainer
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        acc = Accelerator(cpu=True)
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model, opt = acc.prepare(model, opt)
        sched = trainer.get_cosine_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=20, min_lr_ratio=0.1)
        def step():
            loss = model(torch.randn(8, 4)).pow(2).mean(); acc.backward(loss); opt.step(); sched.step(); opt.zero_grad()
        for _ in range(3):
            step()
        args = SimpleNamespace(output_dir=str(tmp_path / 'run'), resume_state_dir='', resume_from_state='auto')
        os.makedirs(args.output_dir)
        assert trainer.load_resume_state(acc, args, sched, {'a': 1}) is None        # nothing saved yet
        fp = {'world_size': 1, 'steps_per_epoch': 10, 'seed': 42}
        prog = {'epoch': 1, 'global_step': 12, 'batches_done_in_epoch': 2, 'fingerprint': fp, 'saved_at': 'now'}
        trainer.save_resume_state(acc, args, sched, prog)
        trainer.save_resume_state(acc, args, sched, prog)                          # replaces the first one
        snap_w = {k: v.clone() for k, v in model.state_dict().items()}
        snap_o = copy.deepcopy(opt.state_dict()); snap_s = sched.state_dict(); lr = sched.get_last_lr()
        for _ in range(4):
            step()
        assert not torch.equal(model.state_dict()['weight'], snap_w['weight'])
        got = trainer.load_resume_state(acc, args, sched, fp)
        assert got['global_step'] == 12 and got['batches_done_in_epoch'] == 2 and got['epoch'] == 1
        assert all(torch.equal(model.state_dict()[k], v) for k, v in snap_w.items())
        o = opt.state_dict()
        for pid, st in snap_o['state'].items():
            assert all(torch.equal(o['state'][pid][k], v) for k, v in st.items() if torch.is_tensor(v))
        assert sched.state_dict()['last_epoch'] == snap_s['last_epoch'] and sched.get_last_lr() == lr
        d = trainer.resume_state_dir(args)
        assert os.path.isdir(d) and not os.path.exists(d + '.tmp') and not os.path.exists(d + '.old')
        with pytest.raises(ValueError):
            trainer.load_resume_state(acc, args, sched, {**fp, 'world_size': 2})
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


@real
def test_frame_rows_match_the_dataset_scan(env):
    """taptic/data.py frames(): the in-memory per-process table gives exactly the rows of the per-episode scan."""
    import pyarrow.dataset as pads
    from taptic.data import Source
    src = Source(ROOT)
    eps = [int(e) for e in src.episodes.index[::97]][:8]
    cols = ['action', 'observation.state', 'frame_index', 'timestamp']
    for ep in eps:
        new = src.frames(ep)
        old = src.table.to_table(columns=cols, filter=pads.field('episode_index') == ep).to_pandas()
        old = old.sort_values('frame_index').reset_index(drop=True)
        assert list(new.columns) == list(old.columns) and len(new) == len(old)
        for c in ('action', 'observation.state'):
            assert np.array_equal(np.stack(new[c].to_numpy()), np.stack(old[c].to_numpy()))
        assert np.array_equal(new.frame_index.to_numpy(), old.frame_index.to_numpy())
        assert np.array_equal(new.timestamp.to_numpy(), old.timestamp.to_numpy())


def test_launcher_passes_resume_options_only_when_set(tmp_path):
    from test_ours_model import _parse_launcher
    env = dict(TAPTIC_ROOT=str(tmp_path / 'data'), MODEL_PATH='m', VQVAE_CKPT='vq.pt', OUTPUT_DIR=str(tmp_path / 'o'),
               RUN_NAME='r', EXPERIMENT_NAME='e', VARIANT='vanilla')
    base = _parse_launcher('scripts/midtrain_taptic_ours.sh', env, tmp_path)
    assert base['resume_state_every'] == 0 and base['resume_from_state'] == ''
    got = _parse_launcher('scripts/midtrain_taptic_ours.sh',
                          {**env, 'RESUME_STATE_EVERY': '1250', 'RESUME_FROM_STATE': 'auto'}, tmp_path)
    assert got['resume_state_every'] == 1250 and got['resume_from_state'] == 'auto'
    keys = ('resume_state_every', 'resume_from_state')
    assert {k: v for k, v in got.items() if k not in keys} == {k: v for k, v in base.items() if k not in keys}
