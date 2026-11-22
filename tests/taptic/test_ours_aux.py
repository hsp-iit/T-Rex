"""OURS training components added on top of the risk tokens:
future-risk head (Tactile expert) + detached probe (Action expert), risk dropout,
compute-neutral high-risk oversampling.

Real-data parts need TAPTIC_ROOT and TAPTIC_PROCESSOR (tiny random models, nothing downloaded);
the sampler and dropout unit tests run anywhere.
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
OURS_TIP = 'b1a651032a47e8c4b8367b985b37dc7d179a8162'      # mfincato/trex-ours before these changes

if ROOT and PROCESSOR:
    from test_ours_data import env, _item, DELAYS, MS  # noqa: F401,E402  (module-scoped fixture)


# ── unit tests (no data) ────────────────────────────────────────────────────

def test_risk_dropout_unit():
    import midtrain as trainer
    v = torch.tensor([True, False] * 5000)
    state = torch.random.get_rng_state()
    assert trainer._risk_dropout(v, 0., True) is v and trainer._risk_dropout(v, .3, False) is v
    assert torch.equal(torch.random.get_rng_state(), state)          # disabled -> no RNG draw
    torch.manual_seed(0)
    d = trainer._risk_dropout(v, .3, True)
    assert not d[~v].any()                                           # missing stays missing
    assert abs(d[v].float().mean().item() - .7) < .03
    assert trainer._risk_dropout(None, .3, True) is None


class _Rows:
    """Dataset stub: episode lengths only (what the samplers use)."""
    def __init__(self, lengths):
        self._cum_frames = np.cumsum(lengths).astype(np.int64)
        self._num_episodes = len(lengths)

    def __len__(self):
        return int(self._cum_frames[-1])


def test_oversampling_sampler_is_compute_neutral_and_grouped():
    import midtrain as trainer
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 60, size=30)
    lengths[-1] += lengths.sum() % 2                                  # even N: 2 ranks cover every slot
    ds = _Rows(lengths)
    N = len(ds)
    high = rng.random(N) < .12
    k = 3
    for epoch in (0, 1):
        per_rank = []
        for rank in (0, 1):
            s = trainer.RiskOversampledEpisodeSampler(ds, high, k, num_replicas=2, rank=rank, seed=7)
            s.set_epoch(epoch)
            idx = list(s)
            assert len(idx) == s.num_samples == len(trainer.EpisodeGroupedSampler(ds, num_replicas=2, rank=rank))
            per_rank.append(np.asarray(idx))
            again = trainer.RiskOversampledEpisodeSampler(ds, high, k, num_replicas=2, rank=rank, seed=7)
            again.set_epoch(epoch)
            assert list(again) == idx                                 # deterministic per (seed, epoch)
        allidx = np.concatenate(per_rank)
        counts = np.bincount(allidx, minlength=N)
        assert (counts[high] == k).all()                              # every high-risk row k times
        assert ((counts[~high] == 0).sum() == (k - 1) * high.sum()) and (counts[~high] <= 1).all()
        # frames of one episode stay contiguous and in order (duplicates adjacent)
        ep_of = np.searchsorted(ds._cum_frames, allidx, side='right')
        blocks = np.flatnonzero(np.diff(ep_of)) + 1
        for a, b in zip(np.r_[0, blocks], np.r_[blocks, len(allidx)]):
            assert (np.diff(allidx[a:b]) >= 0).all()
        assert len(set(ep_of[np.r_[0, blocks]])) == len(blocks) + 1   # each episode visited once
        if epoch == 0:
            first = counts
    assert not np.array_equal(first, counts)                          # fresh drop draw each epoch
    with pytest.raises(ValueError):
        trainer.RiskOversampledEpisodeSampler(ds, np.ones(N, bool), 2)  # nothing left to drop


def test_risk_option_guards():
    from qwen_vla.contract import validate_variant
    base = dict(cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6], tactile_delay_ms=[0, 50],
                risk_stats='r.json', use_risk_tokens=1, risk_future_head_weight=.1, risk_future_probe=1,
                risk_dropout=.1, risk_oversample=2)
    validate_variant(SimpleNamespace(variant='ours', **base))
    validate_variant(SimpleNamespace(variant='multisplit', **{**base, 'use_risk_tokens': 0, 'risk_dropout': 0,
                                                              'risk_future_head_weight': 0, 'risk_oversample': 1}))
    bad = [('multisplit', dict(use_risk_tokens=0, risk_dropout=0, risk_oversample=1)),     # head in multisplit
           ('multisplit', dict(use_risk_tokens=0, risk_dropout=0, risk_future_head_weight=0)),  # oversampling
           ('ours', dict(risk_oversample=2.5)), ('ours', dict(risk_stats='')),
           ('vanilla', dict(cascaded_split_mode='fixed', cascaded_split_step=6, tactile_delay_ms=[0],
                            use_risk_tokens=0, risk_future_head_weight=0, risk_dropout=0, risk_oversample=1))]
    for v, over in bad:
        with pytest.raises(ValueError):
            validate_variant(SimpleNamespace(variant=v, **{**base, **over}))
    with pytest.raises(ValueError):          # risk dropout without risk tokens
        validate_variant(SimpleNamespace(variant='', **{**base, 'use_risk_tokens': 0}))


# ── real data ───────────────────────────────────────────────────────────────

def _future_ds(env, **over):
    from qwen_vla.taptic_dataset import TRexTapticDataset
    cfg = copy.copy(env.cfg)
    cfg.risk_future_head_weight, cfg.risk_future_probe = .1, 1
    for k, v in over.items():
        setattr(cfg, k, v)
    return TRexTapticDataset(cfg, env.proc, SimpleNamespace(print=print), episode_ids=env.ids, _stats=env.stats)


@real
def test_future_risk_targets_are_causal_per_slot_and_never_inputs(env):
    from taptic.risk import risk_features
    ds = _future_ds(env)
    for ep in env.natives[:2] + [env.legacy]:
        rows = [i for i, (e, _) in enumerate(ds._rows) if e == ep]
        for idx in (rows[0], rows[len(rows) // 2]):
            t = ds._rows[idx][1]
            for d in DELAYS:
                x = _item(ds, idx, d)
                y = _item(env.ds, idx, d)                       # same dataset without future targets
                f = x['risk_future']
                assert y['risk_future'] is None and f['risk'].shape == (16,)
                # inputs are unchanged by the targets
                for k in ('force_history', 'tactile_f6', 'action_chunk', 'tactile_slot_mask'):
                    assert np.array_equal(x[k], y[k]), k
                assert x['risk_S'] == y['risk_S'] and x['risk_F'] == y['risk_F']
                d_, s, a, times, sm, av = ds._episode(ep)
                slot = np.asarray(times[t:t + 16], np.int64)
                ref = risk_features(ds._cache_force, slot, ds.risk_stats, ds.risk_cfg, ds._cache_joint) \
                    if ep != env.legacy else None
                if ep == env.legacy:
                    assert not f['valid'].any()
                else:
                    np.testing.assert_array_equal(f['valid'], ref['risk_valid'])
                    np.testing.assert_allclose(f['risk'], ref['risk'].astype(np.float32))
                    # slot 0 is the slow tick itself: r(t_0) == r_S
                    assert f['risk'][0] == pytest.approx(x['risk_S']['risk'], abs=1e-6)
                S, F = x['meta']['slow_ns'], x['meta']['fast_ns']
                assert np.array_equal(f['after_S'], slot > S) and not f['after_S'][0]
                assert np.array_equal(f['after_F'], (slot > F) & x['meta']['fast_tick_aligned'])
    b = ds.collate_fn([_item(ds, 0, 0), _item(ds, 1, 50)])
    assert b['risk_future'].shape == (2, 16) and b['risk_future_valid'].dtype == torch.bool
    assert env.ds.collate_fn([_item(env.ds, 0, 0)])['risk_future'] is None


@real
def test_anchor_risk_matches_getitem(env):
    ds = _future_ds(env)
    r, v = ds.anchor_risk()
    assert r.shape == (len(ds),)
    for i in list(range(0, len(ds), max(1, len(ds) // 25))):
        x = _item(ds, i, 0)
        assert bool(v[i]) == x['risk_S']['risk_valid'] and r[i] == pytest.approx(x['risk_S']['risk'], abs=1e-6)


def _model(env, head=True, probe=True, risk=True, seed=7):
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
                        use_tactile_vqvae=True, vqvae_config=blob['config'], use_risk_tokens=risk,
                        risk_future_head=head, risk_future_probe=probe).float()
    m.initialize_vla_weights()
    m.load_tactile_vqvae(blob)
    reinit_tactile_expert(m)
    torch.nn.init.normal_(m.final_layer.mlp.fc2.weight, std=.05)
    m.visual = base.model.visual
    m.visual.requires_grad_(False)
    object.__setattr__(m, '_rope_index_fn', base.model.get_rope_index)
    return m.eval()


def _batch(ds, env):
    i0 = next(i for i, (e, t) in enumerate(ds._rows) if e == env.natives[0] and t == 20)
    i1 = next(i for i, (e, t) in enumerate(ds._rows) if e == env.natives[1] and t == 20)
    return ds.collate_fn([_item(ds, i0, DELAYS[-1]), _item(ds, i1, DELAYS[1])])


def _args(**kw):
    return SimpleNamespace(action_chunk=16, use_robot_state=0, use_tactile_vec=1, use_tactile_deform=0,
                           cascaded_total_steps=10, flow_time_precision='fp32', cascaded_tactile_dropout=0.,
                           **kw)


def _losses(trainer, m, b, args, tau=.4, training=True, seed=5):
    from qwen_vla.flow_matching import Handover
    with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
        torch.manual_seed(seed)
        return trainer.cascaded_losses(m, m, b, args, Handover(10 - round(tau * 10)), is_stage1=False,
                                       use_flare=True, K=4, training=training)


@real
def test_future_head_trains_tactile_path_and_probe_is_detached(env):
    import midtrain as trainer
    m = _model(env)
    # zero init survives the trainer's init sequence (initialize_vla_weights -> reinit_tactile_expert,
    # which re-initializes every '*_tactile*' parameter): the heads start by predicting 0
    for lin in (m.future_risk_head, m.future_risk_probe):
        assert not lin.weight.any() and not lin.bias.any()
    b = _batch(_future_ds(env), env)
    # zero init would block the gradient into the tactile path; check routing with random weights
    for lin in (m.future_risk_head, m.future_risk_probe):
        torch.nn.init.normal_(lin.weight, std=.1)
    m.train()
    res = _losses(trainer, m, b, _args())
    assert torch.isfinite(res['loss_risk_head']) and torch.isfinite(res['loss_risk_probe'])
    assert res['risk_head_stats'] is not None and res['risk_head_stats'][2] > 0

    res['loss_risk_probe'].backward(retain_graph=True)
    g = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert g == {'future_risk_probe.weight', 'future_risk_probe.bias'}, g
    m.zero_grad(set_to_none=True)

    res['loss_risk_head'].backward()
    g = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert {'future_risk_head.weight', 'future_risk_head.bias',
            'risk_embedder_tactile.mlp.0.weight', 'tactile_code_embedder.weight'} <= g
    assert any('_tactile' in n for n in g if n.startswith('model.layers.'))
    # the cache is no-grad: nothing in the Action expert, its risk token, the probe, vision or VQ-VAE
    assert not any(n.startswith(('risk_embedder_action', 'future_risk_probe', 'visual.', 'tactile_vqvae.'))
                   or '_action' in n for n in g), sorted(g)

    # the head is masked to future, valid slots after F; the probe to slots after S
    pred = torch.zeros(2, 16, requires_grad=True)
    loss, _ = trainer._future_risk_terms(pred, b, 'risk_future_after_F', None, None)
    loss.backward()
    mask = b['risk_future_valid'] & b['risk_future_after_F']
    assert (pred.grad != 0).eq(mask & (b['risk_future'] != 0)).all()


@real
@pytest.mark.parametrize('risk', [True, False])
def test_disabled_components_are_bitwise_identical_to_ours_tip(env, risk):
    """No head/probe, no risk dropout: losses == the trex-ours tip's trainer, bit for bit,
    even when the batch carries future-risk targets."""
    import midtrain as trainer
    src = subprocess.check_output(['git', 'show', f'{OURS_TIP}:scripts/midtrain.py'], text=True, cwd=REPO)
    tip = types.ModuleType('tip_midtrain')
    tip.__file__ = str(REPO / 'scripts' / 'midtrain.py')
    exec(compile(src, '<tip:midtrain>', 'exec'), tip.__dict__)
    m = _model(env, head=False, probe=False, risk=risk)
    m.train()
    b = _batch(_future_ds(env), env)
    for tau in (.2, .6):
        new = _losses(trainer, m, b, _args(risk_dropout=0.), tau=tau)
        old = _losses(tip, m, b, _args(), tau=tau)
        assert torch.equal(new['loss_act'], old['loss_act']) and torch.equal(new['loss_tac'], old['loss_tac'])
        assert new['loss_risk_head'] is None and new['loss_risk_probe'] is None


@real
def test_risk_dropout_reaches_both_tokens_in_training_only(env):
    import midtrain as trainer
    m = _model(env, head=False, probe=False)
    m.train()
    seen = []
    orig = m.risk_token

    def spy(risk, risk_valid, expert, *a, **k):
        seen.append((expert, None if risk_valid is None else risk_valid.clone()))
        return orig(risk, risk_valid, expert, *a, **k)
    m.risk_token = spy
    b = _batch(_future_ds(env), env)
    assert b['risk_S_valid'].all() and b['risk_F_valid'].any()
    _losses(trainer, m, b, _args(risk_dropout=.999), training=True)
    assert seen and all(not v.any() for _, v in seen)                 # dropped in every call
    assert {e for e, _ in seen} == {'action', 'tactile'}
    seen.clear()
    _losses(trainer, m, b, _args(risk_dropout=.999), training=False)  # validation: untouched
    assert any(v.any() for e, v in seen if e == 'action')
    assert b['risk_S_valid'].all()                                     # the batch itself is never modified
