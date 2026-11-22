"""Risk-conditioned T-Rex model: token layout, causal visibility, gradients,
multi-split flow times, cache-ladder coherence/immutability, regression against
the base commit, weight schema / checkpoint contract and the inference engine.

Reduced random models on CPU (hidden 64, 2 MoT layers, real TAPTIC token layout:
5 VQ code tokens + 5 force tokens). No checkpoint is downloaded or written.
"""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_vla import Qwen3VLVLAModel, ActionCacheLadder  # noqa: E402
from qwen_vla.flow_matching import Handover, flow_time, tactile_training_pair  # noqa: E402
from qwen_vla.contract import load_with_contract, validate_variant  # noqa: E402
from tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig  # noqa: E402

BASE = '05c932853ac5984ac7128561de2583c34b568262'
VQ = TactileVQVAEConfig(window=16, per_finger_dim=3, n_hands=1, in_channels=15, codebook_size=16,
                        hidden_channels=16, bottleneck_channels=32, embed_dim=16, granularity='finger',
                        input_signal='taptic_sdk_force')
B, L, H, NF = 2, 12, 64, 4
AC = torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False)


def text_config(layers=2):
    from transformers import Qwen3VLConfig
    t = Qwen3VLConfig().text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = H, 128, layers
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.rope_scaling = {'rope_type': 'default', 'mrope_section': [6, 5, 5], 'mrope_interleaved': True}
    t._attn_implementation = 'eager'
    return t


def tiny(risk=True, seed=0, cls=Qwen3VLVLAModel):
    from midtrain import reinit_tactile_expert
    torch.manual_seed(seed)
    kw = dict(use_risk_tokens=True) if risk else {}
    m = cls(text_config(), action_dim=21, action_chunk=16, tacf6_dim=3, tactile_intermediate_size=64,
            use_tactile_code=True, vqvae_codebook_size=16, use_tactile_vqvae=True, vqvae_config=VQ.to_dict(),
            **kw).float()
    m.initialize_vla_weights()
    m.load_tactile_vqvae({'model_state': m.tactile_vqvae.state_dict(),
                          'stats': {'tacf6_min': -60 * np.ones(15), 'tacf6_max': 60 * np.ones(15),
                                    'tacf6_mask': np.ones(15, bool)}})
    reinit_tactile_expert(m)
    torch.nn.init.normal_(m.final_layer.mlp.fc2.weight, std=.05)     # live action head
    return m.eval()


def ctx(seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(slow=torch.randn(B, L, H, generator=g), fast=torch.randn(B, NF, H, generator=g),
                pos=torch.arange(L).view(1, 1, L).expand(3, B, L).clone(),
                noise=torch.randn(B, 16, 21, generator=g), tac=torch.randn(B, 5, 3, generator=g),
                hist=torch.randint(-50, 50, (B, 16, 5, 3), generator=g).float(),
                rS=torch.tensor([.3, .8]), vS=torch.tensor([True, False]),
                rF=torch.tensor([.6, .1]), vF=torch.tensor([True, True]))


def partial(m, c, steps=6, risk=True, **kw):
    r = dict(risk=c['rS'], risk_valid=c['vS']) if risk else {}
    return m.forward_flow_action_partial(c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'],
                                         split_step=steps, **r, **kw)


def tac_kw(c, risk=True, rF=None):
    kw = dict(tactile_f6=c['tac'], tactile_f6_history=c['hist'])
    if risk:
        kw.update(risk=c['rF'] if rF is None else rF, risk_valid=c['vF'])
    return kw


class Spy:
    """Record every MoT backbone call (routing, positions, embeddings, cache length)."""
    def __init__(self, m):
        self.calls, self.m, self.orig = [], m, m.model.forward
        m.model.forward = self

    def __call__(self, **kw):
        pkv = kw.get('past_key_values')
        self.calls.append({'emb': kw['inputs_embeds'].detach().clone(), 'pos': kw['position_ids'].detach().clone(),
                           'lat': kw['latent_indexes'].tolist(), 'act': kw['action_indexes'].tolist(),
                           'tac': kw['tactile_indexes'].tolist(),
                           'past': pkv.get_seq_length() if pkv is not None else 0})
        out = self.orig(**kw)
        self.calls[-1]['hidden'] = out.last_hidden_state.detach().clone()
        return out

    def close(self):
        del self.m.model.forward


# ── layout, routing, positions ──────────────────────────────────────────────

@torch.no_grad()
def test_token_order_routing_positions_and_cache_lengths():
    m, c = tiny(), ctx()
    with AC:
        spy = Spy(m)
        x, cache, nact, tau = partial(m, c, 6)
        n_act = NF + 1 + 1 + 16                                  # fast | risk_S | t | chunk
        assert nact == n_act and cache.get_seq_length() == L + n_act and tau == .4
        rtok = m.risk_token(c['rS'], c['vS'], 'action', B, 'cpu', torch.bfloat16)
        first = spy.calls[0]
        assert first['lat'] == list(range(L)) and first['act'] == list(range(L, L + n_act)) and not first['tac']
        assert torch.equal(first['emb'][:, L + NF], rtok[:, 0])  # risk_S right before the t token
        max_pos = int(c['pos'].max())
        for call in spy.calls[1:]:                               # 5 continuation forwards + refresh
            assert call['past'] == L and call['act'] == list(range(n_act)) and not call['lat']
            assert call['pos'][0, 0].tolist() == list(range(max_pos + 1, max_pos + 1 + n_act))
            assert torch.equal(call['emb'][:, NF], rtok[:, 0])
        assert len(spy.calls) == 7
        spy.calls.clear()
        h = Handover(6)
        xt, tt, _ = tactile_training_pair(c['noise'] * 0, c['noise'], torch.full((B,), .5), h)
        m.tactile_flow_train_step(cache, c['pos'], nact, xt, tt, **tac_kw(c))
        call = spy.calls[0]
        n_tac = 10 + 1 + 1 + 16                                   # 5 codes + 5 force | risk_F | tau | chunk
        ftok = m.risk_token(c['rF'], c['vF'], 'tactile', B, 'cpu', torch.bfloat16)
        assert call['tac'] == list(range(n_tac)) and not call['act'] and call['past'] == L + n_act
        assert torch.equal(call['emb'][:, 10], ftok[:, 0])
        assert call['pos'][0, 0].tolist() == list(range(max_pos + n_act + 1, max_pos + n_act + 1 + n_tac))
        assert cache.get_seq_length() == L + n_act                # training never mutates the handover
        spy.close()
        assert m.architecture_contract()['action_stream'] == 'fast|state|risk_S|t|x'


@torch.no_grad()
def test_causal_visibility_of_risk_tokens():
    m, c = tiny(), ctx()
    with AC:
        def action_hidden(r):
            tok = m.risk_token(r, torch.ones(B, dtype=torch.bool), 'action', B, 'cpu', torch.bfloat16)
            t = m.t_embedder(torch.full((B,), .7)).unsqueeze(1)
            seq = torch.cat([c['slow'], c['fast'], tok, t, m.x_embedder(c['noise'].bfloat16())], 1)
            n = seq.shape[1]
            return m.model(inputs_embeds=seq, position_ids=c['pos'], use_cache=False,
                           latent_indexes=torch.arange(L), action_indexes=torch.arange(L, n),
                           tactile_indexes=torch.arange(0)).last_hidden_state
        h1, h2 = action_hidden(torch.tensor([.1, .1])), action_hidden(torch.tensor([.9, .9]))
        before = L + NF                                           # latent + fast precede risk_S
        assert torch.equal(h1[:, :before], h2[:, :before])        # nothing earlier can see r_S
        assert not torch.equal(h1[:, -16:], h2[:, -16:])          # the chunk attends to r_S
        # The handover cache: latent/fast KV independent of r_S, risk..chunk KV dependent.
        _, ca, _, _ = partial(m, c, 6)
        c2 = dict(c, rS=torch.tensor([.9, .05]), vS=torch.tensor([True, True]))
        _, cb, _, _ = partial(m, c2, 6)
        for la, lb in zip(ca.layers, cb.layers):
            assert torch.equal(la.keys[:, :, :before], lb.keys[:, :, :before])
            assert not torch.equal(la.keys[:, :, before:], lb.keys[:, :, before:])
        # Tactile: r_F is visible to the chunk only; observation tokens before it unchanged.
        x, cache, nact, tau = partial(m, c, 6)
        spy = Spy(m)
        h = Handover(6)
        xt, tt, _ = tactile_training_pair(c['noise'] * 0, c['noise'], torch.full((B,), .5), h)
        v1 = m.tactile_flow_train_step(cache, c['pos'], nact, xt, tt, **tac_kw(c, rF=torch.tensor([.0, .0])))
        v2 = m.tactile_flow_train_step(cache, c['pos'], nact, xt, tt, **tac_kw(c, rF=torch.tensor([1., 1.])))
        spy.close()
        assert torch.equal(spy.calls[0]['hidden'][:, :10], spy.calls[1]['hidden'][:, :10])
        assert not torch.equal(v1, v2)


@torch.no_grad()
def test_missing_risk_is_not_zero_and_contract_is_strict():
    m, c = tiny(), ctx()
    with AC:
        zero = m.risk_token(torch.zeros(B), torch.ones(B, dtype=torch.bool), 'action', B, 'cpu', torch.bfloat16)
        miss = m.risk_token(torch.full((B,), float('nan')), torch.zeros(B, dtype=torch.bool), 'action', B,
                            'cpu', torch.bfloat16)
        assert not torch.equal(zero, miss) and torch.isfinite(miss).all()
        a = m.risk_token(torch.zeros(B), torch.ones(B, dtype=torch.bool), 'tactile', B, 'cpu', torch.bfloat16)
        assert not torch.equal(a, zero)                           # separate embedders per expert
        with pytest.raises(ValueError, match='nonfinite'):
            m.risk_token(torch.full((B,), float('nan')), torch.ones(B, dtype=torch.bool), 'action', B, 'cpu',
                         torch.bfloat16)
        with pytest.raises(ValueError, match='risk'):
            m.forward_flow_action_partial(c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'])
        x, cache, nact, tau = partial(m, c, 6)
        with pytest.raises(ValueError, match='risk'):
            m.tactile_flow_continue(cache, c['pos'], nact, x, tau, tactile_f6=c['tac'], tactile_f6_history=c['hist'])
        with pytest.raises(ValueError, match='risk'):
            m.forward_flow_action_full(c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'])
        base = tiny(risk=False)
        with pytest.raises(ValueError, match='no risk tokens'):
            partial(base, c, 6)
        # A baseline cache cannot be continued by a risk model (layout mismatch) and vice versa.
        xb, cb, nb, tb = partial(base, c, 6, risk=False)
        with pytest.raises(ValueError, match='layout'):
            m.tactile_flow_continue(cb, c['pos'], nb, xb, tb, **tac_kw(c))


# ── gradients and training-time flow times ─────────────────────────────────

def test_risk_embedders_train_with_their_own_expert():
    m, c = tiny(), ctx()
    m.train()
    with AC:
        tok = m.risk_token(c['rS'], c['vS'], 'action', B, 'cpu', torch.bfloat16)
        t = m.t_embedder(torch.full((B,), .7)).unsqueeze(1)
        seq = torch.cat([c['slow'], c['fast'], tok, t, m.x_embedder(c['noise'].bfloat16())], 1)
        out = m.model(inputs_embeds=seq, position_ids=c['pos'], use_cache=False, latent_indexes=torch.arange(L),
                      action_indexes=torch.arange(L, seq.shape[1]), tactile_indexes=torch.arange(0))
        loss_act = F.mse_loss(m.final_layer(out.last_hidden_state[:, -16:]).float(), c['noise'])
    loss_act.backward()
    g = {n: p.grad.abs().sum().item() for n, p in m.named_parameters() if p.grad is not None}
    assert g.get('risk_embedder_action.mlp.0.weight', 0) > 0 and g.get('risk_embedder_action.mlp.2.weight', 0) > 0
    assert not any(n.startswith('risk_embedder_tactile') and v > 0 for n, v in g.items())
    m.zero_grad(set_to_none=True)
    for steps in (8, 6, 4):
        h = Handover(steps)
        with AC:
            _, cache, nact, tau = partial(m, c, steps)
            seen = []
            hook = m.t_embedder.register_forward_pre_hook(lambda mod, a: seen.append(a[0].detach().clone()))
            xt, tt, target = tactile_training_pair(c['noise'] * .3, c['noise'], torch.full((B,), .999), h)
            v = m.tactile_flow_train_step(cache, c['pos'], nact, xt, tt, **tac_kw(c))
            hook.remove()
            loss_tac = F.mse_loss(v.float(), target.float())
        assert seen[0].dtype == torch.float32 and float(seen[0].max()) <= h.tau + 1e-7 and float(seen[0].min()) > 0
        loss_tac.backward()
        g = {n: p.grad.abs().sum().item() for n, p in m.named_parameters() if p.grad is not None}
        assert g.get('risk_embedder_tactile.mlp.0.weight', 0) > 0
        assert not any((n.startswith('risk_embedder_action') or '_action' in n) and v > 0 for n, v in g.items())
        assert not any(n.startswith('tactile_vqvae') for n in g)
        assert any(n.startswith('tactile_code_embedder') and v > 0 for n, v in g.items())
        m.zero_grad(set_to_none=True)


@torch.no_grad()
@pytest.mark.parametrize('tau', [.2, .4, .6])
def test_split_schedules_use_integer_fp32_flow_times(tau):
    m, c = tiny(), ctx()
    h = Handover(10 - round(tau * 10))
    with AC:
        seen = []
        hook = m.t_embedder.register_forward_pre_hook(lambda mod, a: seen.append(a[0].detach().clone()))
        x, cache, nact, got = partial(m, c, h.action_steps)
        assert got == h.tau and [float(t[0]) for t in seen] == [flow_time(i).item() for i in range(h.action_steps + 1)]
        seen.clear()
        out = m.tactile_flow_continue(cache, c['pos'], nact, x, got, split_step=h.action_steps, **tac_kw(c))
        hook.remove()
    assert [float(t[0]) for t in seen] == [flow_time(i).item() for i in range(h.action_steps, 10)]
    assert all(t.dtype == torch.float32 for t in seen) and h.action_steps + len(seen) == 10
    assert out.shape == (B, 16, 21) and torch.isfinite(out).all()
    assert flow_time(10).item() == 0. and flow_time(0).item() == 1.


# ── cache ladder ────────────────────────────────────────────────────────────

@torch.no_grad()
def test_ladder_matches_single_split_runs_and_is_immutable():
    m, c = tiny(), ctx()
    with AC:
        lad = m.forward_flow_action_ladder(c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'],
                                           risk=c['rS'], risk_valid=c['vS'], slow_tick_ns=123)
        assert isinstance(lad, ActionCacheLadder) and lad.levels == (.6, .4, .2)
        assert lad.risk_S == pytest.approx((.3, .8)) and lad.risk_S_valid == (True, False) and lad.slow_tick_ns == 123
        prints = {tau: lad[tau].kv_fingerprint() for tau in lad.levels}
        for tau in lad.levels:
            snap = lad[tau]
            steps = snap.handover.action_steps
            x, cache, nact, got = partial(m, c, steps)
            assert snap.token_counts == {'latent': L, 'fast': NF, 'state': 0, 'risk': 1, 'time': 1, 'chunk': 16,
                                         'action': nact, 'total': L + nact}
            assert torch.equal(snap.x_tau, x)
            for a, b in zip(snap._cache.layers, cache.layers):
                assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
            md = snap.cache_metadata
            assert (md['n_risk'], md['action_steps'], md['tau']) == (1, steps, tau)
            assert torch.equal(snap.position_ids[..., -nact:],
                               m.model._extend_position_ids(c['pos'], nact, 0)[..., -nact:])
            ref = m.tactile_flow_continue(cache, c['pos'], nact, x, got, split_step=steps, **tac_kw(c))
            out = m.tactile_flow_continue_from_ladder(lad, tau, **tac_kw(c))
            assert torch.equal(ref, out)
        # Many fast ticks at every split with changing r_F / tactile: snapshots unchanged.
        outs = {}
        for k in range(3):
            for tau in (.2, .6, .4, .4):
                rF = torch.tensor([k / 3, 1 - k / 3])
                outs[(k, tau)] = m.tactile_flow_continue_from_ladder(lad, tau, **tac_kw(c, rF=rF))
        assert {tau: lad[tau].kv_fingerprint() for tau in lad.levels} == prints
        assert lad.risk_S == pytest.approx((.3, .8))              # r_F never rewrites r_S
        assert not torch.equal(outs[(0, .4)], outs[(2, .4)])      # but r_F does change the action
        assert len({tuple(outs[(0, t)].flatten()[:8].tolist()) for t in (.2, .4, .6)}) == 3
        # Snapshot exports are copies.
        x = lad[.4].x_tau
        x.add_(1.)
        assert lad[.4].kv_fingerprint() == prints[.4]
        with pytest.raises(ValueError):
            lad[.3]
        m.train()
        with pytest.raises(ValueError, match='eval'):
            m.forward_flow_action_ladder(c['slow'], c['pos'], c['noise'], risk=c['rS'], risk_valid=c['vS'])


@torch.no_grad()
def test_repeated_inference_is_deterministic():
    m, c = tiny(), ctx()
    with AC:
        runs = []
        for _ in range(2):
            lad = m.forward_flow_action_ladder(c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'],
                                               risk=c['rS'], risk_valid=c['vS'])
            runs.append(([lad[t].kv_fingerprint() for t in lad.levels],
                         [m.tactile_flow_continue_from_ladder(lad, t, **tac_kw(c)) for t in (.2, .4, .6)]))
        assert runs[0][0] == runs[1][0]
        assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))


# ── baseline regression and compatibility ──────────────────────────────────

def _base_module():
    src = subprocess.check_output(['git', 'show', f'{BASE}:qwen_vla/modeling_vla.py'], text=True, cwd=REPO)
    mod = types.ModuleType('qwen_vla._base_modeling_vla')
    mod.__package__ = 'qwen_vla'
    exec(compile(src, '<base:modeling_vla>', 'exec'), mod.__dict__)
    return mod


@torch.no_grad()
def test_risk_free_paths_are_bitwise_identical_to_base_commit():
    """Fixed tau=0.4 (and the other splits) with use_risk_tokens=False reproduce
    the base commit exactly: partial, continue, training step and full flow."""
    old = _base_module().Qwen3VLVLAModel
    m, c = tiny(risk=False), ctx()
    with AC:
        for steps in (6, 8, 4):
            xo, co, no, to = old.forward_flow_action_partial(m, c['slow'], c['pos'], c['noise'],
                                                             fast_embeds=c['fast'], split_step=steps)
            xn, cn, nn_, tn = partial(m, c, steps, risk=False)
            assert torch.equal(xo, xn) and (no, to) == (nn_, tn)
            for a, b in zip(co.layers, cn.layers):
                assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
            ao = old.tactile_flow_continue(m, co, c['pos'], no, xo, to, split_step=steps,
                                           tactile_f6=c['tac'], tactile_f6_history=c['hist'])
            an = m.tactile_flow_continue(cn, c['pos'], nn_, xn, tn, split_step=steps, **tac_kw(c, risk=False))
            assert torch.equal(ao, an)
            h = Handover(steps)
            xt, tt, _ = tactile_training_pair(c['noise'] * .2, c['noise'], torch.full((B,), .4), h)
            vo = old.tactile_flow_train_step(m, co, c['pos'], no, xt, tt, tactile_f6=c['tac'],
                                             tactile_f6_history=c['hist'])
            vn = m.tactile_flow_train_step(cn, c['pos'], nn_, xt, tt, **tac_kw(c, risk=False))
            assert torch.equal(vo, vn)
        fo = old.forward_flow_action_full(m, c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'])
        fn = m.forward_flow_action_full(c['slow'], c['pos'], c['noise'], fast_embeds=c['fast'])
        assert torch.equal(fo, fn)


def test_weight_schema_and_checkpoint_contract(tmp_path):
    old = _base_module().Qwen3VLVLAModel
    kw = dict(action_dim=21, action_chunk=16, tacf6_dim=3, tactile_intermediate_size=64, use_tactile_code=True,
              vqvae_codebook_size=16, use_tactile_vqvae=True, vqvae_config=VQ.to_dict())
    with torch.device('meta'):
        s_old = {k: tuple(v.shape) for k, v in old(text_config(), **kw).state_dict().items()}
        s_new = {k: tuple(v.shape) for k, v in Qwen3VLVLAModel(text_config(), **kw).state_dict().items()}
        s_risk = {k: tuple(v.shape) for k, v in Qwen3VLVLAModel(text_config(), use_risk_tokens=True,
                                                                 **kw).state_dict().items()}
    assert s_old == s_new                                         # baseline schema untouched
    extra = set(s_risk) - set(s_old)
    assert set(s_old) <= set(s_risk) and extra and all(k.startswith('risk_embedder_') for k in extra)
    assert {k.split('.')[0] for k in extra} == {'risk_embedder_action', 'risk_embedder_tactile'}
    # vanilla checkpoint -> OURS model: everything loads, new risk params listed.
    vanilla, ours = tiny(risk=False, seed=3), tiny(risk=True, seed=4)
    logs = []
    rep = load_with_contract(ours, vanilla.state_dict(), {'risk_tokens': False}, log=logs.append)
    assert set(rep['new_risk_parameters_initialized']) == extra and not rep['unexpected']
    assert any('NEW risk-token parameters' in line for line in logs)
    for k, v in vanilla.state_dict().items():
        assert torch.equal(ours.state_dict()[k], v)
    # OURS checkpoint -> vanilla model: refused (never silently drop trained weights).
    with pytest.raises(ValueError, match='risk'):
        load_with_contract(tiny(risk=False), ours.state_dict(), ours.architecture_contract(), log=logs.append)
    with pytest.raises(ValueError, match='risk'):
        load_with_contract(tiny(risk=False), ours.state_dict(), None, log=logs.append)   # inferred
    bad = dict(ours.state_dict())
    bad['risk_embedder_action.mlp.0.weight'] = torch.zeros(H, 5)
    with pytest.raises(ValueError, match='risk parameter'):
        load_with_contract(tiny(risk=True), bad, ours.architecture_contract(), log=logs.append)
    with pytest.raises(ValueError, match='carries no risk'):
        load_with_contract(tiny(risk=True), vanilla.state_dict(), ours.architecture_contract(), log=logs.append)
    rep2 = load_with_contract(tiny(risk=True, seed=9), ours.state_dict(), ours.architecture_contract(), log=logs.append)
    assert not rep2['missing'] and not rep2['new_risk_parameters_initialized']


def test_variant_guard():
    from types import SimpleNamespace as NS
    validate_variant(NS(variant='vanilla', cascaded_split_mode='fixed', cascaded_split_step=6, use_risk_tokens=0,
                        tactile_delay_ms=[0]))
    validate_variant(NS(variant='ours', cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6],
                        use_risk_tokens=1, risk_stats='r.json', tactile_delay_ms=[0, 50, 100, 150]))
    for bad in (NS(variant='vanilla', cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6]),
                NS(variant='multisplit', cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6], use_risk_tokens=1),
                NS(variant='ours', cascaded_split_mode='multi', cascaded_splits=[.4], use_risk_tokens=1, risk_stats='x'),
                NS(variant='ours', cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6], use_risk_tokens=0)):
        with pytest.raises(ValueError, match='variant'):
            validate_variant(bad)


# ── inference engine (slow/fast ticks, gate, immutable ladder) ──────────────

def _stream(root, ep):
    from taptic.data import Source
    from taptic.force_history import build_force_track, read_force_samples
    from taptic.risk import samples_from_sidecar
    sm, av = read_force_samples(root, ep)
    src = Source(root)
    frames = src.clocks(ep)
    tr = build_force_track(sm, av, frames)
    ok = np.flatnonzero(tr['segment'] >= 0)
    return samples_from_sidecar(sm.iloc[ok[0]:], av[ok[0]:]), frames


@pytest.fixture(scope='module')
def synth_root(tmp_path_factory):
    from synthetic_root import build
    return build(tmp_path_factory.mktemp('taptic_policy'))


@torch.no_grad()
@pytest.mark.parametrize('risk_model', [True, False])
def test_policy_slow_fast_ticks_all_splits(synth_root, risk_model):
    from taptic.data import Source
    from taptic.ours_policy import RiskAdaptivePolicy
    from taptic.risk import RiskConfig, fit_risk_stats
    from taptic.risk_gate import GateConfig
    from tactile_vqvae.data.stats import TacF6Stats
    stats = fit_risk_stats(Source(synth_root), [0, 1, 2, 3], RiskConfig(), log=None)
    force_stats = TacF6Stats(-60 * np.ones(15, np.float32), 60 * np.ones(15, np.float32), np.ones(15, bool))
    samples, frames = _stream(synth_root, 0)
    m, c = tiny(risk=risk_model), ctx()
    one = {k: (v[:1] if torch.is_tensor(v) and v.dim() and v.shape[0] == B else v) for k, v in c.items()}
    one['pos'] = c['pos'][:, :1]
    results = {}
    for mode in ('fixed:0.2', 'fixed:0.4', 'fixed:0.6', 'adaptive'):
        if mode == 'adaptive':
            gate = GateConfig.from_risk_stats(stats, hysteresis=.01, min_dwell_ms=20)
        else:
            gate = GateConfig(mode='fixed', fixed_split=float(mode.split(':')[1]))
        pol = RiskAdaptivePolicy(m, stats, gate, force_stats)
        pol.start_episode()
        pol.ingest(samples)
        out = []
        with AC:
            for f in (5, 6):
                S = int(frames[f])
                info = pol.slow_tick(S, inputs_embeds=one['slow'], position_ids=one['pos'], noise=one['noise'],
                                     fast_embeds=one['fast'])
                lad_print = [pol.ladder[t].kv_fingerprint() for t in (.2, .4, .6)]
                for k, dt in enumerate(range(0, 100, 13)):
                    r = pol.fast_tick(S + dt * 1_000_000)
                    assert r['norm_actions'].shape == (1, 16, 21) and torch.isfinite(r['norm_actions']).all()
                    assert r['risk_S'] == info['risk_S']['risk'] and r['slow_ns'] == S
                    out.append(r)
                assert [pol.ladder[t].kv_fingerprint() for t in (.2, .4, .6)] == lad_print
        results[mode] = out
        if mode.startswith('fixed'):
            assert {r['gate']['selected_split'] for r in out} == {float(mode.split(':')[1])}
    # Fixed 0.4 engine output == explicit ladder continuation at 0.4 (same r_F, same tactile).
    assert all(r['status'] == 'ok' for r in results['fixed:0.4'])
    a = [r['norm_actions'] for r in results['fixed:0.4']]
    b = [r['norm_actions'] for r in results['fixed:0.2']]
    assert not all(torch.equal(x, y) for x, y in zip(a, b))
    ad = results['adaptive']
    assert all(r['gate']['risk'] == r['risk_F'] or not r['risk_F_valid'] for r in ad)
    assert all(r['first_unelapsed_slot'] in (0, 1) for r in ad)
    # Determinism: replaying the same stream reproduces identical chunks and decisions.
    pol = RiskAdaptivePolicy(m, stats, GateConfig.from_risk_stats(stats, hysteresis=.01, min_dwell_ms=20),
                             force_stats)
    pol.start_episode()
    pol.ingest(samples)
    again = []
    with AC:
        for f in (5, 6):
            S = int(frames[f])
            pol.slow_tick(S, inputs_embeds=one['slow'], position_ids=one['pos'], noise=one['noise'],
                          fast_embeds=one['fast'])
            again += [pol.fast_tick(S + dt * 1_000_000) for dt in range(0, 100, 13)]
    assert all(torch.equal(x['norm_actions'], y['norm_actions']) and x['gate'] == y['gate']
               for x, y in zip(ad, again))


# ── launchers -> real argument parser and guards ────────────────────────────

def _parse_launcher(script, env, tmp_path):
    """Run a launcher with a stub `accelerate`, then parse its midtrain.py flags
    with the real parser/validators (train() replaced by an args dump)."""
    import json as _json
    import os
    out = subprocess.run(['bash', '-c', 'accelerate() { printf "%s\\n" "$@"; }; export -f accelerate; bash ' + script],
                         env={**os.environ, **env}, capture_output=True, text=True, cwd=REPO, check=True).stdout
    lines = out.split('\n')
    argv = [a for a in lines[lines.index('midtrain.py') + 1:] if a and not a.startswith('>>>')]
    src = (REPO / 'scripts' / 'midtrain.py').read_text()
    assert src.rstrip().endswith('train(args)')
    probe = tmp_path / 'midtrain_probe.py'
    probe.write_text(src.rstrip()[:-len('train(args)')] + 'print("ARGS_JSON" + json.dumps(vars(args), default=str))\n')
    code = subprocess.run([sys.executable, str(probe), *argv], capture_output=True, text=True, cwd=REPO / 'scripts',
                          env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert code.returncode == 0, code.stderr[-2000:]
    line = next(l for l in code.stdout.splitlines() if l.startswith('ARGS_JSON'))
    return _json.loads(line[len('ARGS_JSON'):])


def test_launchers_parse_and_vanilla_matches_the_launched_baseline(tmp_path):
    env = dict(TAPTIC_ROOT=str(tmp_path / 'data'), MODEL_PATH='m', VQVAE_CKPT='vq.pt', OUTPUT_DIR=str(tmp_path / 'o'),
               RUN_NAME='r', EXPERIMENT_NAME='e')
    base = _parse_launcher('scripts/midtrain_taptic.sh', env, tmp_path)
    got = {v: _parse_launcher('scripts/midtrain_taptic_ours.sh', {**env, 'VARIANT': v}, tmp_path)
           for v in ('vanilla', 'multisplit', 'ours')}
    ignore = {'variant'}
    assert {k: v for k, v in got['vanilla'].items() if k not in ignore} == \
        {k: v for k, v in base.items() if k not in ignore}
    assert got['ours']['use_risk_tokens'] == 1 and got['ours']['cascaded_split_mode'] == 'multi'
    assert got['ours']['tactile_delay_ms'] == [0, 50, 100, 150] == got['multisplit']['tactile_delay_ms']
    assert got['multisplit']['use_risk_tokens'] == 0 and got['multisplit']['risk_stats'] == got['ours']['risk_stats']
    same = {k for k in base if k not in ('variant', 'cascaded_split_mode', 'tactile_delay_ms', 'risk_stats',
                                          'use_risk_tokens', 'risk_future_head_weight', 'risk_future_probe',
                                          'risk_dropout', 'risk_oversample')}
    assert (got['ours']['risk_future_head_weight'], got['ours']['risk_future_probe'], got['ours']['risk_dropout'],
            got['ours']['risk_oversample']) == (0.1, 1, 0.1, 2.0)
    assert got['multisplit']['risk_future_probe'] == 1 and got['multisplit']['risk_future_head_weight'] == 0
    assert got['vanilla']['risk_future_probe'] == 0 and got['vanilla']['risk_oversample'] == 1.0
    for k in same:   # identical budget, data, init, optimization across variants
        assert got['ours'][k] == got['multisplit'][k] == got['vanilla'][k], k


def test_save_checkpoint_writes_contract_and_reloads(tmp_path):
    """The trainer's own save_checkpoint records architecture + risk definition;
    the saved pair reloads through the contract (and is refused by a baseline model)."""
    import json as _json
    from types import SimpleNamespace as NS
    import midtrain as trainer
    from qwen_vla.contract import architecture_from_training_args
    m = tiny(risk=True, seed=5)
    stats = tmp_path / 'risk.json'
    stats.write_text('{"version": 1}')
    accel = NS(is_main_process=True, get_state_dict=lambda mod: mod.state_dict(), wait_for_everyone=lambda: None,
               unwrap_model=lambda mod: mod)
    proc = NS(save_pretrained=lambda p: Path(p).mkdir(parents=True, exist_ok=True))
    args = NS(output_dir=str(tmp_path), max_ckpts=0, model_path=str(tmp_path / 'none'), data_format='taptic',
              taptic_tactile_input='sdk_force', taptic_stats='stats.json', tactile_delay_ms=[0, 50, 100, 150],
              risk_stats=str(stats), risk_coefficients=[.25, .5, .25], risk_rate_window_ms=100,
              risk_velocity_window_ms=100, risk_joint_source='sidecar', tactile_stale_ms=50., variant='ours',
              use_risk_tokens=1, risk_n_frequencies=8, action_dim=21, action_chunk=16, use_robot_state=0,
              use_tactile_deform=0, use_tactile_vec=1, tactile_intermediate_size=64, training_stage=2, use_flare=0,
              n_flare_tokens_per_frame=0, n_flare_steps=0, flare_layer_index=-1, use_tactile_vqvae=1,
              vqvae_config_dict=VQ.to_dict(), cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6],
              flow_time_precision='fp32', tactile_legacy_stale_ms=150., tactile_gap_ms=50.)
    trainer.save_checkpoint(m, proc, accel, args, 0, 7, {'taptic': {}})
    d = tmp_path / 'checkpoint-0-7'
    ta = _json.loads((d / 'training_args.json').read_text())
    assert ta['architecture'] == m.architecture_contract() and ta['variant'] == 'ours'
    assert ta['tactile_delay_ms'] == [0, 50, 100, 150] and ta['risk_definition']['coefficients'] == [.25, .5, .25]
    assert ta['risk_stats_sha256'] and len(ta['risk_stats_sha256']) == 64
    sd = torch.load(d / 'model.pt', map_location='cpu')
    fresh = tiny(risk=True, seed=6)
    rep = load_with_contract(fresh, sd, architecture_from_training_args(ta), log=lambda *a: None)
    assert not rep['missing'] and not rep['unexpected'] and not rep['new_risk_parameters_initialized']
    assert all(torch.equal(fresh.state_dict()[k], v) for k, v in sd.items())
    with pytest.raises(ValueError, match='risk'):
        load_with_contract(tiny(risk=False), sd, architecture_from_training_args(ta), log=lambda *a: None)
    # Pre-contract checkpoints (no "architecture" block) are read as risk-free.
    assert architecture_from_training_args({'model_path': 'x'})['risk_tokens'] is False
