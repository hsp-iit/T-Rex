"""TAPTIC native SDK-force history, VQ-VAE and sdk_force midtrain input.

CPU-only, synthetic root with the real export schema (tests/taptic/synthetic_root.py).
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_root import EPISODES, build  # noqa: E402
from taptic.data import Source, fit_stats
from taptic.force_history import (build_force_track, history_at, read_force_samples,
                                  training_window_ends, window_indices)
from tactile_vqvae.data.taptic import build_taptic_datasets, fit_force_stats, load_tracks
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig

REPO = Path(__file__).resolve().parents[2]
W = 16


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    return build(tmp_path_factory.mktemp('taptic'))


@pytest.fixture(scope='module')
def source(root):
    return Source(root)


def track(source, ep):
    sm, av = read_force_samples(source.root, ep)
    return build_force_track(sm, av, source.clocks(ep)), sm, av


def test_track_excludes_preroll_tail_pause_and_splits_segments(source):
    for ep in (0, 1, 2, 3):
        tr, sm, _ = track(source, ep)
        frames = source.clocks(ep)
        read, seg = tr['read_ns'], tr['segment']
        assert not sm.columns.str.contains('values').any()          # taxels never loaded
        ok = seg >= 0
        assert read[ok].max() <= frames[-1]
        # Only the latest pre-roll sample (initial causal observation) is kept.
        assert (read[ok] <= frames[0]).sum() == 1
        if ep == 1:      # reset_sequence change -> new segment
            r = sm.reset_sequence.to_numpy()
            j = int(np.flatnonzero(np.diff(r))[0]) + 1
            assert seg[j] != seg[j - 1]
        if ep == 2:      # 80 ms read gap -> new segment
            j = int(np.flatnonzero(np.diff(read) > 50_000_000)[0]) + 1
            assert seg[j] != seg[j - 1]
        if ep == 3:      # recorder pause samples excluded, history restarts after it
            inside = np.flatnonzero((read > frames[19]) & (read < frames[20]))
            # All pause samples dropped except the latest causal one at the resuming
            # frame, which opens the new segment (same rule as episode start).
            assert len(inside) > 1 and (seg[inside[:-1]] == -1).all()
            assert seg[inside[-1]] >= 0
            assert seg[inside[-1]] not in set(seg[read <= frames[19]])
            assert seg[inside[-1]] == seg[inside[-1] + 1]
            for tick in frames:
                assert history_at(tr, int(tick), W, 50_000_000)[1]
        assert len(np.unique(seg[ok])) == {0: 1, 1: 2, 2: 2, 3: 2}[ep]


def test_history_is_causal_native_rate_and_padded_within_segment(source):
    tr, sm, av = track(source, 1)
    frames = source.clocks(1)
    for t, tick in enumerate(frames):
        hist, valid, n_real, age = history_at(tr, int(tick), W, 50_000_000)
        assert hist.shape == (W, 5, 3) and valid
        end = int(np.flatnonzero((tr['read_ns'] <= tick) & (tr['available_ns'] <= tick))[-1])
        idx, n2 = window_indices(tr['segment'], end, W)
        assert n2 == n_real and np.array_equal(hist, tr['force'][idx])
        assert (tr['read_ns'][idx] <= tick).all()                    # no future samples
        assert len(set(tr['segment'][idx])) == 1                     # never crosses a reset
        real = idx[W - n_real:]
        assert np.array_equal(np.diff(real), np.ones(n_real - 1))   # consecutive native reads
        if n_real < W:
            assert (idx[:W - n_real] == real[0]).all()                # left pad = first real sample
        # Native rate: ~13 ms spacing, i.e. ~7.6 samples per 100 ms frame.
        if n_real > 1:
            assert np.allclose(np.diff(tr['read_ns'][real]), 13_090_000)
    assert history_at(tr, int(frames[0]), W, 50_000_000)[2] == 1    # episode start: only initial sample
    # Stale -> invalid and zero.
    hist, valid, _, _ = history_at(tr, int(frames[-1]) + 10 ** 9, W, 50_000_000)
    assert not valid and not hist.any()


def test_vq_dataset_native_only_split_and_train_only_stats(root, source):
    train, val, stats, meta = build_taptic_datasets(str(root), window=W, stride=1, log=lambda *a: None)
    native_train = [e for e, (s, n, _) in EPISODES.items() if s == 'train' and n]
    native_val = [e for e, (s, n, _) in EPISODES.items() if s == 'val' and n]
    assert train.episodes == native_train and val.episodes == native_val
    assert meta['input_signal'] == 'taptic_sdk_force' and meta['source_identity'] == source.identity
    # Stats from training samples only.
    ref = fit_force_stats(load_tracks(source, native_train, log=None))
    assert np.allclose(stats.tacf6_min, ref.tacf6_min) and np.allclose(stats.tacf6_max, ref.tacf6_max)
    x = train[0]
    assert x['f6'].shape == (W, 5, 3) and x['f6'].abs().max() <= 1
    b = train.collate_fn([train[i] for i in range(4)])
    assert b['f6'].shape == (4, W, 5, 3)
    # Every usable sample is a window end once (stride 1); padded starts included.
    n_usable = sum(int((train.tracks[e]['segment'] >= 0).sum()) for e in train.episodes)
    assert len(train) == n_usable
    assert len(training_window_ends(train.tracks[0]['segment'], 1, W, pad_short=False)) == n_usable_ep(train, 0) - (W - 1)


def n_usable_ep(ds, ep):
    return int((ds.tracks[ep]['segment'] >= 0).sum())


@pytest.mark.parametrize('granularity', ['finger', 'hand'])
def test_vqvae_shapes_for_force(granularity):
    cfg = TactileVQVAEConfig(window=W, in_channels=15, per_finger_dim=3, n_hands=1, codebook_size=16,
                             hidden_channels=16, bottleneck_channels=32, embed_dim=16, granularity=granularity,
                             input_signal='taptic_sdk_force')
    m = TactileVQVAE(cfg)
    x = torch.randn(3, W, 5, 3).clamp(-1, 1)
    out = m(x, torch.ones(3))
    assert out['recon'].shape == x.shape
    out['total_loss'].backward()
    assert m.encode(x).shape == ((3, 5) if granularity == 'finger' else (3,))
    assert cfg.stats_dim == 15 and TactileVQVAEConfig().stats_dim == 60


def test_vqvae_train_smoke_writes_taptic_checkpoint(root, tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2'}
    cmd = [sys.executable, '-m', 'tactile_vqvae.train', '--data_format', 'taptic', '--lerobot_root', str(root),
           '--output_dir', str(tmp_path), '--run_name', 'smoke', '--granularity', 'finger',
           '--codebook_size', '16', '--embed_dim', '16', '--hidden_channels', '16', '--bottleneck_channels', '32',
           '--batch_size', '8', '--num_workers', '0', '--mixed_precision', 'no', '--weight_tau', '0',
           '--smoke_test', '1']
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    blob = torch.load(tmp_path / 'smoke/latest.pt', weights_only=False)
    assert blob['config']['input_signal'] == 'taptic_sdk_force'
    assert (blob['config']['n_hands'], blob['config']['per_finger_dim']) == (1, 3)
    assert len(blob['stats']['tacf6_min']) == 15 and blob['data']['weight_tau'] > 0
    assert blob['data']['train_episodes'] == [0, 1, 2, 3]


class _Holder(torch.nn.Module):
    """Minimal carrier for Qwen3VLVLAModel.encode_tactile_f6_history (no Qwen weights)."""
    def __init__(self, cfg, stats_min, stats_max):
        super().__init__()
        self.use_tactile_vqvae = True
        self.tactile_vqvae = TactileVQVAE(cfg).eval()
        self.register_buffer('tacf6_vqvae_min', torch.as_tensor(stats_min, dtype=torch.float32))
        self.register_buffer('tacf6_vqvae_max', torch.as_tensor(stats_max, dtype=torch.float32))
        self.register_buffer('tacf6_vqvae_mask', torch.ones(len(stats_min), dtype=torch.bool))


def test_vla_encoder_matches_vqvae_and_keeps_original_f6_path():
    from qwen_vla.modeling_vla import Qwen3VLVLAModel
    from tactile_vqvae.data.stats import TacF6Stats
    enc = Qwen3VLVLAModel.encode_tactile_f6_history
    small = dict(window=W, codebook_size=16, hidden_channels=16, bottleneck_channels=32, embed_dim=16,
                 granularity='finger')
    cfg = TactileVQVAEConfig(per_finger_dim=3, n_hands=1, in_channels=15, **small)
    lo, hi = -np.arange(15, dtype=np.float32) - 5, np.arange(15, dtype=np.float32) + 5
    h = _Holder(cfg, lo, hi)
    raw = torch.randint(-20, 20, (2, W, 5, 3)).float()
    st = TacF6Stats(lo, hi, np.ones(15, bool))
    ref = h.tactile_vqvae.encode(torch.from_numpy(st.normalize(raw.numpy())))
    assert torch.equal(enc(h, raw), ref)
    with pytest.raises(ValueError):
        enc(h, torch.zeros(2, W, 10, 6))
    # Original bimanual F6 layout still works.
    h6 = _Holder(TactileVQVAEConfig(**small), -np.ones(60), np.ones(60))
    assert enc(h6, torch.zeros(2, W, 10, 6)).shape == (2, 10)


def _vq_blob(path, root, source):
    train, _, stats, meta = build_taptic_datasets(str(root), window=W, stride=4, log=lambda *a: None)
    cfg = TactileVQVAEConfig(window=W, per_finger_dim=3, n_hands=1, in_channels=15, codebook_size=16,
                             hidden_channels=16, bottleneck_channels=32, embed_dim=16, granularity='finger',
                             input_signal='taptic_sdk_force')
    torch.save({'config': cfg.to_dict(), 'stats': stats.to_dict(), 'data': meta,
                'model_state': TactileVQVAE(cfg).state_dict()}, path)
    return stats


class _FakeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, *a, **k):
        return 'x'

    def __call__(self, text, images, return_tensors, padding):
        return SimpleNamespace(input_ids=torch.ones(1, 4, dtype=torch.long),
                               pixel_values=torch.zeros(len(images), 4),
                               image_grid_thw=torch.ones(len(images), 3, dtype=torch.long))


def _cfg(root, **kw):
    base = dict(lerobot_root=str(root), action_dim=21, action_chunk=16, use_tactile_vec=1,
                use_tactile_deform=0, use_tactile_vqvae=1, use_tactile_code=1, vqvae_window=W,
                use_robot_state=0, use_flare=0, image_size=None, tactile_delay_offsets=[0],
                cascaded_total_steps=10, cascaded_split_step=6, taptic_tactile_input='sdk_force')
    base.update(kw)
    return SimpleNamespace(**base)


def test_midtrain_loader_sdk_force_history_and_mask(root, source, tmp_path):
    from qwen_vla.taptic_dataset import TRexTapticDataset
    stats = _vq_blob(tmp_path / 'vq.pt', root, source)
    cfg = _cfg(root, vqvae_ckpt=str(tmp_path / 'vq.pt'))
    ds = TRexTapticDataset(cfg, _FakeProcessor(), None, _stats=fit_stats(source))
    ds._img = lambda *a: np.zeros((8, 8, 3), np.uint8)
    seen = {}
    for i, (ep, t) in enumerate(ds._rows):
        if ep in seen and t < 5:
            continue
        x = ds[i]
        seen[ep] = x
        assert x['tactile_f6'].shape == (5, 3) and x['force_history'].shape == (W, 5, 3)
        if EPISODES[ep][1]:
            assert x['force_valid']
            tick = x['meta']['observation_ns']
            tr = build_force_track(*read_force_samples(root, ep), source.clocks(ep))
            ref, _, _, _ = history_at(tr, tick, W, 50_000_000)
            assert np.array_equal(x['force_history'], ref)
            assert np.allclose(x['tactile_f6'], stats.normalize(ref[-1]))
        else:                                   # legacy 10 Hz: missing, not upsampled
            assert not x['force_valid'] and not x['force_history'].any() and not x['tactile_f6'].any()
    assert set(seen) == {0, 1, 2, 3, 4}
    batch = ds.collate_fn([seen[0], seen[4], seen[1]])
    assert batch['tactile_f6s'].shape == (3, 5, 3)
    assert batch['tactile_f6_history'].shape == (3, W, 5, 3)
    assert batch['tactile_loss_mask'].tolist() == [True, False, True]
    # Without the embedded VQ-VAE: current force vector only, still masked.
    ds2 = TRexTapticDataset(_cfg(root, use_tactile_vqvae=0, use_tactile_code=0,
                                 tactile_f6_stats_ckpt=str(tmp_path / 'vq.pt')),
                            _FakeProcessor(), None, _stats=fit_stats(source))
    ds2._img = ds._img
    b2 = ds2.collate_fn([ds2[0]])
    assert b2['tactile_f6_history'] is None and b2['tactile_loss_mask'] is not None


def test_config_guards(root, source, tmp_path):
    from qwen_vla.taptic_dataset import (TRexTapticDataset, validate_config, validate_vqvae_config,
                                         tactile_feature_dim)
    with pytest.raises(ValueError):
        validate_config(_cfg(root, taptic_tactile_input='taxels'))           # VQ needs sdk_force
    with pytest.raises(ValueError):
        validate_config(_cfg(root, use_tactile_vqvae=0))                       # no code sidecars
    with pytest.raises(ValueError):
        validate_vqvae_config(TactileVQVAEConfig().to_dict())                  # original F6 VQ-VAE
    assert tactile_feature_dim(_cfg(root)) == 3
    assert tactile_feature_dim(_cfg(root, taptic_tactile_input='taxels')) == 360
    # VQ-VAE statistics from another dataset are rejected.
    _vq_blob(tmp_path / 'vq.pt', root, source)
    blob = torch.load(tmp_path / 'vq.pt', weights_only=False)
    blob['data']['source_identity'] = 'other'
    torch.save(blob, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='different dataset'):
        TRexTapticDataset(_cfg(root, vqvae_ckpt=str(tmp_path / 'bad.pt')), _FakeProcessor(), None,
                          _stats=fit_stats(source))


def test_masked_tactile_loss():
    sys.path.insert(0, str(REPO / 'scripts'))
    from midtrain import tactile_flow_loss
    pred = torch.randn(3, 16, 21, requires_grad=True)
    tgt = torch.randn(3, 16, 21)
    assert torch.allclose(tactile_flow_loss(pred, tgt), torch.nn.MSELoss()(pred, tgt))
    m = torch.tensor([True, False, True])
    ref = ((pred - tgt) ** 2).mean(dim=(1, 2))[m].mean()
    assert torch.allclose(tactile_flow_loss(pred, tgt, m), ref)
    zero = tactile_flow_loss(pred, tgt, torch.zeros(3, dtype=torch.bool))
    zero.backward()
    assert zero.item() == 0 and pred.grad is not None and not pred.grad.any()


def test_tactile_observation_tokens_codes_plus_force(root, source, tmp_path):
    """New embedding path without the Qwen backbone: 5 VQ code tokens + 5 force tokens."""
    from qwen_vla.modeling_vla import Qwen3VLVLAModel
    from diffusion import ActionEmbedder
    _vq_blob(tmp_path / 'vq.pt', root, source)
    blob = torch.load(tmp_path / 'vq.pt', weights_only=False)
    cfg = TactileVQVAEConfig.from_dict(blob['config'])
    h = _Holder(cfg, blob['stats']['tacf6_min'], blob['stats']['tacf6_max'])
    h.tactile_vqvae.load_state_dict(blob['model_state'])
    h.use_tactile_code, h.use_tactile_deform = True, False
    h.tactile_code_embedder = torch.nn.Embedding(16, 64)
    h.tacf6_embedder = ActionEmbedder(3, 64)
    h.encode_tactile_f6_history = Qwen3VLVLAModel.encode_tactile_f6_history.__get__(h)
    hist = torch.randint(-20, 20, (2, W, 5, 3)).float()
    vec = torch.randn(2, 5, 3)
    obs = Qwen3VLVLAModel._embed_tactile_observations(h, vec, None, 'cpu', torch.float32,
                                                       tactile_f6_history=hist)
    assert obs.shape == (2, 10, 64)
    codes = h.encode_tactile_f6_history(hist)
    assert torch.equal(obs[:, :5], h.tactile_code_embedder(codes))
    obs.sum().backward()
    assert h.tacf6_embedder.mlp.fc1.weight.grad is not None
    assert all(p.grad is None for p in h.tactile_vqvae.parameters())


@pytest.mark.skipif(int(__import__('transformers').__version__.split('.')[0]) >= 5,
                    reason='Qwen3VLModelMoT rotary wrapper targets the pinned transformers 4.57')
def test_tiny_mot_tactile_step_with_force_and_vq_codes(root, source, tmp_path):
    """Real MoT tactile path: 5 VQ code tokens + 5 force tokens, masked loss, grads."""
    from transformers import Qwen3VLConfig
    from qwen_vla import Qwen3VLVLAModel
    from qwen_vla.flow_matching import Handover, tactile_training_pair
    sys.path.insert(0, str(REPO / 'scripts'))
    from midtrain import reinit_tactile_expert, tactile_flow_loss
    torch.manual_seed(0)
    _vq_blob(tmp_path / 'vq.pt', root, source)
    blob = torch.load(tmp_path / 'vq.pt', weights_only=False)
    t = Qwen3VLConfig().text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 1
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.rope_scaling = {'rope_type': 'default', 'mrope_section': [6, 5, 5], 'mrope_interleaved': True}
    t._attn_implementation = 'eager'
    model = Qwen3VLVLAModel(t, action_dim=21, action_chunk=16, tacf6_dim=3, tactile_intermediate_size=64,
                            use_tactile_code=True, vqvae_codebook_size=16, use_tactile_vqvae=True,
                            vqvae_config=blob['config']).float()
    model.initialize_vla_weights()
    model.load_tactile_vqvae(blob)
    assert model.tacf6_vqvae_min.shape == (15,)
    reinit_tactile_expert(model)                                   # the trainer's own init
    norms = {n: p for n, p in model.named_parameters() if '_tactile' in n and 'norm' in n}
    assert norms and all(torch.equal(p, torch.ones_like(p)) for p in norms.values())
    from qwen_vla.taptic_dataset import TRexTapticDataset
    ds = TRexTapticDataset(_cfg(root, vqvae_ckpt=str(tmp_path / 'vq.pt')), _FakeProcessor(), None,
                           _stats=fit_stats(source))
    ds._img = lambda *a: np.zeros((8, 8, 3), np.uint8)
    native = next(i for i, (e, tt) in enumerate(ds._rows) if e == 0 and tt == 10)
    legacy = next(i for i, (e, tt) in enumerate(ds._rows) if e == 4)
    b = ds.collate_fn([ds[native], ds[legacy]])
    B, L, H = 2, 6, 64
    slow = torch.randn(B, L, H)
    pos = torch.arange(L).view(1, 1, L).expand(3, B, L)
    fast = torch.randn(B, 3, H)
    h = Handover(6)
    # Tactile RMSNorm gains start at 1, so the force and code tokens get a gradient
    # from the very first step (with zero gains the branch was off at step 0).
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for it in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            _, cache, nact, tau = model.forward_flow_action_partial(
                slow, pos, b['eps_r'].float(), attention_mask=torch.ones(B, L, dtype=torch.long),
                fast_embeds=fast, split_step=6)
            codes = model.encode_tactile_f6_history(b['tactile_f6_history'])
            assert codes.shape == (2, 5)
            obs = model._embed_tactile_observations(b['tactile_f6s'], None, 'cpu', torch.bfloat16,
                                                    tactile_f6_history=b['tactile_f6_history'])
            assert obs.shape == (2, 10, H)                   # 5 code tokens + 5 force tokens
            xt, tt, target = tactile_training_pair(b['norm_actions'], b['eps_r'], b['time_r'], h)
            v = model.tactile_flow_train_step(cache, pos, nact, xt, tt, tactile_f6=b['tactile_f6s'],
                                              tactile_f6_history=b['tactile_f6_history'])
            loss = tactile_flow_loss(v, target, b['tactile_loss_mask'])
        assert torch.isfinite(loss)
        loss.backward()
        grads = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        if it == 0:
            assert any(n.startswith('tacf6_embedder.') for n in grads), sorted(grads)
            assert any(n.startswith('tactile_code_embedder.') for n in grads), sorted(grads)
        opt.step()
    assert any(n.startswith('tacf6_embedder.') for n in grads), sorted(grads)
    assert any(n.startswith('tactile_code_embedder.') for n in grads), sorted(grads)
    assert not any(n.startswith('tactile_vqvae.') for n in grads)       # frozen
    # Only the native sample drives the loss: legacy row's velocity has no gradient path.
    v2 = v.detach().clone().requires_grad_(True)
    tactile_flow_loss(v2, target, b['tactile_loss_mask']).backward()
    assert v2.grad[0].abs().sum() > 0 and not v2.grad[1].any()
