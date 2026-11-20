"""Real-data validation of the TAPTIC risk-conditioned adaptive T-Rex (CPU only).

  alignment     every k-th anchor of the selected episodes x every delay: F = S + d,
                slot masks, fresh tactile at F, and a TRUNCATION causality check —
                r_S / r_F / VQ history recomputed on a copy of the sensor track cut at
                S / F must equal what the training loader produced.
  gate_replay   OnlineRiskExtractor + RiskGate over every native read of held-out
                episodes, for several hysteresis/dwell settings (occupancy,
                transitions/s, fallbacks), plus one full diagnostic CSV log.
  model_replay  tiny random model, real processed context (images + caption):
                slow ticks build risk-conditioned ladders, fast ticks at native
                reads run gate -> snapshot -> Tactile Expert (plumbing only;
                random weights, no performance meaning).

python scripts/validate_taptic_ours.py {alignment,gate_replay,model_replay} --root ROOT --out DIR
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from taptic.data import Source  # noqa: E402
from taptic.force_history import build_force_track, history_at, read_force_samples  # noqa: E402
from taptic.risk import (OnlineRiskExtractor, RiskConfig, RiskStats, features_at, risk_features,  # noqa: E402
                         samples_from_sidecar)
from taptic.risk_gate import GateConfig, RiskGate  # noqa: E402

MS = 1_000_000
DELAYS = (0, 50, 100, 150)


def q(x):
    x = np.asarray(x, float)
    return {k: float(np.quantile(x, p)) for k, p in (('p50', .5), ('p99', .99), ('max', 1.))} if len(x) else None


def _vq_blob(src, block, path):
    from tactile_vqvae.data.taptic import fit_force_stats, load_tracks
    from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig
    natives = [e for e in block['train_episodes'] if not bool(src.episodes.loc[e, 'tactile_interpolated'])]
    cfg = TactileVQVAEConfig(window=16, per_finger_dim=3, n_hands=1, in_channels=15, codebook_size=16,
                             hidden_channels=16, bottleneck_channels=32, embed_dim=16, granularity='finger',
                             input_signal='taptic_sdk_force')
    torch.manual_seed(0)
    torch.save({'config': cfg.to_dict(), 'stats': fit_force_stats(load_tracks(src, natives, log=None)).to_dict(),
                'model_state': TactileVQVAE(cfg).state_dict(),
                'data': {'source_identity': src.identity, 'train_episodes': natives, 'window': 16}}, path)


def dataset(a, out, delays=DELAYS, processor=None):
    from qwen_vla.taptic_dataset import TRexTapticDataset
    src = Source(a.root)
    block = json.loads(Path(a.taptic_stats).read_text())
    vq = Path(a.work_dir) / 'random_vq_with_train_force_stats.pt'   # untracked helper
    if not vq.exists():
        _vq_blob(src, block['taptic'], vq)
    cfg = SimpleNamespace(lerobot_root=a.root, action_dim=21, action_chunk=16, use_tactile_vec=1,
                          use_tactile_deform=0, use_tactile_vqvae=1, use_tactile_code=1, vqvae_window=16,
                          vqvae_ckpt=str(vq), taptic_tactile_input='sdk_force', use_robot_state=0, use_flare=0,
                          image_size=(56, 56), tactile_delay_offsets=[0], tactile_delay_ms=list(delays),
                          risk_stats=a.risk_stats, use_risk_tokens=1, cascaded_total_steps=10,
                          cascaded_split_step=6, cascaded_split_mode='multi', cascaded_splits=[.2, .4, .6],
                          flow_time_precision='fp32', tactile_stale_ms=50., tactile_gap_ms=50.,
                          taptic_stats=a.taptic_stats)
    ds = TRexTapticDataset(cfg, processor, SimpleNamespace(print=print), _stats=block)
    return src, ds


def truncated(track, tick):
    n = int(np.searchsorted(track['available_ns'], tick, side='right'))
    return {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in track.items() if k != '_risk_derived'}


def alignment(a, out):
    src, ds = dataset(a, out)
    ds._img = lambda *x: np.zeros((8, 8, 3), np.uint8)
    e = src.episodes
    eps = sorted({ep for ep, _ in ds._rows})
    val = ds.create_val_split()
    val._img = ds._img
    val_eps = sorted({ep for ep, _ in val._rows})
    rng = np.random.default_rng(0)
    train_pick = sorted(rng.choice(eps, size=min(a.train_episodes, len(eps)), replace=False).tolist())
    stats = {d: {'items': 0, 'native': 0, 'tactile_valid': 0, 'risk_S_valid': 0, 'risk_F_valid': 0,
                 'supervised_slots': [], 'age_ms': [], 'new_reads': [], 'slot1_elapsed': 0, 'violations': 0}
             for d in DELAYS}
    violations, t0 = [], time.time()
    for split_ds, split_eps in ((val, val_eps), (ds, train_pick)):
        rows = {}
        for i, (ep, t) in enumerate(split_ds._rows):
            rows.setdefault(ep, []).append(i)
        for ep in split_eps:
            native = not bool(e.loc[ep, 'tactile_interpolated'])
            for idx in rows[ep][::a.stride]:
                for d in DELAYS:
                    split_ds.delays_ns = (d * MS,)
                    x = split_ds[idx]
                    _, _, _, times, _, _ = split_ds._episode(ep)
                    tr = split_ds._cache_force
                    m, t, st = x['meta'], x['meta']['frame'], stats[d]
                    S, F = m['slow_ns'], m['fast_ns']
                    bad = []
                    if S != int(times[t]) or F != S + d * MS:
                        bad.append('tick')
                    if not np.array_equal(x['tactile_slot_mask'], times[t:t + 16] >= F) or not x['tactile_slot_mask'].any():
                        bad.append('slot_mask')
                    st['items'] += 1
                    st['supervised_slots'].append(int(x['tactile_slot_mask'].sum()))
                    st['slot1_elapsed'] += int(not x['tactile_slot_mask'][1])
                    st['risk_S_valid'] += x['risk_S']['risk_valid']
                    st['risk_F_valid'] += x['risk_F']['risk_valid']
                    if native:
                        st['native'] += 1
                        for tag, tick in (('S', S), ('F', F)):     # causality by truncation
                            cut = truncated(tr, tick)
                            ref = features_at(risk_features(cut, np.array([tick]), ds.risk_stats, ds.risk_cfg), 0)
                            if ref != x['risk_' + tag]:
                                bad.append('risk_' + tag)
                        hist, valid, _, age = history_at(truncated(tr, F), F, 16, 50 * MS)
                        if not np.array_equal(hist, x['force_history']) or bool(valid) != x['force_valid']:
                            bad.append('history')
                        if x['force_valid']:
                            st['tactile_valid'] += 1
                            st['age_ms'].append(age / 1e6)
                            st['new_reads'].append(int(((tr['available_ns'] > S) & (tr['available_ns'] <= F)).sum()))
                    elif x['force_valid'] or x['risk_S']['risk_valid'] or x['risk_F']['risk_valid']:
                        bad.append('legacy_valid')
                    if bad:
                        st['violations'] += 1
                        violations.append({'episode': ep, 'frame': int(t), 'delay_ms': d, 'checks': bad})
        split_ds.delays_ns = tuple(dd * MS for dd in DELAYS)
    res = {'episodes': {'val': val_eps, 'train_sample': train_pick}, 'anchor_stride': a.stride,
           'seconds': round(time.time() - t0, 1), 'violations': violations[:50],
           'per_delay_ms': {str(d): {'items': v['items'], 'native_items': v['native'],
                                     'tactile_valid_frac_native': v['tactile_valid'] / max(v['native'], 1),
                                     'risk_S_valid_frac_native': v['risk_S_valid'] / max(v['native'], 1),
                                     'risk_F_valid_frac_native': v['risk_F_valid'] / max(v['native'], 1),
                                     'supervised_slots_mean': float(np.mean(v['supervised_slots'])),
                                     'supervised_slots_min': int(np.min(v['supervised_slots'])),
                                     'slot1_elapsed_frac': v['slot1_elapsed'] / max(v['items'], 1),
                                     'tactile_age_ms': q(v['age_ms']), 'new_native_reads_after_S': q(v['new_reads']),
                                     'violations': v['violations']} for d, v in stats.items()}}
    (out / 'alignment_evidence.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'episodes'}, indent=1))
    if violations:
        raise SystemExit(f'{len(violations)} alignment violations')


def gate_replay(a, out):
    src = Source(a.root)
    stats = RiskStats.load(a.risk_stats)
    e = src.episodes
    eps = sorted(int(i) for i in e.index[(e.suggested_split == 'val') & ~e.tactile_interpolated.astype(bool)])
    settings = {'no_hysteresis': dict(hysteresis=0., min_dwell_ms=0.),
                'h0.005_dwell50': dict(hysteresis=.005, min_dwell_ms=50.),
                'h0.005_dwell100': dict(hysteresis=.005, min_dwell_ms=100.)}
    agg = {k: {'ticks': 0, 'seconds': 0., 'transitions': 0, 'fallbacks': 0, 'occupancy': {'0.2': 0, '0.4': 0, '0.6': 0}}
           for k in settings}
    per_task = {}
    for n, ep in enumerate(eps):
        sm, av = read_force_samples(src.root, ep)
        frames = src.clocks(ep)
        tr = build_force_track(sm, av, frames)
        ok = np.flatnonzero(tr['segment'] >= 0)
        samples = samples_from_sidecar(sm.iloc[ok[0]:ok[-1] + 1], av[ok[0]:ok[-1] + 1])
        ticks = tr['available_ns'][ok]
        task = str(e.loc[ep, 'source_task'])
        for name, kw in settings.items():
            ex = OnlineRiskExtractor(stats, RiskConfig())
            ex.ingest(samples)
            g = RiskGate(GateConfig.from_risk_stats(stats, **kw), keep_log=True)
            for tk in ticks:
                f = ex.at(int(tk))
                g.select(int(tk), f['risk'], f['risk_valid'], f)
            s = g.summary()
            A = agg[name]
            A['ticks'] += s['ticks']
            A['seconds'] += (ticks[-1] - ticks[0]) / 1e9
            A['transitions'] += s['transitions']
            A['fallbacks'] += s['fallbacks']
            for k, v in s['occupancy'].items():
                A['occupancy'][k] += v
            if name == 'h0.005_dwell50':
                pt = per_task.setdefault(task, {'ticks': 0, 'seconds': 0., 'transitions': 0,
                                                'occupancy': {'0.2': 0, '0.4': 0, '0.6': 0}})
                pt['ticks'] += s['ticks']
                pt['seconds'] += (ticks[-1] - ticks[0]) / 1e9
                pt['transitions'] += s['transitions']
                for k, v in s['occupancy'].items():
                    pt['occupancy'][k] += v
                if n == 0:
                    g.write_log(out / f'gate_log_episode_{ep:06d}.csv')
    for A in list(agg.values()) + list(per_task.values()):
        A['transitions_per_s'] = A['transitions'] / max(A['seconds'], 1e-9)
        A['occupancy_frac'] = {k: v / max(A['ticks'], 1) for k, v in A['occupancy'].items()}
    res = {'episodes': eps, 'thresholds': list(GateConfig.from_risk_stats(stats).thresholds),
           'fast_ticks': 'every usable native read (~76 Hz)', 'settings': settings, 'aggregate': agg,
           'per_task_h0.005_dwell50': per_task}
    (out / 'gate_replay.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'episodes'}, indent=1))


def model_replay(a, out):
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vla import Qwen3VLVLAModel, split_slow_fast_embeds
    from taptic.ours_policy import RiskAdaptivePolicy
    sys.path.insert(0, str(REPO / 'scripts'))
    from midtrain import reinit_tactile_expert
    torch.set_num_threads(4)
    proc = AutoProcessor.from_pretrained(a.processor, local_files_only=True)
    src, ds = dataset(a, out, delays=(0,), processor=proc)
    val = ds.create_val_split()
    stats0 = RiskStats.load(a.risk_stats)
    thr = GateConfig.from_risk_stats(stats0).thresholds
    # First held-out native window of `slow_ticks` anchors whose measured r (offline,
    # same features) spans all three gate bins: exercises every split end to end.
    ep, rows = None, None
    for e_ in sorted({e for e, _ in val._rows if not bool(src.episodes.loc[e, 'tactile_interpolated'])}):
        r_ = [i for i, (e, t) in enumerate(val._rows) if e == e_]
        sm_, av_ = read_force_samples(src.root, e_)
        tr_ = build_force_track(sm_, av_, src.clocks(e_))
        tk = tr_['available_ns'][tr_['segment'] >= 0]
        f_ = risk_features(tr_, tk, stats0, RiskConfig())
        times_ = src.clocks(e_)
        for j in range(0, len(r_) - a.slow_ticks, 2):
            S0 = times_[val._rows[r_[j]][1]]
            sel = (tk >= S0) & (tk < S0 + a.slow_ticks * 100 * MS) & f_['risk_valid']
            bins = set(np.searchsorted(thr, f_['risk'][sel], side='right').tolist())
            if bins == {0, 1, 2}:
                ep, rows, a.first_anchor = e_, r_, j
                break
        if ep is not None:
            break
    if ep is None:
        raise SystemExit('no held-out window spans all gate bins')
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(a.processor, local_files_only=True)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    t.rope_scaling = {'rope_type': 'default', 'mrope_section': [6, 5, 5], 'mrope_interleaved': True}
    v.hidden_size, v.intermediate_size, v.depth, v.num_heads = 32, 64, 1, 2
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0]
    cfg._attn_implementation = 'eager'
    base = Qwen3VLForConditionalGeneration(cfg).float().eval()
    blob = torch.load(Path(a.work_dir) / 'random_vq_with_train_force_stats.pt', weights_only=False)
    results = {}
    for risk_tokens in (True, False):
        m = Qwen3VLVLAModel(t, action_dim=21, action_chunk=16, tacf6_dim=3, tactile_intermediate_size=64,
                            use_tactile_code=True, vqvae_codebook_size=16, use_tactile_vqvae=True,
                            vqvae_config=blob['config'], use_risk_tokens=risk_tokens).float()
        m.initialize_vla_weights()
        m.load_tactile_vqvae(blob)
        reinit_tactile_expert(m)
        torch.nn.init.normal_(m.final_layer.mlp.fc2.weight, std=.05)
        m.visual = base.model.visual
        object.__setattr__(m, '_rope_index_fn', base.model.get_rope_index)
        m.eval()
        stats = RiskStats.load(a.risk_stats)
        pol = RiskAdaptivePolicy(m, stats, GateConfig.from_risk_stats(stats, hysteresis=.005, min_dwell_ms=50),
                                 ds.force_stats, action_stats=ds.action_stats)
        sm, av = read_force_samples(src.root, ep)
        tr = build_force_track(sm, av, src.clocks(ep))
        ok = np.flatnonzero(tr['segment'] >= 0)
        pol.start_episode()
        pol.ingest(samples_from_sidecar(sm.iloc[ok[0]:ok[-1] + 1], av[ok[0]:ok[-1] + 1]))
        log, slow_ms, fast_ms = [], [], []
        g = torch.Generator().manual_seed(1)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            for idx in rows[a.first_anchor:a.first_anchor + a.slow_ticks]:
                b = val.collate_fn([val[idx]])
                emb = m.prepare_inputs_embeds(b['input_ids'], b['pixel_values'].float(), b['image_grid_thw'])
                slow, fast = split_slow_fast_embeds(emb, b['input_ids'], m.image_token_id,
                                                    int(b['image_grid_thw'][0].prod() // 4))
                pos, _ = m.get_rope_index(b['input_ids'], b['image_grid_thw'], b['attention_mask'])
                S = b['meta'][0]['slow_ns']
                info = pol.slow_tick(S, inputs_embeds=slow, position_ids=pos[:, :, :slow.shape[1]],
                                     noise=torch.randn(1, 16, 21, generator=g), fast_embeds=fast,
                                     attention_mask=b['attention_mask'])
                slow_ms.append(info['build_ms'])
                fast_ticks = tr['available_ns'][(tr['available_ns'] >= S) & (tr['available_ns'] < S + 100 * MS)]
                for F in fast_ticks:
                    r = pol.fast_tick(int(F))
                    fast_ms.append(r['fast_ms'])
                    log.append({'slow_ns': S, 'fast_ns': int(F), 'split': r['gate']['selected_split'],
                                'reason': r['gate']['reason'], 'risk_S': r['risk_S'], 'risk_F': r['risk_F'],
                                'risk_F_valid': r['risk_F_valid'], 'status': r['status'],
                                'finite': bool(np.isfinite(r['actions']).all())})
        results['ours' if risk_tokens else 'external_gate_no_risk_tokens'] = {
            'episode': ep, 'first_anchor_row': a.first_anchor, 'slow_ticks': len(slow_ms), 'fast_ticks': len(log),
            'gate_thresholds': list(thr), 'risk_F_range': [min(x['risk_F'] for x in log), max(x['risk_F'] for x in log)],
            'splits_used': sorted({x['split'] for x in log}),
            'transitions': pol.gate.summary()['transitions'],
            'all_finite': all(x['finite'] for x in log), 'statuses': sorted({x['status'] for x in log}),
            'cpu_tiny_model_ms': {'slow_ladder_build': q(slow_ms), 'fast_tick': q(fast_ms)},
            'note': 'tiny random weights on CPU: plumbing/latency of the toy model only, no performance claim'}
        if risk_tokens:
            (out / 'model_replay_log.json').write_text(json.dumps(log, indent=1))
    (out / 'model_replay.json').write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['alignment', 'gate_replay', 'model_replay'])
    p.add_argument('--root', required=True)
    p.add_argument('--taptic_stats', default=str(REPO / 'docs/taptic_train_stats.json'))
    p.add_argument('--risk_stats', default=str(REPO / 'docs/taptic_ours/risk_stats.json'))
    p.add_argument('--processor', default=os.environ.get('TAPTIC_PROCESSOR', ''))
    p.add_argument('--out', default=str(REPO / 'docs/taptic_ours'))
    p.add_argument('--work_dir', default=os.path.join(tempfile.gettempdir(), 'taptic_ours_validate'),
                   help='scratch dir for the random-weight VQ helper (needs only train force stats)')
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--train_episodes', type=int, default=24)
    p.add_argument('--slow_ticks', type=int, default=6)
    p.add_argument('--first_anchor', type=int, default=40)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    {'alignment': alignment, 'gate_replay': gate_replay, 'model_replay': model_replay}[a.cmd](a, out)


if __name__ == '__main__':
    main()
