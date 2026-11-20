"""Contact-risk features, train-only statistics, online/offline equivalence and gate.

CPU only. Synthetic tracks are checked against an independent brute-force
reference; real-data checks run when TAPTIC_ROOT is set.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_root import EPISODES, build  # noqa: E402
from taptic.data import Source
from taptic.force_history import build_force_track, read_force_samples
from taptic.risk import (OnlineRiskExtractor, RiskConfig, RiskStats, combine_risk, fit_risk_stats,
                         joint_track_from_frames, risk_features, samples_from_sidecar)
from taptic.risk_gate import GateConfig, RiskGate, raw_bin

MS = 1_000_000
CFG = RiskConfig()                     # rate_mode='vector' (default)
SCALES = {'vector': (500., 400., 350., 350., 350.), 'positive': (400., 300., 300., 300., 300.),
          'negative': (380., 290., 310., 300., 280.)}
STATS = RiskStats((50., 40., 30., 30., 30.), SCALES['vector'], 1.5, 100, 100, 'sidecar', rate_scales=SCALES)
CFG_POS = RiskConfig(rate_mode='positive')
STATS_POS = RiskStats((50., 40., 30., 30., 30.), SCALES['positive'], 1.5, 100, 100, 'sidecar',
                      rate_mode='positive', rate_scales=SCALES)


def reference(track, tick, stats=STATS, cfg=CFG):
    """Loop implementation of the module docstring definitions (independent code)."""
    read, avail, seg, F, q = (track['read_ns'], track['available_ns'], track['segment'],
                              track['force'].astype(np.float64), track['joints'])
    out = dict(M=0., R=0., V=0., r=0., R_pos=0., R_neg=0., M_valid=False, R_valid=False, V_valid=False)
    cand = [i for i in range(len(read)) if read[i] <= tick and avail[i] <= tick]
    if not cand:
        return out
    e = cand[-1]
    if seg[e] < 0 or tick - read[e] > cfg.stale_ns:
        return out
    m = lambda i: np.linalg.norm(F[i], axis=1)
    out['M'] = float(np.max(np.minimum(m(e) / np.array(stats.magnitude_scale), 1)))
    out['M_valid'] = True
    first = e
    while first > 0 and seg[first - 1] == seg[e]:
        first -= 1
    W = cfg.rate_window_ns
    if read[first] <= tick - W:
        u = {k: np.zeros(5) for k in ('vector', 'positive', 'negative')}
        for i in range(first + 1, e + 1):
            if read[i] > tick - W:
                u['positive'] += np.maximum(m(i) - m(i - 1), 0)
                u['negative'] += np.maximum(m(i - 1) - m(i), 0)
                u['vector'] += np.linalg.norm(F[i] - F[i - 1], axis=1)
        norm = lambda kind, scale: float(np.max(np.minimum(u[kind] / (W / 1e9) / np.array(scale), 1)))
        out['R'] = norm(cfg.rate_mode, stats.rate_scale)
        out['R_pos'] = norm('positive', SCALES['positive'])
        out['R_neg'] = norm('negative', SCALES['negative'])
        out['R_valid'] = True
    ks = [k for k in range(first, e) if read[k] <= read[e] - cfg.velocity_window_ns]
    if ks:
        k = ks[-1]
        speed = np.linalg.norm(q[e] - q[k]) / ((read[e] - read[k]) / 1e9)
        out['V'] = float(min(speed / stats.speed_scale, 1))
        out['V_valid'] = True
    if out['M_valid'] and out['R_valid'] and out['V_valid']:
        a, b, c = cfg.coefficients
        out['r'] = float(np.clip(a * out['M'] + b * out['R'] + c * out['M'] * out['V'], 0, 1))
    return out


def synthetic_track(seed=0, n=120):
    rng = np.random.default_rng(seed)
    dt = rng.integers(11 * MS, 16 * MS, n)                       # irregular native reads
    read = 10_000 * MS + np.cumsum(dt)
    read[60:] += 70 * MS                                         # > 50 ms gap -> new segment
    avail = read + rng.integers(0, 2 * MS, n)
    avail = np.maximum.accumulate(avail)
    seg = np.ones(n, np.int64)
    seg[60:] = 2
    seg[90] = -1                                                 # excluded sample
    seg[91:] = 3
    force = np.cumsum(rng.integers(-4, 6, (n, 5, 3)), 0).astype(np.float32)
    joints = np.cumsum(rng.normal(0, 2e-3, (n, 12)), 0)
    return {'read_ns': read, 'available_ns': avail, 'segment': seg, 'force': force, 'joints': joints}


@pytest.mark.parametrize('mode', ['vector', 'positive'])
def test_features_match_independent_reference_and_formula(mode):
    stats, cfg = (STATS, CFG) if mode == 'vector' else (STATS_POS, CFG_POS)
    tr = synthetic_track()
    ticks = np.unique(np.concatenate([tr['read_ns'], tr['available_ns'], tr['read_ns'] + 7 * MS,
                                      tr['read_ns'] + 60 * MS, [tr['read_ns'][0] - MS]]))
    got = risk_features(tr, ticks, stats, cfg)
    n_valid = 0
    for j, t in enumerate(ticks):
        ref = reference(tr, int(t), stats, cfg)
        for k in ('M_valid', 'R_valid', 'V_valid'):
            assert bool(got[k][j]) == ref[k], (k, j)
        for k in ('M', 'R', 'V', 'R_pos', 'R_neg'):
            assert got[k][j] == pytest.approx(ref[k], abs=1e-12), (k, j)
        assert got['risk'][j] == pytest.approx(ref['r'], abs=1e-12)
        n_valid += ref['M_valid'] and ref['R_valid'] and ref['V_valid']
    assert n_valid > 50 and (~got['risk_valid']).sum() > 20      # both regimes exercised
    np.testing.assert_allclose(combine_risk(.4, .2, .5), .25 * .4 + .5 * .2 + .25 * .4 * .5)
    assert combine_risk(1, 1, 1) == 1.                          # clipped
    v = got['risk_valid']
    np.testing.assert_allclose(got['risk'][v], np.clip(.25 * got['M'][v] + .5 * got['R'][v]
                                                       + .25 * got['M'][v] * got['V'][v], 0, 1))


def _two_finger_track(force):
    n = len(force)
    read = 1_000 * MS + np.arange(n) * 13 * MS
    return {'read_ns': read, 'available_ns': read.copy(), 'segment': np.ones(n, np.int64),
            'force': np.asarray(force, np.float32), 'joints': np.zeros((n, 12))}


def test_vector_R_sees_decreases_and_direction_changes():
    """Release (|F| falling) and a rotation of F at constant |F| (shear / slip
    candidate) are invisible to the previous positive-only R and visible to the
    vector R; both are logged separately as R_pos / R_neg."""
    n = 30
    hold = np.zeros((n, 5, 3)); hold[:, 0, 2] = 80.
    release = hold.copy(); release[20:, 0, 2] = np.linspace(80, 0, 10)          # thumb lets go
    turn = hold.copy()
    ang = np.r_[np.zeros(20), np.linspace(0, np.pi / 3, 10)]                      # same |F|, new direction
    turn[:, 0, 0], turn[:, 0, 2] = 80 * np.sin(ang), 80 * np.cos(ang)
    tick = np.array([1_000 * MS + (n - 1) * 13 * MS])
    for name, force in (('hold', hold), ('release', release), ('turn', turn)):
        tr = _two_finger_track(force)
        vec = risk_features(tr, tick, STATS, CFG)
        pos = risk_features(tr, tick, STATS_POS, CFG_POS)
        assert vec['R_valid'][0] and pos['R_valid'][0]
        if name == 'hold':                                   # plateau: no variation of any kind
            assert vec['R'][0] == pos['R'][0] == vec['R_neg'][0] == 0
        if name == 'release':
            assert pos['R'][0] == 0 and vec['R_pos'][0] == 0      # previous R: blind to it
            assert vec['R'][0] > 0 and vec['R_neg'][0] > 0
            assert vec['rate_vector'][0, 0] == pytest.approx(vec['rate_negative'][0, 0])   # pure |F| change
        if name == 'turn':
            assert pos['R'][0] == pytest.approx(0, abs=1e-4) and vec['R_neg'][0] == pytest.approx(0, abs=1e-4)
            assert vec['R'][0] > .1                          # direction change at constant |F|
    # Triangle inequality: vector variation >= |F| variation, always.
    tr = synthetic_track()
    f = risk_features(tr, tr['available_ns'], STATS, CFG)
    # per read pair ||dF|| >= | d|F| | = (positive + negative) increment, hence per window:
    assert np.all(f['rate_vector'] + 1e-6 >= f['rate_positive'] + f['rate_negative'])
    assert (f['rate_vector'] > f['rate_positive'] + f['rate_negative'] + 1e-6).any()   # strictly, somewhere


def test_statistics_and_config_must_agree_on_R_mode():
    with pytest.raises(ValueError, match='R mode'):
        risk_features(synthetic_track(), np.array([0]), STATS_POS, CFG)
    with pytest.raises(ValueError, match='version'):
        RiskStats.from_dict({**STATS.to_dict(), 'version': 1})            # v1 = positive-only R files
    again = RiskStats.from_dict(json.loads(json.dumps(STATS_POS.to_dict())))
    assert again == STATS_POS and again.rate_scales == STATS_POS.rate_scales
    with pytest.raises(ValueError):
        RiskStats((50.,) * 5, SCALES['positive'], 1.5, 100, 100, 'sidecar', rate_scales=SCALES)  # mismatch


def test_validity_is_explicit_and_never_zero_risk():
    tr = synthetic_track()
    read = tr['read_ns']
    # Before any sample, stale, inside a fresh segment (window not covered), excluded sample.
    ticks = np.array([read[0] - 1, read[59] + 51 * MS, read[60] + 5 * MS, tr['available_ns'][90] + 0])
    f = risk_features(tr, ticks, STATS, CFG)
    assert not f['M_valid'][0] and not f['R_valid'][0] and f['force_age_ns'][0] == -1
    assert not f['M_valid'][1] and f['force_age_ns'][1] > 50 * MS      # stale, not "no contact"
    assert f['M_valid'][2] and not f['R_valid'][2] and not f['V_valid'][2]   # new segment
    assert not f['M_valid'][3]                                          # excluded sample
    assert not f['risk_valid'].any() and not f['risk'].any()
    # No native stream at all (legacy): invalid everywhere.
    none = risk_features(None, ticks, STATS, CFG)
    assert not none['risk_valid'].any() and (none['force_age_ns'] == -1).all()


def test_future_samples_never_change_features():
    tr = synthetic_track()
    for cut in (30, 75, 110):
        tick = int(tr['available_ns'][cut]) + 3 * MS
        base = risk_features(tr, np.array([tick]), STATS, CFG)
        fut = {k: v.copy() for k, v in tr.items() if k != '_risk_derived'}
        later = fut['available_ns'] > tick
        fut['force'][later] = 999.
        fut['joints'][later] = 7.
        alt = risk_features(fut, np.array([tick]), STATS, CFG)
        for k in ('risk', 'M', 'R', 'V', 'risk_valid', 'force_age_ns', 'joint_age_ns'):
            assert np.array_equal(base[k], alt[k]), k
        # A sample READ before the tick but AVAILABLE after it is not used either.
        late = {k: v.copy() for k, v in tr.items() if k != '_risk_derived'}
        i = int(np.searchsorted(late['read_ns'], tick, side='right') - 1)
        late['available_ns'][i] = tick + 1
        late['available_ns'][i + 1:] = np.maximum(late['available_ns'][i + 1:], tick + 1)
        lf = risk_features(late, np.array([tick]), STATS, CFG)
        assert lf['source_index'][0] < i


def test_joint_velocity_uses_actual_timestamps_and_exposes_age():
    n = 60
    dt = np.r_[0, np.tile([9, 17, 13], n)[:n - 1]] * MS          # strongly irregular reads
    read = 5_000 * MS + np.cumsum(dt)
    v_true = np.linspace(.1, .5, 12)                             # rad/s per joint
    joints = 0.2 + (read - read[0])[:, None] / 1e9 * v_true[None]
    tr = {'read_ns': read, 'available_ns': read.copy(), 'segment': np.ones(n, np.int64),
          'force': np.ones((n, 5, 3), np.float32), 'joints': joints}
    tick = int(read[-1]) + 4 * MS
    f = risk_features(tr, np.array([tick]), STATS, CFG)
    assert f['V_valid'][0]
    assert f['joint_speed'][0] == pytest.approx(np.linalg.norm(v_true), rel=1e-9)
    assert f['joint_age_ns'][0] == 4 * MS
    assert f['velocity_span_ns'][0] >= 100 * MS
    # 10 Hz fallback stream: latest frame by the tick, age exposed, no interpolation.
    frames = read[::8]
    state = np.zeros((len(frames), 40))
    state[:, 7:19] = joints[::8]
    jt = joint_track_from_frames(frames, state)
    cfg10 = RiskConfig(joint_source='state10hz')
    st10 = RiskStats(STATS.magnitude_scale, STATS.rate_scale, 1.5, 100, 100, 'state10hz')
    tick = int(frames[-2]) + 60 * MS
    g = risk_features(tr, np.array([tick]), st10, cfg10, joint_track=jt)
    assert g['V_valid'][0] and g['joint_age_ns'][0] == 60 * MS
    assert g['joint_speed'][0] == pytest.approx(np.linalg.norm(v_true), rel=1e-9)
    stale = risk_features(tr, np.array([int(frames[-2]) + 151 * MS]), st10, cfg10, joint_track=jt)
    with pytest.raises(ValueError):
        risk_features(tr, np.array([tick]), STATS, cfg10, joint_track=jt)   # stats/config mismatch
    assert not stale['V_valid'][0] or stale['joint_age_ns'][0] <= 150 * MS


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    return build(tmp_path_factory.mktemp('taptic_risk'))


def test_fit_is_train_only_and_checks_provenance(root):
    src = Source(root)
    train = [e for e, (sp, _, _) in EPISODES.items() if sp == 'train']
    st = fit_risk_stats(src, train, CFG, log=None)
    natives = [e for e in train if EPISODES[e][1]]
    assert list(st.train_episodes) == natives and st.source_identity == src.identity
    assert len(st.magnitude_scale) == 5 and st.speed_scale > 0 and st.gate_thresholds[0] < st.gate_thresholds[1]
    st.check_source(src, train)
    with pytest.raises(ValueError, match='training'):
        fit_risk_stats(src, train + [5], CFG, log=None)          # validation episode
    with pytest.raises(ValueError):
        st.check_source(src, [0])                                # episodes outside the split
    other = RiskStats.from_dict({**st.to_dict(), 'source_identity': 'x'})
    with pytest.raises(ValueError, match='different dataset'):
        other.check_source(src, train)
    again = RiskStats.from_dict(json.loads(json.dumps(st.to_dict())))
    assert again == st


def _online_vs_offline(src_root, ep, stats, n_ticks=400, seed=0):
    sm, av = read_force_samples(src_root, ep)
    src = Source(src_root)
    frames = src.clocks(ep)
    tr = build_force_track(sm, av, frames)
    ok = np.flatnonzero(tr['segment'] >= 0)
    first = int(ok[0])
    pause = np.flatnonzero(np.diff(frames) > 150 * MS)
    last = int(ok[-1]) if not len(pause) else int(np.searchsorted(tr['read_ns'], frames[pause[0]]))
    ex = OnlineRiskExtractor(stats, CFG)
    ex.ingest(samples_from_sidecar(sm.iloc[first:last + 1], av[first:last + 1]))
    rng = np.random.default_rng(seed)
    ticks = np.sort(rng.integers(int(tr['read_ns'][first]), int(tr['read_ns'][last]), n_ticks))
    off = risk_features(tr, ticks, stats, CFG)
    n_valid = 0
    for j, t in enumerate(ticks):
        on = ex.at(int(t))
        for k in ('risk', 'M', 'R', 'V'):
            assert on[k] == pytest.approx(float(off[k][j]), abs=1e-12), (ep, k, j)
        for k in ('risk_valid', 'M_valid', 'R_valid', 'V_valid'):
            assert on[k] == bool(off[k][j]), (ep, k, j)
        n_valid += on['risk_valid']
    return n_valid


def test_online_extractor_matches_offline_synthetic(root):
    src = Source(root)
    st = fit_risk_stats(src, [0, 1, 2, 3], CFG, log=None)
    for ep in (0, 1, 2):          # plain, sensor reset, read gap
        assert _online_vs_offline(root, ep, st) > 50


@pytest.mark.skipif(not os.environ.get('TAPTIC_ROOT'), reason='set TAPTIC_ROOT for real-data checks')
def test_online_extractor_matches_offline_real():
    root = os.environ['TAPTIC_ROOT']
    src = Source(root)
    e = src.episodes
    native_train = e[(e.suggested_split == 'train') & ~e.tactile_interpolated.astype(bool)]
    eps = [int(g.index[0]) for _, g in native_train.groupby('source_task')][:4]   # one per task
    st = fit_risk_stats(src, eps, CFG, log=None)
    for ep in eps[:3] + [int(e[(e.suggested_split == 'val') & ~e.tactile_interpolated.astype(bool)].index[0])]:
        assert _online_vs_offline(root, ep, st, n_ticks=300) > 200


@pytest.mark.skipif(not os.environ.get('TAPTIC_ROOT'), reason='set TAPTIC_ROOT for real-data checks')
def test_real_sidecar_joint_stream_is_measured_native_rate():
    """The native sidecar joint column is a genuine ~76 Hz measured stream that equals
    observation.state[7:19] at frame samples; legacy rows are 10 Hz frame samples only."""
    root = os.environ['TAPTIC_ROOT']
    src = Source(root)
    for ep, native in ((0, True), (553, True), (300, False)):
        sm, _ = src.tactile(ep)
        J = np.stack(sm.joints.to_numpy())
        s = np.stack(src.frames(ep)['observation.state'])
        fs = sm[sm.is_frame_sample.to_numpy(bool)]
        np.testing.assert_array_equal(np.stack(fs.joints.to_numpy()), s[fs.frame_index.to_numpy(), 7:19])
        dt = np.diff(sm.read_end_ns.to_numpy()) / 1e6
        if native:
            assert 12 < np.median(dt) < 14.5
            assert not np.all(np.diff(J, axis=0) == 0, axis=1).any()     # never a held copy
        else:
            assert np.median(dt) == pytest.approx(100, abs=1) and len(sm) == len(s)


# ── gate ────────────────────────────────────────────────────────────────────

def test_gate_monotone_fixed_and_fallback():
    g = RiskGate(GateConfig(mode='adaptive', thresholds=(.1, .3)))
    rs = np.linspace(0, 1, 201)
    picks = [g.select(i, r, True)['selected_split'] for i, r in enumerate(rs)]
    assert all(a <= b for a, b in zip(picks, picks[1:]))       # monotone in r
    assert picks[0] == .2 and picks[-1] == .6 and .4 in picks
    assert [raw_bin(r, (.1, .3)) for r in (0, .0999, .1, .2999, .3, 1)] == [0, 0, 1, 1, 2, 2]
    f = RiskGate(GateConfig(mode='fixed'))
    assert {f.select(i, r, True)['selected_split'] for i, r in enumerate(rs)} == {.4}
    assert f.select(999, None, False)['selected_split'] == .4
    a = RiskGate(GateConfig(mode='adaptive', thresholds=(.1, .3)))
    a.select(0, .9, True)
    d = a.select(1, float('nan'), True)
    assert d['selected_split'] == .4 and d['reason'] == 'missing_risk_fallback' and d['transition']
    assert a.select(2, .9, False)['reason'] == 'missing_risk_fallback'
    with pytest.raises(ValueError):
        a.select(1, .5, True)                                    # time goes backwards
    with pytest.raises(ValueError):
        GateConfig(mode='adaptive', thresholds=(.3, .1))
    with pytest.raises(ValueError, match='unreachable'):
        GateConfig(mode='adaptive', thresholds=(.0094, .1675), hysteresis=.01)


def test_gate_hysteresis_and_dwell():
    cfg = GateConfig(mode='adaptive', thresholds=(.1, .3), hysteresis=.05)
    g = RiskGate(cfg)
    seq = [.0, .12, .149, .151, .12, .06, .049, .2]
    out = [g.select(i * MS, r, True)['selected_split'] for i, r in enumerate(seq)]
    assert out == [.2, .2, .2, .4, .4, .4, .2, .4]
    d = RiskGate(GateConfig(mode='adaptive', thresholds=(.1, .3), min_dwell_ms=50))
    t = [0, 10, 40, 49, 50, 60, 70]
    r = [.0, .5, .5, .5, .5, .0, .0]
    res = [d.select(ti * MS, ri, True) for ti, ri in zip(t, r)]
    assert [x['selected_split'] for x in res] == [.2, .2, .2, .2, .6, .6, .6]
    assert [x['reason'] for x in res] == ['risk'] + ['dwell_hold'] * 3 + ['risk'] + ['dwell_hold'] * 2
    assert sum(x['transition'] for x in res) == 1
    # Fallback overrides dwell; recovery right after it is immediate.
    d.select(71 * MS, None, False)
    assert d.select(72 * MS, .5, True)['selected_split'] == .6
    s = d.summary()
    assert s['ticks'] == 9 and s['fallbacks'] == 1 and s['dwell_holds'] == 5


def test_gate_thresholds_from_training_stats(root):
    st = fit_risk_stats(Source(root), [0, 1, 2, 3], CFG, gate_quantiles=(.5, .85), log=None)
    g = GateConfig.from_risk_stats(st, hysteresis=0.)
    assert g.mode == 'adaptive' and g.thresholds[0] >= 1e-3 and g.thresholds[0] < g.thresholds[1]
    assert st.gate_quantiles == (.5, .85)
