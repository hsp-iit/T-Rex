"""Scientific feature checks; no network or production training changes."""
import numpy as np
import pytest

from scripts.analyze_taptic_risk import causal_features, contact_events, fit_normalization


def force(m):
    f = np.zeros((len(m), 5, 3))
    f[:, :, 0] = np.asarray(m)[:, None]
    return f


def test_irregular_actual_time_and_repeated_payloads():
    t = np.array([0, 10, 30, 50, 70, 100])*1_000_000
    f = causal_features(t, force([0, 1, 1, 3, 2, 2]), [True, False, False, False, False, False])
    np.testing.assert_allclose(f['signed_rate'][:, 0], [0, 100, 0, 100, -50, 0])
    assert f['derivative_valid'][2]  # Equal payload remains a valid zero derivative.
    assert f['negative_rate'][4, 0] == 50
    # Held integral [0,50ms]: 0*.01+1*.04 = .04; new endpoint is zero duration.
    assert f['windows'][50]['mean_magnitude'][3, 0] == pytest.approx(.8)
    assert f['windows'][50]['positive_variation_rate'][3, 0] == pytest.approx(60)


def test_rolling_peak_includes_value_held_from_before_left_edge():
    t = np.array([0, 20, 40, 60, 80])*1_000_000
    f = causal_features(t, force([10, 1, 1, 1, 1]), [True, False, False, False, False])
    # Window (10,60] contains the value 10 held until 20 ms.
    assert f['windows'][50]['peak_magnitude'][3, 0] == 10
    assert f['windows'][50]['mean_magnitude'][3, 0] == pytest.approx(2.8)


def test_prefix_causality_and_gap_resets():
    t = np.arange(40, dtype=np.int64)*10_000_000
    b = np.zeros(40, bool); b[[0, 25]] = True
    values = np.arange(40) % 8
    a = causal_features(t, force(values), b)
    prefix = causal_features(t[:23], force(values[:23]), b[:23])
    for ms in (50, 100, 200):
        for key, x in prefix['windows'][ms].items():
            np.testing.assert_allclose(x, a['windows'][ms][key][:23])
        assert not a['windows'][ms]['valid'][25]
    assert not a['derivative_valid'][25]
    assert np.all(a['signed_rate'][25] == 0)
    assert a['windows'][50]['valid'][30]


def test_activity_event_censoring_and_peak():
    t = np.arange(9, dtype=np.int64)*10_000_000
    m = np.tile([0, 0, 5, 8, 10, 6, 2, 0, 0], (5, 1)).T
    b = np.array([True]+[False]*8)
    ev = contact_events(t, m, b, np.full(5, 5.))
    assert len(ev) == 5
    assert ev[0]['peak'] == 10
    assert ev[0]['duration_s'] == pytest.approx(.04)
    assert ev[0]['rise_10_90_s'] == pytest.approx(.02)
    assert not ev[0]['left_censored'] and not ev[0]['right_censored']
    b[4] = True
    ev = contact_events(t, m, b, np.full(5, 5.))
    assert any(e['right_censored'] for e in ev)
    assert any(e['left_censored'] for e in ev)


def test_normalization_uses_train_only():
    t = np.arange(30, dtype=np.int64)*10_000_000
    def record(ep, split, scale):
        return dict(episode=ep, split=split, t=t, features=causal_features(
            t, force(scale*(np.arange(30)%5)), [True]+[False]*29))
    train = record(0, 'train', 1)
    expected = fit_normalization([train])
    assert fit_normalization([train, record(1, 'val', 1e9)]) == expected
    assert expected['train_episodes'] == [0]


def test_bad_force_and_clock_are_rejected():
    with pytest.raises(ValueError):
        causal_features([1, 1], force([0, 1]), [True, False])
    with pytest.raises(ValueError):
        causal_features([1, 2], force([0, np.nan]), [True, False])
