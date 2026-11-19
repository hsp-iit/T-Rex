"""Timestamp and quality metadata. Repeated values are retained measurements.

50 ms native / 150 ms legacy freshness limits are configurable engineering
limits, not sensor calibration or contact/risk thresholds. age_ns measures time
since the SDK read, not the unknown hardware acquisition time. History samples can
be old and still valid: freshness at the current anchor is a separate mask.
"""
import numpy as np


def build_quality(samples, available, native, gap_ns=None):
    read = samples.read_end_ns.to_numpy(np.int64)
    available = np.asarray(available, np.int64)
    gap_ns = (50_000_000 if native else 150_000_000) if gap_ns is None else gap_ns
    if gap_ns <= 0 or len(read) != len(available) or np.any(np.diff(read) <= 0) or np.any(available < read):
        raise ValueError('invalid tactile times or gap limit')
    breaks = np.zeros(len(read), bool)
    if len(read):
        breaks[0] = True
        breaks[1:] = np.diff(read) > gap_ns
    for key in ('reset_sequence', 'read_errors', 'dropped'):
        if key in samples:
            breaks[1:] |= np.diff(samples[key].to_numpy(np.int64)) != 0
    # Synthetic row insertion re-numbered legacy sequences. After excluding
    # interpolation their sequence gaps are not missing recorded measurements.
    if native and 'sequence' in samples:
        breaks[1:] |= np.diff(samples.sequence.to_numpy(np.int64)) != 1
    return {'read_end_ns': read, 'available_ns': available, 'break_before': breaks,
            'segment_id': np.cumsum(breaks).astype(np.int64)}


def latest_quality(quality, tick, first_tick, native, stale_ns=None):
    stale_ns = (50_000_000 if native else 150_000_000) if stale_ns is None else stale_ns
    if stale_ns <= 0:
        raise ValueError('freshness limit must be positive')
    read, available = quality['read_end_ns'], quality['available_ns']
    candidates = np.flatnonzero((read <= tick) & (available <= tick))
    initial = np.flatnonzero((read <= first_tick) & (available <= first_tick))
    lower = int(initial[-1]) if len(initial) else 0
    i = int(candidates[-1]) if len(candidates) else -1
    if i < lower:
        i = -1
    age = int(tick-read[i]) if i >= 0 else -1
    return {'source_index': i, 'age_ns': age,
            'availability_age_ns': int(tick-available[i]) if i >= 0 else -1,
            'fresh': i >= 0 and 0 <= age <= stale_ns,
            'segment_id': int(quality['segment_id'][i]) if i >= 0 else -1,
            'break_before': bool(quality['break_before'][i]) if i >= 0 else False}


def interval_quality(samples, quality, indices, tick, native, stale_ns=None):
    idx = np.asarray(indices, np.int64)
    stale_ns = (50_000_000 if native else 150_000_000) if stale_ns is None else stale_ns
    age = tick - quality['read_end_ns'][idx]
    segment = quality['segment_id'][idx]
    same_segment = segment == segment[-1] if len(idx) else np.empty(0, bool)
    repeated_f, repeated_x = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
    for j, i in enumerate(idx):
        if i > 0 and not quality['break_before'][i]:
            repeated_f[j] = np.array_equal(samples.iloc[i]['force'], samples.iloc[i-1]['force'])
            repeated_x[j] = native and np.array_equal(samples.iloc[i]['values'], samples.iloc[i-1]['values'])
    derivative_valid = np.zeros(len(idx), bool)
    if len(idx) > 1:
        derivative_valid[1:] = same_segment[1:] & same_segment[:-1] & ~quality['break_before'][idx[1:]] & (np.diff(idx) == 1)
    return {'source_index': idx.copy(), 'age_ns': age,
            'derivative_valid': derivative_valid,
            'availability_age_ns': tick-quality['available_ns'][idx],
            'fresh_at_anchor': (age >= 0) & (age <= stale_ns),
            'history_valid': same_segment,
            'break_before': quality['break_before'][idx].copy(), 'segment_id': segment.copy(),
            'force_unchanged_from_previous_read': repeated_f,
            'taxels_unchanged_from_previous_read': repeated_x}
