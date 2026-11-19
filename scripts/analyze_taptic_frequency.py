"""Causal grid comparison on native recordings, no source writes or interpolation.

Thresholds below define diagnostic force-count events, not contact/risk labels.
Ten grid phases per rate expose phase-dependent short-transient retention.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

RATES = [76, 70, 65, 60, 50]


def quant(x):
    a = np.asarray(x)
    return dict(zip(['min', 'p01', 'p50', 'p95', 'p99', 'max'],
                    np.quantile(a, [0, .01, .5, .95, .99, 1]).tolist())) if a.size else {}


def segments(root, ep):
    sm = pd.read_parquet(root / f'tactile_raw/episode_{ep:06d}_right.parquet')
    fr = pd.read_parquet(root / f'tactile_raw/episode_{ep:06d}_frames.parquet')
    if 'interpolated' in sm:
        sm = sm[~sm.interpolated]
    t = sm.read_end_ns.to_numpy(np.int64)
    obs = fr.observation_ns.to_numpy(np.int64)
    keep = (t >= obs[0]) & (t <= obs[-1])
    # Exclude known recorder pauses, even if tactile continued through them.
    for i in np.flatnonzero(np.diff(obs) > 150_000_000):
        keep &= ~((t > obs[i]) & (t < obs[i+1]))
    idx = np.flatnonzero(keep)
    reset = sm.reset_sequence.to_numpy()
    errors, drops = sm.read_errors.to_numpy(), sm.dropped.to_numpy()
    cuts = (np.diff(t[idx]) > 50_000_000) | (np.diff(reset[idx]) != 0)
    cuts |= (np.diff(errors[idx]) != 0) | (np.diff(drops[idx]) != 0) | (np.diff(idx) != 1)
    chunks = np.split(idx, np.flatnonzero(cuts)+1)
    chunks = [c for c in chunks if len(c) >= 4]
    force = np.stack(sm.force).astype(np.float64)
    taxels = np.stack(sm['values']).astype(np.float64)
    return [(t[c], force[c], taxels[c]) for c in chunks], len(sm), int(keep.sum())


def events(t, f):
    # Count-norm peaks with >=5 count prominence; short half-prominence spans
    # <=50 ms. Span uses held native samples, never interpolated observations.
    env = np.linalg.norm(f.reshape(-1, 5, 3), axis=2)
    out = []
    for finger in range(5):
        y = env[:, finger]
        peaks, props = find_peaks(y, prominence=5.)
        for j, p in enumerate(peaks):
            cut = y[p] - props['prominences'][j] / 2
            lo, hi = p, p
            while lo > 0 and y[lo-1] >= cut:
                lo -= 1
            while hi+1 < len(y) and y[hi+1] >= cut:
                hi += 1
            if hi+1 < len(y) and t[hi+1]-t[lo] <= 50_000_000:
                out.append((finger, int(t[lo]), int(t[hi+1]), float(y[p])))
    # Single native increments >=5 count-norm increase; operational onset proxy.
    rise = np.argwhere(np.diff(env, axis=0) >= 5.)
    return env, out, rise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', nargs='+', type=int, default=[0, 48, 98, 197, 553, 554, 771])
    args = p.parse_args()
    root = Path(args.root)
    meta = pd.read_parquet(root / 'meta/episodes/chunk-000/file-000.parquet').set_index('episode_index')
    rows, native = [], []
    hashes = {}
    prepared = []
    for ep in args.episodes:
        if bool(meta.loc[ep, 'tactile_interpolated']):
            raise ValueError('frequency comparison requires native data')
        segs, all_n, used = segments(root, ep)
        dt, unchanged_f, unchanged_x, duration = [], 0, 0, 0.
        comparisons, n_peaks = 0, 0
        active_pairs, active_repeats, force_same_taxel_new = 0, 0, 0
        changed_dt = []
        for t, f, x in segs:
            duration += (t[-1]-t[0])/1e9
            dt.extend(np.diff(t)/1e6)
            unchanged_f += int(np.all(f[1:] == f[:-1], 1).sum())
            unchanged_x += int(np.all(x[1:] == x[:-1], 1).sum())
            comparisons += len(t)-1
            active = np.any(f != 0, axis=1)
            pairs = active[1:] & active[:-1]
            same_f, same_x = np.all(f[1:] == f[:-1],1), np.all(x[1:] == x[:-1],1)
            active_pairs += int(pairs.sum())
            active_repeats += int((same_f & pairs).sum())
            force_same_taxel_new += int((same_f & ~same_x).sum())
            change_times = t[np.r_[True, ~same_f]]
            changed_dt.extend(np.diff(change_times)/1e6)
            env, peak, rise = events(t, f)
            n_peaks += len(peak)
            prepared.append((ep, t, f, x, env, peak, rise))
        for suffix in ['right', 'frames']:
            path = root / f'tactile_raw/episode_{ep:06d}_{suffix}.parquet'
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        native.append({'episode': ep, 'task': meta.loc[ep, 'source_task'], 'all_recorded_rows': all_n,
            'kept_rows_before_min_segment_length': used, 'segments': len(segs), 'duration_s': duration,
            'read_interval_ms': quant(dt), 'comparisons': comparisons,
            'force_repeated_fraction': unchanged_f/comparisons, 'taxel_repeated_fraction': unchanged_x/comparisons,
            'force_change_events_per_s': (comparisons-unchanged_f)/duration,
            'taxel_change_events_per_s': (comparisons-unchanged_x)/duration,
            'active_force_pairs': active_pairs, 'active_force_repeat_fraction': active_repeats/active_pairs if active_pairs else None,
            'force_unchanged_taxels_changed_pairs': force_same_taxel_new,
            'force_payload_change_interval_ms': quant(changed_dt),
            'short_peaks': n_peaks})
    for hz in RATES:
        for phase in range(10):
            total, repeat_f, repeat_x, same_read = 0, 0, 0, 0
            ages, change_ages, deriv, native_deriv = [], [], [], []
            tv, native_tv = 0., 0.
            peak_ratios, rises_detected, rise_total = [], 0, 0
            selected_duration = 0.
            for ep, t, f, x, env, peak, rise in prepared:
                # Integer-ns timestamps from a regular mathematical grid.
                g = t[0] + np.rint((np.arange(int((t[-1]-t[0])*hz/1e9)+2) + phase/10) * 1e9/hz).astype(np.int64)
                g = g[(g >= t[0]) & (g <= t[-1])]
                ix = np.searchsorted(t, g, side='right')-1
                assert np.all(t[ix] <= g)
                if len(ix) < 2:
                    continue
                dt = np.diff(g)/1e9
                total += len(ix)-1
                repeat_f += int(np.all(f[ix[1:]] == f[ix[:-1]], 1).sum())
                repeat_x += int(np.all(x[ix[1:]] == x[ix[:-1]], 1).sum())
                same_read += int((np.diff(ix) == 0).sum())
                ages.extend((g-t[ix])/1e6)
                changed = np.r_[True, np.any(f[1:] != f[:-1], axis=1)]
                change_ix = np.maximum.accumulate(np.where(changed, np.arange(len(t)), 0))
                change_ages.extend((g-t[change_ix[ix]])/1e6)
                selected_duration += (g[-1]-g[0])/1e9
                # Norm-vector increments: total variation reveals missed reversals.
                sample_df = np.linalg.norm(np.diff(f[ix].reshape(-1,5,3), axis=0), axis=2)
                deriv.extend((sample_df / dt[:,None]).ravel())
                # Native reference restricted to same physical analysis span.
                # Include the held sample at the first grid tick, even when
                # its read preceded that tick by part of a period.
                nf = f[ix[0]:ix[-1]+1].reshape(-1,5,3)
                nt = t[ix[0]:ix[-1]+1]
                nd = np.linalg.norm(np.diff(nf, axis=0), axis=2)
                native_deriv.extend((nd / (np.diff(nt)/1e9)[:,None]).ravel())
                tv += float(sample_df.sum())
                native_tv += float(nd.sum())
                for finger, start, end, height in peak:
                    idx = ix[(g >= start) & (g < end)]
                    peak_ratios.append(float(env[idx, finger].max()/height) if len(idx) else 0.)
                for i, finger in rise:
                    # Preserve at least 90% of this reference rise amplitude
                    # at some grid interval crossing the event, within 50 ms.
                    candidates = np.flatnonzero((g[1:] >= t[i+1]) & (g[1:] <= t[i+1]+50_000_000) & (g[:-1] <= t[i+1]))
                    amplitude = env[i+1, finger]-env[i, finger]
                    rises_detected += int(any(env[ix[j+1],finger]-env[ix[j],finger] >= .9*amplitude for j in candidates))
                    rise_total += 1
            d = np.asarray(deriv); nd = np.asarray(native_deriv)
            rows.append({'hz': hz, 'phase_fraction': phase/10, 'comparisons': total,
                'force_repeat_fraction': repeat_f/total, 'taxel_repeat_fraction': repeat_x/total,
                'same_software_read_fraction': same_read/total,
                'force_changes_per_s': (total-repeat_f)/selected_duration,
                'mean_selected_age_ms': float(np.mean(ages)),
                'mean_held_output_age_ms': float(np.mean(ages)+500/hz),
                'measurement_age_ms': quant(ages), 'time_since_force_payload_change_ms': quant(change_ages),
                'short_peak_count': len(peak_ratios), 'short_peak_retained90_fraction': float(np.mean(np.asarray(peak_ratios)>=.9)),
                'short_peak_amplitude_ratio_mean': float(np.mean(peak_ratios)),
                'rise_proxy_count': rise_total, 'rise_proxy_retained90_fraction': rises_detected/rise_total,
                'derivative_counts_per_s': quant(d), 'derivative_rms': float(np.sqrt(np.mean(d*d))),
                'native_derivative_rms': float(np.sqrt(np.mean(nd*nd))),
                'derivative_rms_ratio': float(np.sqrt(np.mean(d*d)/np.mean(nd*nd))),
                'force_total_variation_ratio': tv/native_tv})
    aggregate = []
    for hz in RATES:
        group = [r for r in rows if r['hz'] == hz]
        a = {'hz': hz}
        for key in ['force_repeat_fraction','taxel_repeat_fraction','same_software_read_fraction',
                    'force_changes_per_s','short_peak_retained90_fraction','short_peak_amplitude_ratio_mean',
                    'rise_proxy_retained90_fraction','derivative_rms','derivative_rms_ratio','force_total_variation_ratio']:
            v = [r[key] for r in group]
            a[key] = {'mean': float(np.mean(v)), 'phase_min': min(v), 'phase_max': max(v)}
        a['mean_selected_age_ms'] = float(np.mean([r['mean_selected_age_ms'] for r in group]))
        a['mean_held_output_age_ms'] = float(np.mean([r['mean_held_output_age_ms'] for r in group]))
        for q in ['p50','p95','p99','max']:
            a['measurement_age_ms_'+q] = float(np.mean([r['measurement_age_ms'][q] for r in group]))
        aggregate.append(a)
    report = {'episodes': native, 'source_sha256': hashes, 'phases_per_rate': 10,
        'definitions': {'gap_ms': 50, 'recorder_pause_ms': 150, 'peak_prominence_counts': 5,
            'short_peak_half_prominence_span_ms': 50, 'rise_increment_counts': 5,
            'retained_amplitude_fraction': .9,
            'selection': 'latest recorded read_end_ns <= grid_ns; no interpolation; no deduplication',
            'derivative': 'norm(F_grid[k]-F_grid[k-1])/grid_dt per finger; counts/s, not physical risk',
            'warning': 'force changes are payload changes, not proven independent hardware acquisitions'},
        'aggregate': aggregate, 'per_phase': rows}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({'native': native, 'rates': aggregate}, indent=2))


if __name__ == '__main__':
    main()
