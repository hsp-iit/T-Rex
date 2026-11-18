"""Read-only audit of metadata/all action-state frames and sampled sidecars.

python -m taptic.inspect_dataset --root ROOT --out audit.json [--stats stats.json]
"""
import argparse
import json
import numpy as np
from .data import Source, fit_stats


def audit(src):
    e = src.episodes
    table = src.table.to_table().to_pandas()
    schema = {k: v for k, v in src.info['features'].items()}
    report = {'source_identity': src.identity, 'info': src.info, 'feature_schema': schema,
              'length_summary': e.length.describe().to_dict(),
              'tasks': sorted(e.caption.unique().tolist()),
              'counts': e.groupby(['source_task', 'tactile_raw', 'suggested_split']).agg(
                  episodes=('length', 'size'), frames=('length', 'sum')).reset_index().to_dict('records'),
              'action_state': {}, 'haptics': {}, 'sampled_sidecars': []}
    for col in ['action', 'observation.state']:
        x = np.stack(table[col])
        report['action_state'][col] = {'shape': list(x.shape), 'dtype': str(x.dtype),
            'nonfinite': int((~np.isfinite(x)).sum()), 'min': x.min(0).tolist(), 'max': x.max(0).tolist()}
    for col in table:
        if col.startswith('observation.haptics.'):
            x = np.stack(table[col])
            report['haptics'][col] = {'shape': list(x.shape), 'dtype': str(x.dtype),
                'nonfinite': int((~np.isfinite(x)).sum()), 'nonzero': int(np.count_nonzero(x)),
                'min': np.min(x).item(), 'max': np.max(x).item()}
    invalid_indices, invalid_timestamps, valid_chunks = 0, 0, 0
    for ep, group in table.groupby('episode_index'):
        n = int(e.loc[ep, 'length'])
        invalid_indices += int(len(group) != n or not np.array_equal(group.frame_index, np.arange(n)))
        invalid_timestamps += int(not np.allclose(group.timestamp, np.arange(n)/10, atol=1e-5))
        valid_chunks += len(src.anchors(int(ep)))
    report['all_episode_windows'] = {'invalid_frame_indices': invalid_indices,
        'invalid_nominal_timestamps': invalid_timestamps, 'valid_16_frame_chunks': valid_chunks,
        'invalid_gap_or_short_windows': int(sum(max(0, n-15) for n in e.length) - valid_chunks)}
    def quant(x):
        return np.quantile(x, [0, .01, .5, .99, 1]).tolist() if len(x) else []
    for _, group in e.groupby('source_task'):
        ep = int(group.index[0])
        raw, _ = src.tactile(ep)
        t = raw.read_end_ns.to_numpy(np.int64)
        avail = np.maximum(t, raw.available_ns.to_numpy(np.int64)) if 'available_ns' in raw else t
        clocks = src.clocks(ep)
        ix = np.searchsorted(avail, clocks, side='right') - 1
        values = np.stack(raw['values'])
        force = np.stack(raw.force)
        d = src.frames(ep)
        state = np.stack(d['observation.state'])
        # Frame association uses sequence, not nearest/future timestamp.
        frames = __import__('pandas').read_parquet(src.root / f'tactile_raw/episode_{ep:06d}_frames.parquet')
        assoc = raw.set_index('sequence').reindex(frames.xhand_sequence)
        associated = np.stack(assoc.force) if not assoc.force.isna().any() else None
        report['sampled_sidecars'].append({'episode': ep, 'task': group.iloc[0].source_task,
            'native': not bool(group.iloc[0].tactile_interpolated), 'recorded_samples': len(raw),
            'sample_schema': {k: str(v) for k, v in raw.dtypes.items()},
            'taxel_dtype': str(values.dtype), 'taxel_shape': list(values.shape),
            'taxel_nonfinite': int((~np.isfinite(values)).sum()), 'taxel_nonzero': int(np.count_nonzero(values)),
            'force_nonfinite': int((~np.isfinite(force)).sum()),
            'sample_interval_ms_min_p1_p50_p99_max': quant(np.diff(t) / 1e6),
            'frame_interval_ms_min_p1_p50_p99_max': quant(np.diff(clocks) / 1e6),
            'causal_age_ms_min_p1_p50_p99_max': quant((clocks[ix >= 0] - avail[ix[ix >= 0]]) / 1e6),
            'read_errors_max': int(raw.read_errors.max()), 'dropped_max': int(raw.dropped.max()),
            'reset_sequences': sorted(raw.reset_sequence.unique().tolist()),
            'frame_force_max_error': float(np.abs(associated - state[:, 19:34]).max()) if associated is not None else None,
            'missing_causal_samples': int((ix < 0).sum()),
            'invalid_action_windows': int(max(0, len(clocks) - 15) - len(src.anchors(ep)))})
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stats')
    args = p.parse_args()
    src = Source(args.root)
    report = audit(src)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    if args.stats:
        with open(args.stats, 'w') as f:
            json.dump(fit_stats(src), f, indent=2)

if __name__ == '__main__':
    main()
