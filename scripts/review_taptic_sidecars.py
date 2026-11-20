"""Read-only audit of actual Hub sidecars; small downloads are parsed in memory.

No interpolation, force conversion, model changes or training. Uses configured
Hugging Face credentials. Also compares inspected files with a local dataset.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

REPO = 'mfincato/taptic_trex_style_midtrain_v1'


def summary(x):
    x = np.asarray(x)
    return dict(zip(['min', 'p01', 'median', 'p99', 'max'],
                    np.quantile(x, [0, .01, .5, .99, 1]).tolist())) if x.size else {}


def repeats(x):
    equal = np.all(x[1:] == x[:-1], axis=1)
    boundaries = np.r_[0, np.flatnonzero(~equal) + 1, len(x)]
    active = np.any(x != 0, axis=1)
    both_active = active[:-1] & active[1:]
    return {'unchanged_consecutive': int(equal.sum()), 'comparisons': len(equal),
            'unchanged_fraction': float(equal.mean()), 'distinct_vectors': len(np.unique(x, axis=0)),
            'longest_identical_run_samples': int(np.diff(boundaries).max()),
            'active_samples': int(active.sum()),
            'unchanged_active_pairs': int((equal & both_active).sum()),
            'active_pair_comparisons': int(both_active.sum())}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', nargs='+', type=int, default=[0, 48, 553, 554, 771, 297])
    p.add_argument('--verified-local', action='store_true', help='Use local bytes only when their hashes match the pinned Hub metadata')
    args = p.parse_args()
    root = Path(args.root)
    remote = HfApi().dataset_info(REPO, files_metadata=True)
    revision = remote.sha
    files = {s.rfilename: s for s in remote.siblings}
    token = get_token()
    session = requests.Session()
    if token:
        session.headers['Authorization'] = 'Bearer ' + token
    verification = []

    def fetch(path):
        s = files[path]
        local = root / path
        data = local.read_bytes() if args.verified_local and local.exists() else None
        def matches(b):
            if s.lfs:
                return hashlib.sha256(b).hexdigest() == s.lfs.sha256
            return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == s.blob_id
        origin = 'local_verified_against_Hub_hash'
        if data is None or not matches(data):
            url = hf_hub_url(REPO, path, repo_type='dataset', revision=revision)
            response = session.get(url, timeout=60)
            response.raise_for_status()
            data = response.content
            origin = 'direct_Hub_download'
        digest = hashlib.sha256(data).hexdigest()
        valid = matches(data)
        if not valid:
            raise ValueError('Hub content hash mismatch: ' + path)
        local_match = local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == digest
        verification.append({'read_source': origin, 'path': path, 'bytes': len(data), 'hub_verified': valid,
                             'local_sha256_match': local_match, 'sha256': digest})
        print('verified', path, 'local_match=', local_match, flush=True)
        return data

    info = json.loads(fetch('meta/info.json'))
    tinfo = json.loads(fetch('tactile_raw/info.json'))
    episodes = pq.read_table(io.BytesIO(fetch('meta/episodes/chunk-000/file-000.parquet'))).to_pandas().set_index('episode_index')
    # One shared data file, fetched once to verify frame-associated forces/actions.
    frame_data = pq.read_table(io.BytesIO(fetch('data/chunk-000/file-000.parquet')),
                              columns=['episode_index', 'frame_index', 'observation.state']).to_pandas()
    report = {'repo': REPO, 'revision': revision, 'info': {k: v for k, v in info.items() if k != 'features'},
              'sidecar_info': tinfo, 'verification': verification, 'episodes': [],
              'sidecar_paths_count': sum(k.startswith('tactile_raw/') for k in files)}
    report['split_provenance'] = episodes.groupby(['source_task', 'tactile_raw', 'suggested_split']).agg(
        episodes=('length', 'size'), frames=('length', 'sum')).reset_index().to_dict('records')
    for ep in args.episodes:
        sample_path = f'tactile_raw/episode_{ep:06d}_right.parquet'
        frame_path = f'tactile_raw/episode_{ep:06d}_frames.parquet'
        st = pq.read_table(io.BytesIO(fetch(sample_path)))
        ft = pq.read_table(io.BytesIO(fetch(frame_path)))
        sm, fr = st.to_pandas(), ft.to_pandas()
        meta = episodes.loc[ep]
        read = sm.read_end_ns.to_numpy(np.int64)
        obs = fr.observation_ns.to_numpy(np.int64)
        avail = np.maximum(read, sm.available_ns.to_numpy(np.int64)) if 'available_ns' in sm else read
        synthetic = sm.interpolated.to_numpy(bool) if 'interpolated' in sm else np.zeros(len(sm), bool)
        force = np.stack(sm.force).astype(np.float64)
        values = np.stack(sm['values']).astype(np.float64)
        joints = np.stack(sm.joints).astype(np.float64)
        # Fresh causal recorded measurements, excluding every synthetic row.
        recorded = np.flatnonzero(~synthetic)
        rt, ra = read[recorded], avail[recorded]
        ix = np.searchsorted(ra, obs, side='right') - 1
        causal_ok = ix >= 0
        indices = recorded[ix[causal_ok]]
        in_episode = recorded[(rt >= obs[0]) & (rt <= obs[-1])]
        actual = frame_data[frame_data.episode_index == ep].sort_values('frame_index')
        states = np.stack(actual['observation.state'])
        assoc = sm.set_index('sequence').reindex(fr.xhand_sequence)
        assoc_valid = ~assoc.force.isna()
        assoc_error = float(np.abs(np.stack(assoc.loc[assoc_valid, 'force']) - states[assoc_valid, 19:34]).max()) if assoc_valid.any() else None
        d = {'episode': ep, 'task': meta.source_task, 'source_episode': int(meta.source_episode_index),
             'source_frame_start': int(meta.source_frame_start), 'source_frame_end': int(meta.source_frame_end),
             'length': int(meta.length), 'provenance': meta.tactile_raw,
             'samples_path': sample_path, 'frames_path': frame_path,
             'samples': len(sm), 'recorded_samples': len(recorded), 'synthetic_samples': int(synthetic.sum()),
             'sample_schema': {f.name: str(f.type) for f in st.schema},
             'frame_schema': {f.name: str(f.type) for f in ft.schema},
             'sample_scalar_nonfinite': {k: int((~np.isfinite(sm[k])).sum()) for k in sm if sm[k].dtype.kind == 'f' and (~np.isfinite(sm[k])).any()},
             'frame_scalar_nonfinite': {k: int((~np.isfinite(fr[k])).sum()) for k in fr if fr[k].dtype.kind == 'f' and (~np.isfinite(fr[k])).any()},
             'read_gaps_over_50ms_in_episode': int(np.sum((np.diff(rt) > 50_000_000) & (rt[:-1] >= obs[0]) & (rt[1:] <= obs[-1]))),
             'sample_null_counts': {f.name: st[f.name].null_count for f in st.schema},
             'frame_null_counts': {f.name: ft[f.name].null_count for f in ft.schema},
             'parquet_metadata': {k.decode(): v.decode(errors='replace') for k, v in (st.schema.metadata or {}).items() if k != b'pandas'},
             'list_lengths': {c: sorted(set(map(len, sm[c].dropna()))) for c in ['force', 'values', 'joints', 'command_q', 'command_target']},
             'nonfinite_forces': int((~np.isfinite(force)).sum()), 'nonfinite_taxels': int((~np.isfinite(values)).sum()),
             'force_min_xyz': force.reshape(-1, 3).min(0).tolist(), 'force_max_xyz': force.reshape(-1, 3).max(0).tolist(),
             'taxel_min_xyz': values.reshape(-1, 3).min(0).tolist(), 'taxel_max_xyz': values.reshape(-1, 3).max(0).tolist(),
             'integer_counts': bool(np.equal(force, np.round(force)).all() and np.equal(values, np.round(values)).all()),
             'sample_dt_ms': summary(np.diff(read) / 1e6), 'recorded_dt_ms': summary(np.diff(rt) / 1e6),
             'recorded_median_hz': float(1e9 / np.median(np.diff(rt))),
             'read_duration_ms': summary((read - sm.read_start_ns.to_numpy(np.int64)) / 1e6),
             'frame_dt_ms': summary(np.diff(obs) / 1e6),
             'causal_sample_age_ms': summary((obs[causal_ok] - read[indices]) / 1e6),
             'causal_availability_age_ms': summary((obs[causal_ok] - avail[indices]) / 1e6),
             'missing_causal_recorded_samples': int((~causal_ok).sum()),
             'nonincreasing_sample_times': int((np.diff(read) <= 0).sum()),
             'sequence_gap_count': int((np.diff(sm.sequence.to_numpy()) != 1).sum()),
             'nonincreasing_frame_times': int((np.diff(obs) <= 0).sum()),
             'frame_gaps_over_150ms': int((np.diff(obs) > 150_000_000).sum()),
             'read_gaps_over_50ms': int((np.diff(rt) > 50_000_000).sum()),
             'resets': int(np.count_nonzero(np.diff(sm.reset_sequence))),
             'read_errors_range': [int(sm.read_errors.min()), int(sm.read_errors.max())],
             'dropped_range': [int(sm.dropped.min()), int(sm.dropped.max())],
             'frame_sequence_matches': int(assoc_valid.sum()), 'frame_force_max_error_counts': assoc_error,
             'frame_sequence_references_synthetic': int(assoc.interpolated.fillna(False).sum()) if 'interpolated' in assoc else 0,
             'timestamp_causal_force_vs_state_max_abs_counts': float(np.abs(force[indices] - states[causal_ok, 19:34]).max()),
             'force_repetition_in_episode': repeats(force[in_episode]),
             'taxel_repetition_in_episode': repeats(values[in_episode]),
             'pre_first_frame_samples': int((read < obs[0]).sum()),
             'post_last_frame_samples': int((read > obs[-1]).sum()),
             'preroll_seconds': float((obs[0] - read[0]) / 1e9),
             'postroll_seconds': float((read[-1] - obs[-1]) / 1e9),
             'frame_t_rel_minus_dataset_timestamp_s': summary(fr.t_rel.to_numpy() - fr.timestamp.to_numpy()),
             'force_repetition_recorded': repeats(force[recorded]),
             'taxel_repetition_recorded': repeats(values[recorded]),
             'joints_repetition_recorded': repeats(joints[recorded]),
             'both_force_and_taxels_unchanged_recorded': int(np.sum(np.all(force[recorded][1:] == force[recorded][:-1], 1) & np.all(values[recorded][1:] == values[recorded][:-1], 1))),
             'interpolation_availability_delay_ms': summary((avail[synthetic] - read[synthetic]) / 1e6)}
        report['episodes'].append(d)
        print('episode', ep, 'recorded=', d['recorded_samples'], 'Hz=', d['recorded_median_hz'],
              'force_repeats=', d['force_repetition_recorded']['unchanged_fraction'], flush=True)
        del sm, st, values, force, joints
    # Adjacent re-cut episodes may share pre-roll; the policy must not treat
    # it as part of a new episode's within-episode risk history.
    report['adjacent_episode_source_slices'] = {str(ep): episodes.loc[ep, ['source_episode_index', 'source_frame_start', 'source_frame_end', 'suggested_split']].map(lambda v: v.item() if hasattr(v, 'item') else v).to_dict() for ep in args.episodes}
    if {553, 554}.issubset(args.episodes):
        a = pd.read_parquet(root / 'tactile_raw/episode_000553_right.parquet')
        b = pd.read_parquet(root / 'tactile_raw/episode_000554_right.parquet')
        start_b = pd.read_parquet(root / 'tactile_raw/episode_000554_frames.parquet').observation_ns.iloc[0]
        shared = np.intersect1d(a.sequence, b.sequence)
        report['boundary_overlap_553_554'] = {'shared_sample_sequences': len(shared),
            'all_shared_samples_before_episode_554_first_observation': bool((b[b.sequence.isin(shared)].read_end_ns < start_b).all())}
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
