"""Append re-collected recordings of one task to a T-Rex-style TAPTIC dataset, as a NEW derived dataset.

python -I scripts/taptic/append_recordings.py --base BASE --add ADD --out OUT --task unscrew_the_bottle \
    --source-repo TAPTIC/unscrew_the_bottle_fs65 --source-revision SHA --note "glove full scale 30" \
    [--val-frac 0.1 --seed 42]

BASE  T-Rex-style dataset (e.g. mfincato/taptic_trex_style_midtrain_v1); never modified.
ADD   per-task LeRobot v3 recording repo with tactile_raw/ sidecars (same robot, cameras and feature names).
OUT   new directory: BASE + the ADD recordings. Documented in docs/TOUCH_MODULES.md.

- Every BASE episode keeps its index and its frame rows (verified at the end). The ADD recordings become
  episodes n_base .. n_base + k - 1, one episode per recording (whole recording, like the base's unscrew).
- Episode metadata of the added episodes copies the base task's caption / motor primitive / object / target.
  They have no primitive annotation: subtask_index -1 per frame, subtasks [].
- Split: round(--val-frac * k) added recordings go to 'val' (numpy seed --seed), as the base does per task.
- tactile_raw sidecars are copied unchanged (renamed), videos copied as new files (not re-encoded), frame rows
  appended as a new data file (cast to the base schema), meta/stats.json recomputed for the frame features.
- The old recordings of the task stay in OUT; choose per run with --taptic_exclude_sources <old source_repo>.
- meta/derived_from.json records the parent identity, the unchanged parent episodes, the added source and the
  verification results; the data loader accepts the parent's VQ-VAE only through this file.
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from taptic.data import CAMERAS, Source  # noqa: E402
from taptic.video import VideoReader  # noqa: E402

STAT_KEYS = ('min', 'max', 'mean', 'std', 'count', 'q01', 'q10', 'q50', 'q90', 'q99')


def read_episodes(root):
    files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    return pd.concat([pd.read_parquet(p) for p in files]).sort_values('episode_index').reset_index(drop=True), files


def read_data(root):
    files = sorted((root / 'data').rglob('*.parquet'))
    return pd.concat([pd.read_parquet(p) for p in files]).sort_values(['episode_index', 'frame_index']).reset_index(drop=True)


def frame_stats(x):
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    q = np.quantile(x, [.01, .1, .5, .9, .99], axis=0)
    return {'min': x.min(0).tolist(), 'max': x.max(0).tolist(), 'mean': x.mean(0).tolist(), 'std': x.std(0).tolist(),
            'count': [int(len(x))], 'q01': q[0].tolist(), 'q10': q[1].tolist(), 'q50': q[2].tolist(),
            'q90': q[3].tolist(), 'q99': q[4].tolist()}


def episode_stat(template, value):
    """One per-episode stats cell in the base cell's container/dtype."""
    t = np.asarray(template)
    return np.asarray([value], dtype=t.dtype) if t.ndim else t.dtype.type(value)


def scalar_stat(x, stat):
    x = np.asarray(x, np.float64)
    return {'min': x.min(), 'max': x.max(), 'mean': x.mean(), 'std': x.std(), 'count': len(x),
            'q01': np.quantile(x, .01), 'q10': np.quantile(x, .1), 'q50': np.quantile(x, .5),
            'q90': np.quantile(x, .9), 'q99': np.quantile(x, .99)}[stat]


def column_array(df, key):
    v = df[key].iloc[0]
    return np.stack(df[key].to_numpy()) if isinstance(v, (np.ndarray, list)) else df[key].to_numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', required=True)
    p.add_argument('--add', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--task', required=True, help='source_task of the added recordings (must exist in BASE)')
    p.add_argument('--source-repo', required=True)
    p.add_argument('--source-revision', default='')
    p.add_argument('--note', default='')
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    base, add, out = Path(a.base).resolve(), Path(a.add).resolve(), Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')

    eb, eb_files = read_episodes(base)
    n_base = len(eb)
    assert np.array_equal(eb.episode_index.to_numpy(), np.arange(n_base)), 'base episodes must be 0..n-1'
    tmpl = eb[eb.source_task == a.task]
    if not len(tmpl):
        raise SystemExit(f'task {a.task} not in base')
    caption, prim, obj, target = (tmpl.iloc[0][k] for k in ('caption', 'motor_primitive', 'object', 'target'))
    tasks = pd.read_parquet(base / 'meta/tasks.parquet')
    tidx = int(tasks.loc[caption, 'task_index'])
    info = json.loads((base / 'meta/info.json').read_text())
    tinfo = json.loads((base / 'tactile_raw/info.json').read_text())
    en, _ = read_episodes(add)
    k = len(en)
    assert np.array_equal(en.episode_index.to_numpy(), np.arange(k)), 'added episodes must be 0..k-1'
    dn = read_data(add)
    ainfo = json.loads((add / 'meta/info.json').read_text())
    atinfo = json.loads((add / 'tactile_raw/info.json').read_text())
    if atinfo['order'] != tinfo['order']:
        raise SystemExit('taxel order differs from the base')
    for key in ('action', 'observation.state'):
        if ainfo['features'][key]['names'] != info['features'][key]['names']:
            raise SystemExit(f'{key} names differ from the base')
    print(f'[append] base {n_base} episodes, adding {k} recordings of {a.task} from {a.source_repo}', flush=True)

    shutil.copytree(base, out, ignore=shutil.ignore_patterns('.cache'))
    print('[append] base copied', flush=True)

    # ---- frame rows
    base_schema = pq.read_schema(sorted((base / 'data').rglob('*.parquet'))[0]).remove_metadata()
    total0 = int(info['total_frames'])
    new = dn.copy()
    new['episode_index'] = new['episode_index'].astype(np.int64) + n_base
    new['index'] = np.arange(total0, total0 + len(new), dtype=np.int64)
    new['task_index'] = tidx
    new['subtask_index'] = np.int64(-1)
    new['annotation_review_required'] = False
    new = new[[f.name for f in base_schema]]
    data_dir = out / 'data/chunk-000'
    next_file = max(int(q.stem.split('-')[1]) for q in data_dir.glob('file-*.parquet')) + 1
    pq.write_table(pa.Table.from_pandas(new, schema=base_schema, preserve_index=False),
                   data_dir / f'file-{next_file:03d}.parquet')
    lengths = new.groupby('episode_index').size()

    # ---- videos (copied as new files) and sidecars (renamed)
    vmap = {}
    for cam in CAMERAS:
        vdir = out / 'videos' / cam / 'chunk-000'
        nxt = max(int(q.stem.split('-')[1]) for q in vdir.glob('file-*.mp4')) + 1
        for ch, fi in sorted({(int(r[f'videos/{cam}/chunk_index']), int(r[f'videos/{cam}/file_index'])) for _, r in en.iterrows()}):
            src = add / ainfo['video_path'].format(video_key=cam, chunk_index=ch, file_index=fi)
            shutil.copy2(src, vdir / f'file-{nxt:03d}.mp4')
            vmap[(cam, ch, fi)] = nxt
            nxt += 1
    for e in range(k):
        for kind in ('right', 'frames'):
            shutil.copy2(add / f'tactile_raw/episode_{e:06d}_{kind}.parquet',
                         out / f'tactile_raw/episode_{n_base + e:06d}_{kind}.parquet')
    print('[append] videos and sidecars copied', flush=True)

    # ---- episode metadata
    rng = np.random.default_rng(a.seed)
    val = set(rng.choice(k, max(1, int(round(a.val_frac * k))), replace=False).tolist())
    src_ep0 = int(eb.source_episode_index.max()) + 1
    rows, start = [], total0
    for e in range(k):
        r = en.iloc[e].to_dict()
        n = int(lengths.loc[n_base + e])
        assert n == int(r['length']), f'episode {e}: length mismatch'
        r.update({'episode_index': n_base + e, 'tasks': [caption], 'data/chunk_index': 0, 'data/file_index': next_file,
                  'dataset_from_index': start, 'dataset_to_index': start + n,
                  'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
                  'subtasks': [], 'caption': caption, 'motor_primitive': prim, 'object': obj, 'target': target,
                  'object_needs_review': False, 'source_task': a.task, 'source_repo': a.source_repo,
                  'source_episode_index': src_ep0 + e, 'source_local_episode': e, 'source_frame_start': 0,
                  'source_frame_end': n, 'object_in_recording': 0, 'tactile_raw': 'native',
                  'tactile_interpolated': False, 'suggested_split': 'val' if e in val else 'train'})
        for cam in CAMERAS:
            ch, fi = int(r[f'videos/{cam}/chunk_index']), int(r[f'videos/{cam}/file_index'])
            r[f'videos/{cam}/chunk_index'], r[f'videos/{cam}/file_index'] = 0, vmap[(cam, ch, fi)]
        # per-episode stats of the columns whose values change here (indices) or are new (labels)
        ne = new[new.episode_index == n_base + e]
        for col in eb.columns:
            parts = col.split('/')
            if len(parts) == 3 and parts[0] == 'stats' and parts[1] in (
                    'episode_index', 'index', 'task_index', 'subtask_index', 'annotation_review_required'):
                r[col] = episode_stat(eb[col].iloc[0], scalar_stat(ne[parts[1]].to_numpy(), parts[2]))
        missing = [c for c in eb.columns if c not in r]
        if missing:
            raise SystemExit(f'cannot fill episode columns {missing}')
        rows.append({c: r[c] for c in eb.columns})
        start += n
    ep_schema = pq.read_schema(eb_files[0]).remove_metadata()
    allep = pd.concat([eb, pd.DataFrame(rows)], ignore_index=True)
    for f in eb_files:
        (out / f.relative_to(base)).unlink()
    pq.write_table(pa.Table.from_pandas(allep, schema=ep_schema, preserve_index=False),
                   out / 'meta/episodes/chunk-000/file-000.parquet')

    # ---- json metadata
    info['total_episodes'] = n_base + k
    info['total_frames'] = total0 + len(new)
    info['splits'] = {'train': f'0:{n_base + k}'}
    (out / 'meta/info.json').write_text(json.dumps(info, indent=4))
    tinfo.setdefault('sources', {})[a.source_repo] = {
        'episodes': [n_base, n_base + k], 'sample_rate_hz': atinfo.get('sample_rate_hz'), 'legacy_inline_10hz': False,
        'revision': a.source_revision, 'note': a.note, 'appended_by': 'scripts/taptic/append_recordings.py'}
    (out / 'tactile_raw/info.json').write_text(json.dumps(tinfo, indent=2))
    stats = json.loads((base / 'meta/stats.json').read_text())
    full = read_data(out)
    for key in list(stats):           # frame features; video features keep the base values
        if key in full.columns:
            stats[key] = frame_stats(column_array(full, key).astype(np.float64))
    (out / 'meta/stats.json').write_text(json.dumps(stats, indent=4))
    ts = out / 'meta/trex_style'
    if ts.exists():
        pc = pd.read_csv(ts / 'pieces.csv')
        pid0 = int(pc.piece_id.max()) + 1
        add_p = pd.DataFrame([{'task': a.task, 'source_episode': src_ep0 + e, 'local_episode': e, 'tactile_raw': 'native',
                               'source_length': int(lengths.loc[n_base + e]), 'start': 0, 'end': int(lengths.loc[n_base + e]),
                               'keep': True, 'cycle': 0, 'reason': f'appended from {a.source_repo}',
                               'length': int(lengths.loc[n_base + e]), 'piece_id': pid0 + e, 'primitive': prim}
                              for e in range(k)])
        pd.concat([pc, add_p], ignore_index=True).to_csv(ts / 'pieces.csv', index=False)
        oc = pd.read_csv(ts / 'objects.csv')
        pd.concat([oc, pd.DataFrame({'piece_id': pid0 + np.arange(k), 'object': obj, 'needs_review': False})],
                  ignore_index=True).to_csv(ts / 'objects.csv', index=False)

    # ---- verification
    print('[append] verifying', flush=True)
    psrc, osrc = Source(base), Source(out)
    cols = ['episode_index', 'frame_index', 'timestamp', 'action', 'observation.state', 'subtask_index',
            'observation.haptics.force', 'observation.haptics.valid', 'observation.haptics.local_time_ns']
    bd = read_data(base)[cols]
    od = full[full.episode_index < n_base][cols].reset_index(drop=True)
    same_rows = len(bd) == len(od) and all(np.array_equal(column_array(bd, c), column_array(od, c)) for c in cols)
    same_meta = all(np.array_equal(psrc.episodes[c].astype(str).to_numpy(), osrc.episodes.loc[:n_base - 1, c].astype(str).to_numpy())
                    for c in ['length', 'source_task', 'source_repo', 'suggested_split', 'caption', 'tactile_interpolated'])
    vr, anchors, clock_ok = VideoReader(), 0, True
    for e in range(n_base, n_base + k):
        try:
            osrc.frames(e)
            osrc.clocks(e)
            osrc.tactile(e)
        except ValueError as ex:
            clock_ok = False
            print(f'  episode {e}: {ex}', flush=True)
        anchors += len(osrc.anchors(e))
    dec = []
    for e in (n_base, n_base + k - 1):
        row = osrc.episodes.loc[e]
        for cam in CAMERAS:
            path = osrc.root / osrc.info['video_path'].format(video_key=cam, chunk_index=int(row[f'videos/{cam}/chunk_index']),
                                                              file_index=int(row[f'videos/{cam}/file_index']))
            for t in (0, int(row.length) - 1):
                img = vr.frame(str(path), float(row[f'videos/{cam}/from_timestamp']) + t / 10, 10)
                dec.append(tuple(np.asarray(img).shape))
    ver = {'parent_rows_identical': bool(same_rows), 'parent_episode_meta_identical': bool(same_meta),
           'added_frames_and_clocks_ok': bool(clock_ok), 'added_anchor_chunks': int(anchors),
           'video_frames_decoded': dec, 'identity': osrc.identity}
    derived = {'parent': {'path': str(base), 'identity': psrc.identity}, 'parent_identity': psrc.identity,
               'unchanged_episodes': list(range(n_base)),
               'appended': {'episodes': [n_base, n_base + k], 'source_repo': a.source_repo,
                            'source_revision': a.source_revision, 'source_path': str(add), 'task': a.task,
                            'val_episodes': sorted(n_base + v for v in val), 'note': a.note},
               'select': {'replace_old_recordings': {'--taptic_exclude_sources': sorted(set(tmpl.source_repo.astype(str)))},
                          'keep_both': {'--taptic_exclude_sources': []}},
               'created': datetime.datetime.now().isoformat(timespec='seconds'),
               'tool': 'scripts/taptic/append_recordings.py', 'verification': ver}
    (out / 'meta/derived_from.json').write_text(json.dumps(derived, indent=1))
    with open(out / 'README.md', 'a') as f:
        f.write(f"\n\n## Derived dataset: {a.source_repo} appended (trex-touch)\n\n"
                f"Built {derived['created']} by `scripts/taptic/append_recordings.py` from this dataset plus "
                f"`{a.source_repo}` (revision {a.source_revision or 'unrecorded'}). {a.note}\n\n"
                f"- Episodes 0..{n_base - 1}: the parent, rows unchanged. Episodes {n_base}..{n_base + k - 1}: the {k} "
                f"added `{a.task}` recordings (native ~80 Hz sidecar; no primitive labels: subtask_index -1); "
                f"val = {sorted(n_base + v for v in val)}.\n"
                f"- The old `{a.task}` recordings are still here: train with `--taptic_exclude_sources "
                f"{' '.join(sorted(set(tmpl.source_repo.astype(str))))}` to replace them, or without it to keep both.\n"
                f"- Provenance and verification: `meta/derived_from.json`.\n")
    print(json.dumps(ver, indent=1))
    ok = same_rows and same_meta and clock_ok and anchors > 0
    print('APPEND_OK' if ok else 'APPEND_VERIFY_FAILED', out)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
