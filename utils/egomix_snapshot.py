"""Snapshot views of the egomix outputs while the conversion jobs still write them: only the episodes
that are complete and safe to read, as LeRobot v3 datasets that TRexLeRobotDataset loads unchanged.

    python -m utils.egomix_snapshot config/egomix_snapshot.yaml --dry-run   # what a refresh would add
    python -m utils.egomix_snapshot config/egomix_snapshot.yaml             # build, or refresh, the views
    python -m utils.egomix_snapshot config/egomix_snapshot.yaml --verify    # do the views still match?
    sbatch scripts/egomix_snapshot.sh                                       # the refresh as a cpu job
    LEROBOT_ROOT=<view_dir>/retarget sbatch scripts/pretrain.sh             # train on the views

Optional: nothing else in trex-mod imports this module, and the loader is not changed.

Why views. An egomix output (<output>/{lerobot,retarget}/<dataset>_<W>x<H>[-k], LeRobot v3) cannot be
opened as is while its job runs: LeRobot globs every data/meta parquet file, including the one being
written (no footer yet); a crashed writer session leaves episodes counted in info.json, and in the
data's `index` column, that meta/episodes does not list; and episodes that must not be trained on
(failed retargeting, short videos, ...) sit next to the good ones.

What is safe. Decided once, from what is on disk at startup:
- An egomix worker appends to a dataset in sessions (one per batch and resolution). LeRobot starts new
  data, video and meta/episodes files for each session and never touches a finished session's files
  again. A session's meta/episodes parquet is readable (has its footer) only once the session is
  finalized, after its data and videos are closed. So the episodes listed in readable meta/episodes
  files (also older than settle_seconds) are complete and their files immutable; only those count.
- The state files (<output>/egomix_state.jsonl) are read after the datasets, merged across the
  sources (EgoVerse B retargets some episodes whose LeRobot version is in A's folder).

What is taken:
- retarget: episodes whose retarget state is ok.
- lerobot: episodes whose lerobot state is ok and whose retargeting is ok too (an episode's LeRobot
  version is kept only if it can be retargeted). Failed retargeting: never; not retargeted yet: a
  later refresh.
- Skipped as egomix's drop_episodes.py would drop them: videos shorter than their data and, with
  world_up.recheck, EgoVerse retarget episodes converted before the Nov 10 pitch test that fail
  convert_worker.check_world_up now (and their LeRobot versions). The recheck imports egomix's
  convert_worker and the sim, so it runs in a child process with the egomix environment's python.
- An episode id is taken once (the copy the state file points to, if converted twice).

The views. <view_dir>/<mode>/<output folder>__<dataset root>/, one per source dataset: data and
meta/episodes files holding only the taken episodes, renumbered 0.. (consistent even where the source
lost a session's metadata); videos are symlinks to the source video files (nothing large is copied);
info.json, stats.json (from the episodes' own stats) and tasks.parquet rewritten. A refresh only
appends: new data and meta/episodes files and links, then info/stats/tasks and, last,
meta/egomix_snapshot/state.json (what the view holds; manifest-<first index>.jsonl maps each
episode to its source). New files are written as *.tmp and renamed in that order, so an interrupted
refresh leaves the view as it was. A training that is already running keeps the episodes it loaded.

After egomix's drop_episodes.py --apply rebuilds a source dataset, the files the view links to change:
--verify reports it (and a refresh refuses to append to that view); rebuild it with --rebuild.
"""
import argparse
import csv
import fcntl
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MODES = ('lerobot', 'retarget')
FORMAT = 1
STATE_DIR = 'meta/egomix_snapshot'
META_ROWS_PER_FILE = 1000
DEFAULTS = dict(sources=None, view_dir=None, modes=['retarget'], settle_seconds=300, world_up=None)
WORLD_UP_DEFAULTS = dict(recheck=True, python='/home/lgori/.conda/envs/egomix/bin/python',
                         egomix_dir='/home/lgori/egomix', sim_repo='/home/lgori/panda-sim')
RESULT = 'EGOMIX_SNAPSHOT_WORLD_UP_RESULT '


def log(message=''):
    print(f'{time.strftime("%H:%M:%S")} {message}', flush=True)


def gb(n):
    return f'{n / 1e9:,.1f} GB'


def retry(read, tries=5):
    """read() of a file a job may be rewriting at that moment (info.json, tasks.parquet)."""
    for attempt in range(tries):
        try:
            return read()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(1 + attempt)


def read_parquet(path, columns=None):
    # pre_buffer off: with it, pyarrow fails (EFAULT) on recently written files on this cluster's BeeGFS
    return pq.read_table(path, columns=columns, pre_buffer=False)


# ---------------------------------------------------------------- config

def load_config(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f'{path}: unknown keys {sorted(unknown)}')
    cfg = {**DEFAULTS, **cfg}
    world_up = {**WORLD_UP_DEFAULTS, **(cfg['world_up'] or {})}
    if set(world_up) - set(WORLD_UP_DEFAULTS):
        raise SystemExit(f'{path}: unknown world_up keys {sorted(set(world_up) - set(WORLD_UP_DEFAULTS))}')
    cfg['world_up'] = world_up
    here = path.parent
    cfg['sources'] = [(here / Path(s).expanduser()).resolve() for s in cfg['sources'] or []]
    if not cfg['sources']:
        raise SystemExit(f'{path}: list the egomix output folders under sources')
    for source in cfg['sources']:
        if not (source / 'egomix_state.jsonl').exists():
            raise SystemExit(f'{source}: no egomix_state.jsonl, not an egomix output folder')
    names = Counter(s.name for s in cfg['sources'])
    if any(n > 1 for n in names.values()):
        raise SystemExit(f'{path}: two sources named {names.most_common(1)[0][0]}; their views would collide')
    if not cfg['view_dir']:
        raise SystemExit(f'{path}: set view_dir')
    cfg['view_dir'] = (here / Path(cfg['view_dir']).expanduser()).resolve()
    if any(cfg['view_dir'].is_relative_to(s) for s in cfg['sources']):
        raise SystemExit(f'{path}: view_dir must not be inside a source folder')
    if not set(cfg['modes']) <= set(MODES):
        raise SystemExit(f'{path}: modes must be among {MODES}')
    return cfg


# ---------------------------------------------------------------- the sources, as they are at startup

class Cache:
    """file path -> value computed from the file, kept while the file's size and mtime are unchanged."""
    MISS = object()

    def __init__(self, path):
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def lookup(self, path):
        """The cached value, None for a missing file, or Cache.MISS."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        hit = self.data.get(str(path))
        return hit[2] if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns else Cache.MISS

    def store(self, path, value):
        try:
            st = os.stat(path)
        except OSError:
            return
        self.data[str(path)] = [st.st_size, st.st_mtime_ns, value]

    def get(self, path, compute):
        value = self.lookup(path)
        if value is Cache.MISS:
            value = compute(path)
            self.store(path, value)
        return value

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)


_STATS = {}


def stat_of(path):
    """(size, mtime_ns) at the snapshot, or None."""
    if path not in _STATS:
        try:
            st = os.stat(path)
            _STATS[path] = (st.st_size, st.st_mtime_ns)
        except OSError:
            _STATS[path] = None
    return _STATS[path]


class Root:
    """One source LeRobot dataset, <output>/<mode>/<dataset>_<W>x<H>[-k]."""

    def __init__(self, source, mode, path, view_dir):
        self.source, self.mode, self.path = source, mode, path
        self.dataset = path.name.rsplit('_', 1)[0]
        self.key = f'{source.name}/{mode}/{path.name}'
        self.view = view_dir / mode / f'{source.name}__{path.name}'
        self.episodes, self.unreadable_meta, self.unfinished, self.error = [], 0, 0, None

    def scan(self, cutoff):
        """The episodes of the finished sessions: those in readable meta/episodes files."""
        self.info = retry(lambda: json.loads((self.path / 'meta/info.json').read_text()))
        self.video_keys = [k for k, v in self.info['features'].items() if v['dtype'] == 'video']
        columns = ['episode_index', 'length', 'data/chunk_index', 'data/file_index'] + [
            f'videos/{k}/{c}' for k in self.video_keys
            for c in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
        for meta in sorted((self.path / 'meta/episodes').glob('*/*.parquet')):
            try:
                if meta.stat().st_mtime > cutoff:
                    raise ValueError('written too recently')
                table = read_parquet(meta, columns)  # no footer yet: the session is still open
            except Exception:  # noqa: BLE001
                self.unreadable_meta += 1
                continue
            for position, row in enumerate(table.to_pylist()):
                self.episodes.append(dict(row, meta=meta, position=position))
        indices = [e['episode_index'] for e in self.episodes]
        if len(set(indices)) != len(indices):
            raise ValueError('episode indices repeat across meta/episodes files')
        self.episodes.sort(key=lambda e: e['episode_index'])
        # read after the meta files, so it holds every task their episodes use
        tasks = retry(lambda: read_parquet(self.path / 'meta/tasks.parquet').to_pandas())
        self.tasks = {int(i): str(t) for t, i in zip(tasks.index, tasks['task_index'])}
        self.unfinished = max(0, int(self.info['total_episodes']) - len(self.episodes))

    def data_path(self, ep):
        return self.path / self.info['data_path'].format(chunk_index=ep['data/chunk_index'],
                                                         file_index=ep['data/file_index'])

    def video_path(self, ep, key):
        return self.path / self.info['video_path'].format(
            video_key=key, chunk_index=ep[f'videos/{key}/chunk_index'], file_index=ep[f'videos/{key}/file_index'])

    def sidecar(self, index, suffix='.json'):
        return self.path / 'meta' / self.dataset / f'episode_{index:06d}{suffix}'

    def short_video(self, ep):
        fps = self.info['fps']
        return any(ep['length'] != round(ep[f'videos/{k}/to_timestamp'] * fps)
                   - round(ep[f'videos/{k}/from_timestamp'] * fps) for k in self.video_keys)

    def settled(self, ep, cutoff):
        files = [self.data_path(ep), *(self.video_path(ep, k) for k in self.video_keys)]
        return all(stat_of(f) is not None and stat_of(f)[1] / 1e9 <= cutoff for f in files)


def discover(sources, view_dir):
    roots = []
    for source in sources:
        for mode in MODES:
            for path in sorted((source / mode).glob('*_*x*')):
                if (path.name.endswith(('.before_drop', '.dropping', '.partial'))
                        or not (path / 'meta/info.json').exists()):
                    continue
                roots.append(Root(source, mode, path, view_dir))
    return roots


def read_states(sources):
    """(dataset, id, mode) -> latest record, over all the sources' state files."""
    latest = {}
    for source in sources:
        with open(source / 'egomix_state.jsonl') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # the line being appended
                key = (record['dataset'], record['id'], record['mode'])
                old = latest.get(key)
                if old is None or record.get('time', '') >= old.get('time', ''):
                    latest[key] = record
    return latest


def sidecar_id(path):
    try:
        content = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return [(content.get('egomix') or {}).get('id'), content.get('episode_index')]


class WorldUp:
    """egomix drop_episodes.py's world-up recheck of a retarget episode's source clip (its .npz
    sidecar) with convert_worker.check_world_up, run by the egomix environment's python in a child
    process (it imports the sim and torch); verdicts are cached per .npz."""

    def __init__(self, cfg, cache):
        self.cfg, self.cache = cfg, cache
        worker = Path(cfg['egomix_dir']) / 'convert_worker.py'
        if not worker.exists():
            raise SystemExit(f'{worker} not found: set world_up.egomix_dir, or world_up.recheck: false')
        self.datasets = set(cache.get(worker, lambda _: self._child([])['datasets']) or ())
        if not self.datasets:
            log(f'  {worker} has no SLAM_WORLD (an older version?): no world-up recheck')

    def _child(self, items):
        log(f'world-up recheck of {len(items):,} episodes in a child process (imports the sim, ~1 min) ...'
            if items else 'reading SLAM_WORLD from convert_worker.py in a child process (imports the sim) ...')
        request = dict(sim_repo=self.cfg['sim_repo'], egomix_dir=self.cfg['egomix_dir'], items=items)
        done = subprocess.run([self.cfg['python'], str(Path(__file__).resolve()), '--world-up-worker'],
                              input=json.dumps(request), text=True, capture_output=True,
                              env=dict(os.environ, MALLOC_ARENA_MAX='2'))
        results = [line for line in done.stdout.split('\n') if line.startswith(RESULT)]
        if done.returncode or not results:
            raise RuntimeError(f'world-up recheck failed ({done.returncode}): {done.stderr.strip()[-3000:]}')
        return json.loads(results[-1][len(RESULT):])

    def verdicts(self, items):
        """[(npz path, dataset)] -> ['pass' or 'fail: <why>']; no .npz passes, as in drop_episodes.py."""
        out = [self.cache.lookup(npz) for npz, _ in items]
        missing = [i for i, v in enumerate(out) if v is Cache.MISS]
        if missing:
            found = self._child([[str(items[i][0]), items[i][1]] for i in missing])['verdicts']
            for i, verdict in zip(missing, found):
                out[i] = verdict
                self.cache.store(items[i][0], verdict)
        return [v or 'pass' for v in out]


def world_up_worker():
    """Child process of WorldUp: {"sim_repo", "egomix_dir", "items": [[npz, dataset]]} on stdin."""
    request = json.loads(sys.stdin.read())
    sys.path[:0] = [request['sim_repo'], request['egomix_dir']]
    import convert_worker

    def check(item):
        path, dataset = Path(item[0]), item[1]
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                transforms = {'camera': z['transforms.camera'],
                              convert_worker.WRIST: z[f'transforms.{convert_worker.WRIST}']}
        except Exception as error:  # noqa: BLE001
            return f'fail: unreadable {path.name} ({type(error).__name__})'
        try:
            convert_worker.check_world_up(SimpleNamespace(transforms=transforms), dataset)
            return 'pass'
        except ValueError as error:
            return f'fail: {error}'

    with ThreadPoolExecutor(4) as pool:
        verdicts = list(pool.map(check, request['items']))
    print(RESULT + json.dumps(dict(datasets=sorted(getattr(convert_worker, 'SLAM_WORLD', ())), verdicts=verdicts)),
          flush=True)


# ---------------------------------------------------------------- what to take

def decide(roots, latest, taken, cfg, args, cutoff, cache):
    """Sets ep['decision'] ('take' or why not) on every episode; returns {root key: [episodes to add]}."""
    jobs = [(root, ep) for root in roots for ep in root.episodes]
    with ThreadPoolExecutor(16) as pool:
        ids = list(pool.map(lambda job: cache.get(job[0].sidecar(job[1]['episode_index']), sidecar_id), jobs))
    occurrences = defaultdict(list)
    for (root, ep), found in zip(jobs, ids):
        ep['id'] = found[0] if found and found[1] == ep['episode_index'] else None
        if ep['id'] is None:
            ep['decision'] = 'no sidecar' if not found else 'sidecar does not match'
        else:
            occurrences[(root.mode, root.dataset, ep['id'])].append((root, ep))
    for (mode, dataset, id_), places in occurrences.items():  # one per id: the one the state points to
        record = latest.get((dataset, id_, mode)) or {}
        chosen = next((ep for root, ep in places if record.get('root') == str(root.path)), places[0][1])
        for root, ep in places:
            if ep is not chosen:
                ep['decision'] = 'duplicate'

    def own_mode(root, ep):
        if ep['id'] in taken[(root.mode, root.dataset)]:
            return 'already in a view'
        record = latest.get((root.dataset, ep['id'], root.mode))
        if record is None:
            return f'{root.mode} not recorded yet'
        if record['status'] != 'ok':
            return f'{root.mode} {record["status"]}'
        if root.short_video(ep):
            return 'short video'
        if not root.settled(ep, cutoff):
            return 'files too recent'
        return None

    for root, ep in jobs:
        if 'decision' not in ep:
            ep['decision'] = own_mode(root, ep)
    # world-up recheck of the retarget episodes that would be taken (a lerobot episode needs its own's)
    world_up, slam = {}, set()
    if cfg['world_up']['recheck'] and any(e['decision'] is None for _, e in jobs):
        checker = WorldUp(cfg['world_up'], cache)
        slam = checker.datasets
        candidates = [(r, e) for r, e in jobs if r.mode == 'retarget' and e['decision'] is None and r.dataset in slam]
        verdicts = checker.verdicts([(r.sidecar(e['episode_index'], '.npz'), r.dataset) for r, e in candidates])
        for (root, ep), verdict in zip(candidates, verdicts):
            world_up[(root.dataset, ep['id'])] = verdict
            if verdict != 'pass':
                ep['decision'] = 'world not gravity up'
    for root, ep in jobs:
        if root.mode != 'lerobot' or ep['decision'] is not None:
            continue
        record = latest.get((root.dataset, ep['id'], 'retarget'))
        if record is None:
            ep['decision'] = 'retarget not done yet'
        elif record['status'] != 'ok':
            ep['decision'] = f'retarget {record["status"]}'
        elif root.dataset in slam and ep['id'] not in taken[('retarget', root.dataset)]:
            verdict = world_up.get((root.dataset, ep['id']))
            if verdict is None:
                ep['decision'] = 'retarget not checked yet'  # its retarget episode is not in a finished session
            elif verdict != 'pass':
                ep['decision'] = 'retarget world not gravity up'
    additions = {}
    for root in roots:
        chosen = []
        for ep in root.episodes:
            if ep['decision'] is None:
                if root.mode not in cfg['modes'] or (args.only and not any(fnmatch.fnmatch(root.key, p)
                                                                           for p in args.only)):
                    ep['decision'] = 'mode or dataset not selected'
                elif args.max_episodes is not None and len(chosen) >= args.max_episodes:
                    ep['decision'] = 'over --max-episodes'
                else:
                    ep['decision'] = 'take'
                    chosen.append(ep)
        if chosen:
            additions[root.key] = chosen
    return additions


# ---------------------------------------------------------------- one view

def features_digest(info):
    keep = {k: (v['dtype'], v['shape'], v.get('names')) for k, v in info['features'].items()}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def new_state(root):
    return dict(format=FORMAT, source=str(root.path), dataset=root.dataset, mode=root.mode,
                features=features_digest(root.info), total_episodes=0, total_frames=0, next_data_file=[0, 0],
                next_meta_file=[0, 0], videos={}, tasks=[], ids=[], stats=None, refreshes=[])


def read_view_states(view_dir):
    """{view root: state} of the views built so far."""
    out = {}
    for path in view_dir.glob(f'*/*/{STATE_DIR}/state.json'):
        out[path.parents[2]] = json.loads(path.read_text())
    return out


def verify_view(view, state, check_info=True):
    """Problems with a view: a linked source video that changed or vanished, a missing link, and
    (check_info) an info.json ahead of state.json, left by an interrupted refresh (the next one repairs it)."""
    problems = []
    info = json.loads((view / 'meta/info.json').read_text()) if (view / 'meta/info.json').exists() else None
    if info is None or (check_info and info['total_episodes'] != state['total_episodes']):
        problems.append('meta/info.json does not match state.json (an interrupted refresh: refresh again)')
        return problems
    for key, files in state['videos'].items():
        for identity, (chunk, file) in files['files'].items():
            relpath, size, mtime = identity.rsplit('|', 2)
            source = Path(state['source']) / relpath
            link = view / info['video_path'].format(video_key=key, chunk_index=chunk, file_index=file)
            if stat_of(source) != (int(size), int(mtime)):
                problems.append(f'{source} changed since it was linked')
            elif not link.is_symlink() or os.readlink(link) != str(source):
                problems.append(f'{link} is not the link to {source}')
            if len(problems) >= 5:
                return problems
    return problems


def advance(chunk_file, chunks_size):
    chunk, file = chunk_file
    return [chunk + 1, 0] if file == chunks_size - 1 else [chunk, file + 1]


def aggregate_stats(stats_list):
    """LeRobot's compute_stats.aggregate_stats (0.4-0.6), without importing LeRobot."""
    out = {}
    for key in sorted({k for stats in stats_list for k in stats}):
        parts = [stats[key] for stats in stats_list if key in stats]
        means = np.stack([s['mean'] for s in parts])
        variances = np.stack([s['std'] ** 2 for s in parts])
        counts = np.stack([s['count'] for s in parts])
        total = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        mean = (means * counts).sum(axis=0) / total
        variance = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total
        out[key] = dict(min=np.min(np.stack([s['min'] for s in parts]), axis=0),
                        max=np.max(np.stack([s['max'] for s in parts]), axis=0),
                        mean=mean, std=np.sqrt(variance), count=total)
        for q in [k for k in parts[0] if k.startswith('q') and k[1:].isdigit()]:
            if all(q in s for s in parts):
                out[key][q] = (np.stack([s[q] for s in parts]) * counts).sum(axis=0) / total
    return out


def episode_stats(row):
    stats = defaultdict(dict)
    for name in row.column_names:
        if name.startswith('stats/'):
            feature, stat = name[len('stats/'):].rsplit('/', 1)
            value = row.column(name)[0].as_py()
            if value is not None:
                stats[feature][stat] = np.asarray(value)
    return dict(stats)


def _replace(table, name, values):
    i = table.schema.get_field_index(name)
    field = table.schema.field(i)
    return table.set_column(i, field, pa.array(values, type=field.type))


def _tmp(path):
    return path.with_name(path.name + '.tmp')


def build(root, episodes, state):
    """Writes this refresh's files into the view as *.tmp; returns the new state and the renames
    [(tmp, final)] to commit, in commit order (state.json last)."""
    info, view = root.info, root.view
    chunks = int(info.get('chunks_size', 1000))
    limit = float(info.get('data_files_size_in_mb', 100)) * 1e6
    state = json.loads(json.dumps(state))
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    first_index = state['total_episodes']
    tasks = {t: i for i, t in enumerate(state['tasks'])}
    links, data_files, meta_rows, new_stats, manifest = [], [], [], [], []
    writer, data_file, data_bytes = None, None, 0.

    def data_writer(schema, size):
        nonlocal writer, data_file, data_bytes
        if writer is None or data_bytes + size > limit:
            if writer is not None:
                writer.close()
            data_file = state['next_data_file']
            state['next_data_file'] = advance(data_file, chunks)
            path = view / info['data_path'].format(chunk_index=data_file[0], file_index=data_file[1])
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(_tmp(path), schema=schema, compression='snappy', use_dictionary=True)
            data_bytes = 0.
            data_files.append(path)
        data_bytes += size
        return writer

    meta_tables = {}
    by_data_file = defaultdict(list)
    for ep in episodes:
        by_data_file[root.data_path(ep)].append(ep)
    for path in sorted(by_data_file, key=lambda p: min(e['episode_index'] for e in by_data_file[p])):
        table = read_parquet(path)
        per_row = stat_of(path)[0] / max(1, table.num_rows)
        for ep in sorted(by_data_file[path], key=lambda e: e['episode_index']):
            rows = table.filter(pc.equal(table['episode_index'], ep['episode_index']))
            if rows.num_rows != ep['length']:
                raise ValueError(f'{path}: {rows.num_rows} rows for episode {ep["episode_index"]} of '
                                 f'{ep["length"]} frames')
            index, frame0 = state['total_episodes'], state['total_frames']
            source_tasks = rows['task_index'].to_numpy()
            lookup = {}
            for t in np.unique(source_tasks):
                text = root.tasks[int(t)]
                if text not in tasks:
                    tasks[text] = len(tasks)
                    state['tasks'].append(text)
                lookup[int(t)] = tasks[text]
            rows = _replace(rows, 'episode_index', np.full(rows.num_rows, index))
            rows = _replace(rows, 'index', np.arange(frame0, frame0 + rows.num_rows))
            rows = _replace(rows, 'task_index', np.vectorize(lookup.get, otypes=[np.int64])(source_tasks))
            w = data_writer(rows.schema, per_row * rows.num_rows)
            w.write_table(rows.cast(w.schema))
            # its meta/episodes row: the source's, with the view's places
            if ep['meta'] not in meta_tables:
                meta_tables[ep['meta']] = read_parquet(ep['meta'])
            meta = meta_tables[ep['meta']].slice(ep['position'], 1)
            if meta['episode_index'][0].as_py() != ep['episode_index']:
                raise ValueError(f'{ep["meta"]}: row {ep["position"]} is not episode {ep["episode_index"]}')
            values = {'episode_index': index, 'data/chunk_index': data_file[0], 'data/file_index': data_file[1],
                      'dataset_from_index': frame0, 'dataset_to_index': frame0 + rows.num_rows}
            for key in root.video_keys:
                source = root.video_path(ep, key)
                identity = f'{source.relative_to(root.path).as_posix()}|{stat_of(source)[0]}|{stat_of(source)[1]}'
                files = state['videos'].setdefault(key, {'next': [0, 0], 'files': {}})
                if identity not in files['files']:
                    files['files'][identity] = files['next']
                    files['next'] = advance(files['next'], chunks)
                    chunk, file = files['files'][identity]
                    links.append((view / info['video_path'].format(video_key=key, chunk_index=chunk,
                                                                   file_index=file), source))
                values[f'videos/{key}/chunk_index'], values[f'videos/{key}/file_index'] = files['files'][identity]
            for name, value in values.items():
                meta = _replace(meta, name, [value])
            meta_rows.append(meta)
            new_stats.append(episode_stats(meta))
            manifest.append(dict(index=index, id=ep['id'], source=str(root.path),
                                 source_episode_index=ep['episode_index'], frames=rows.num_rows, added=stamp))
            state['ids'].append(ep['id'])
            state['total_episodes'] += 1
            state['total_frames'] += rows.num_rows
        del table
    if writer is not None:
        writer.close()
    meta_files = []
    for start in range(0, len(meta_rows), META_ROWS_PER_FILE):
        block = pa.concat_tables(meta_rows[start:start + META_ROWS_PER_FILE], promote_options='permissive')
        meta_file = state['next_meta_file']
        state['next_meta_file'] = advance(meta_file, chunks)
        block = _replace(block, 'meta/episodes/chunk_index', np.full(block.num_rows, meta_file[0]))
        block = _replace(block, 'meta/episodes/file_index', np.full(block.num_rows, meta_file[1]))
        path = view / f'meta/episodes/chunk-{meta_file[0]:03d}/file-{meta_file[1]:03d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(block, _tmp(path), compression='snappy')
        meta_files.append(path)
    previous = ([{f: {s: np.asarray(v) for s, v in stats.items()} for f, stats in state['stats'].items()}]
                if state['stats'] else [])
    state['stats'] = {f: {s: np.asarray(v).tolist() for s, v in stats.items()}
                      for f, stats in aggregate_stats(previous + new_stats).items()}
    meta_dir = view / 'meta'
    (view / STATE_DIR).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'task_index': np.arange(len(state['tasks']), dtype=np.int64)},  # as LeRobot's write_tasks
                 index=pd.Index(state['tasks'], name='task')).to_parquet(_tmp(meta_dir / 'tasks.parquet'))
    _tmp(meta_dir / 'stats.json').write_text(json.dumps(state['stats'], indent=4))
    out_info = dict(info, total_episodes=state['total_episodes'], total_frames=state['total_frames'],
                    total_tasks=len(state['tasks']), splits={'train': f'0:{state["total_episodes"]}'})
    _tmp(meta_dir / 'info.json').write_text(json.dumps(out_info, indent=4))
    manifest_path = view / STATE_DIR / f'manifest-{first_index:06d}.jsonl'
    _tmp(manifest_path).write_text(''.join(json.dumps(m) + '\n' for m in manifest))
    state['refreshes'].append(dict(time=stamp, first_index=first_index, episodes=len(manifest),
                                   frames=sum(m['frames'] for m in manifest),
                                   source_info_total=info['total_episodes']))
    _tmp(view / STATE_DIR / 'state.json').write_text(json.dumps(state))
    renames = [(_tmp(p), p) for p in data_files + meta_files] + [
        (_tmp(meta_dir / n), meta_dir / n) for n in ('tasks.parquet', 'stats.json')] + [
        (_tmp(manifest_path), manifest_path), (_tmp(meta_dir / 'info.json'), meta_dir / 'info.json'),
        (_tmp(view / STATE_DIR / 'state.json'), view / STATE_DIR / 'state.json')]
    return state, links, renames, manifest


def refresh_view(root, episodes, state):
    view = root.view
    if state is None:
        state = new_state(root)
    elif state['features'] != features_digest(root.info):
        raise RuntimeError(f'{view} holds other features than {root.path}; rebuild it (--rebuild)')
    elif state['source'] != str(root.path):
        raise RuntimeError(f'{view} was built from {state["source"]}, not {root.path}')
    if view.exists():  # what an interrupted refresh left: *.tmp, and parquet files past the committed ones
        for leftover in view.rglob('*.tmp'):
            leftover.unlink()
        for folder, first_free in (('data', state['next_data_file']), ('meta/episodes', state['next_meta_file'])):
            for path in (view / folder).glob('chunk-*/file-*.parquet'):
                chunk, file = int(path.parent.name[len('chunk-'):]), int(path.stem[len('file-'):])
                if [chunk, file] >= first_free:
                    path.unlink()
    started = time.time()
    new, links, renames, manifest = build(root, episodes, state)
    for link, source in links:  # links of later files than any the view references: safe before the commit
        link.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp(link)
        tmp.unlink(missing_ok=True)
        os.symlink(source, tmp)
        os.replace(tmp, link)
    for tmp, final in renames:
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, final)
    log(f'{root.key}: +{len(manifest):,} episodes ({sum(m["frames"] for m in manifest):,} frames, '
        f'{len(links)} new video links) -> {view} now {new["total_episodes"]:,} episodes '
        f'({time.time() - started:.0f} s)')
    return len(manifest)


# ---------------------------------------------------------------- main

def report(roots, additions, plan_path):
    with open(plan_path, 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['source', 'mode', 'root', 'episode_index', 'id', 'decision'])
        for root in roots:
            for ep in root.episodes:
                out.writerow([root.source.name, root.mode, root.path.name, ep['episode_index'], ep.get('id'),
                              ep['decision']])
    log('')
    log('per source and mode (episodes of finished sessions; "unfinished" = still being written or lost):')
    groups = defaultdict(list)
    for root in roots:
        groups[(root.source.name, root.mode)].append(root)
    for (source, mode), group in groups.items():
        counts = Counter(ep['decision'] for root in group for ep in root.episodes)
        parts = [f'{n:,} {d}' for d, n in counts.most_common()] + [f'{sum(r.unfinished for r in group):,} unfinished']
        errors = sum(1 for r in group if r.error)
        log(f'  {source}/{mode}: ' + ', '.join(parts + ([f'{errors} datasets unreadable'] if errors else [])))
    log(f'to add now: {sum(len(v) for v in additions.values()):,} episodes to {len(additions)} views; '
        f'every episode\'s decision: {plan_path}')


def main():
    if sys.argv[1:] == ['--world-up-worker']:
        return world_up_worker()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='the snapshot config (see config/egomix_snapshot.yaml)')
    parser.add_argument('--dry-run', action='store_true', help='decide and report; write no view')
    parser.add_argument('--verify', action='store_true', help='only check that the views still match their sources')
    parser.add_argument('--rebuild', action='store_true',
                        help='delete the selected views first and build them again from scratch')
    parser.add_argument('--only', action='append',
                        help='only the source datasets matching this glob on <output>/<mode>/<dataset>, '
                             'e.g. "egomix_10tb/retarget/hoi4d_*" (repeatable)')
    parser.add_argument('--max-episodes', type=int, help='add at most this many episodes per view (a trial)')
    args = parser.parse_args()
    cfg = load_config(args.config)
    view_dir = cfg['view_dir']
    view_dir.mkdir(parents=True, exist_ok=True)
    lock = open(view_dir / '.egomix_snapshot.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if not args.verify else 0))
    except OSError:
        raise SystemExit(f'another egomix_snapshot refresh runs on {view_dir}')
    states = read_view_states(view_dir)
    if args.verify:
        bad = 0
        for view, state in sorted(states.items()):
            problems = verify_view(view, state)
            bad += bool(problems)
            log(f'{view.relative_to(view_dir)}: {state["total_episodes"]:,} episodes, '
                + ('OK' if not problems else 'STALE: ' + '; '.join(problems)))
        log(f'{len(states)} views, {bad} stale' + (' (rebuild them with --rebuild)' if bad else ''))
        sys.exit(1 if bad else 0)
    snapshot = time.time()
    cutoff = snapshot - float(cfg['settle_seconds'])
    log(f'snapshot of {len(cfg["sources"])} sources at {time.strftime("%Y-%m-%d %H:%M:%S")} '
        f'(files newer than {cfg["settle_seconds"]} s are left for later); views in {view_dir}')
    roots = discover(cfg['sources'], view_dir)
    if args.rebuild and not args.dry_run:
        for root in roots:
            if root.mode in cfg['modes'] and (not args.only or any(fnmatch.fnmatch(root.key, p) for p in args.only)):
                if (root.view / STATE_DIR / 'state.json').exists():
                    shutil.rmtree(root.view)
                    states.pop(root.view, None)
                    log(f'  removed {root.view} to rebuild it')

    def scan(root):
        try:
            root.scan(cutoff)
        except Exception as error:  # noqa: BLE001
            root.error, root.episodes = f'{type(error).__name__}: {error}', []

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(scan, roots))
    for root in roots:
        if root.error:
            log(f'  skipping {root.key}: {root.error}')
    latest = read_states(cfg['sources'])
    log(f'{len(roots)} source datasets, {sum(len(r.episodes) for r in roots):,} episodes in finished sessions, '
        f'{len(latest):,} state records, {len(states)} views so far')
    taken = defaultdict(set)
    for state in states.values():
        taken[(state['mode'], state['dataset'])].update(state['ids'])
    cache = Cache(view_dir / '.egomix_snapshot_cache.json')
    try:
        additions = decide(roots, latest, taken, cfg, args, cutoff, cache)
    finally:
        cache.save()
    (view_dir / 'plans').mkdir(exist_ok=True)
    report(roots, additions, view_dir / 'plans' / f'plan_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    by_key = {root.key: root for root in roots}
    if args.dry_run:
        rows = {}
        for key, episodes in additions.items():
            root, data = by_key[key], 0.
            for ep in episodes:
                path = root.data_path(ep)
                if path not in rows:
                    rows[path] = pq.ParquetFile(path, pre_buffer=False).metadata.num_rows
                data += stat_of(path)[0] * ep['length'] / max(1, rows[path])
            log(f'  {key}: +{len(episodes):,} episodes, ~{gb(data)} of data tables')
        log('dry run: no view written')
        return
    failed, added = [], 0
    for key, episodes in additions.items():
        root = by_key[key]
        state = states.get(root.view)
        try:
            if state is not None:
                problems = verify_view(root.view, state, check_info=False)
                if problems:
                    raise RuntimeError('its source changed since it was built (drop_episodes.py?): '
                                       + '; '.join(problems[:2]) + '; rebuild it with --rebuild')
            added += refresh_view(root, episodes, state)
        except Exception as error:  # noqa: BLE001
            failed.append(key)
            log(f'{key}: FAILED, its view is as before: {type(error).__name__}: {error}')
    log(f'added {added:,} episodes to {len(additions) - len(failed)} views'
        + (f'; {len(failed)} failed: {", ".join(failed)}' if failed else '')
        + f'. Train on them with --lerobot_root {" ".join(str(view_dir / m) for m in cfg["modes"])}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
