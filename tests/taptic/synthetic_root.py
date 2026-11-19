"""Tiny synthetic TAPTIC root with the real export schema (no videos).

Native episodes: SDK force + taxels at ~76.4 Hz (13.09 ms) on the frame clock,
with pre-roll before the first frame. Legacy episodes: one recorded sample per
10 Hz frame plus synthetic `interpolated=True` rows, zero taxels. Special
episodes inject a sensor reset, a >50 ms read gap and a >150 ms recorder pause.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from taptic.constants import FINGERS, XHAND_JOINTS

EXPECTED = ['ee.x', 'ee.y', 'ee.z', 'ee.wx', 'ee.wy', 'ee.wz']
PERIOD_NS = 13_090_000
FRAME_NS = 100_000_000
# episode -> (split, native, special)
EPISODES = {0: ('train', True, None), 1: ('train', True, 'reset'), 2: ('train', True, 'gap'),
            3: ('train', True, 'pause'), 4: ('train', False, None),
            5: ('val', True, None), 6: ('val', False, None)}
LENGTH = 40


def _frames(special):
    t = 1_000_000_000_000 + np.arange(LENGTH, dtype=np.int64) * FRAME_NS
    if special == 'pause':
        t[20:] += 400_000_000           # recorder pause between frames 19 and 20
    return t


def build(root, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for d in ['meta/episodes/chunk-000', 'data/chunk-000', 'tactile_raw']:
        (root / d).mkdir(parents=True, exist_ok=True)
    feats = {
        'action': {'dtype': 'float32', 'shape': [18], 'names': EXPECTED + ['xhand.action.' + j for j in XHAND_JOINTS]},
        'observation.state': {'dtype': 'float32', 'shape': [40], 'names': (
            [f'panda_joint{i}.pos' for i in range(1, 8)] + ['xhand.joint.' + j for j in XHAND_JOINTS]
            + ['xhand.force.' + f + '.' + a for f in FINGERS for a in 'xyz'] + EXPECTED)},
        'observation.images.left': {'dtype': 'video'}, 'observation.images.wrist': {'dtype': 'video'}}
    (root / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'fps': 10, 'features': feats,
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}))
    order = [f'{f}.taxel_{t:03d}.{a}' for f in FINGERS for t in range(120) for a in 'xyz']
    (root / 'tactile_raw/info.json').write_text(json.dumps({'order': {'values': order}}))

    meta, rows = [], []
    for ep, (split, native, special) in EPISODES.items():
        frames = _frames(special)
        pd.DataFrame({'frame_index': np.arange(LENGTH), 'observation_ns': frames}).to_parquet(
            root / f'tactile_raw/episode_{ep:06d}_frames.parquet')
        # Actions/state: smooth pose near gripper-down, finite everything.
        p = np.cumsum(rng.normal(0, 1e-3, (LENGTH, 3)), 0) + [.4, 0, .3]
        rv = np.cumsum(rng.normal(0, 1e-3, (LENGTH, 3)), 0)
        hand = rng.uniform(0, 1, (LENGTH, 12))
        for t in range(LENGTH):
            state = np.concatenate([rng.normal(0, 10, 7), hand[t], rng.integers(-20, 20, 15), p[t], rv[t]])
            act = np.concatenate([p[min(t + 1, LENGTH - 1)], rv[min(t + 1, LENGTH - 1)], hand[min(t + 1, LENGTH - 1)]])
            rows.append({'episode_index': ep, 'frame_index': t, 'timestamp': np.float32(t / 10),
                         'action': act.astype(np.float32).tolist(), 'observation.state': state.astype(np.float32).tolist()})
        if native:
            read = np.arange(frames[0] - 200_000_000, frames[-1] + 50_000_000, PERIOD_NS, dtype=np.int64)
            if special == 'gap':
                read = read[(read < frames[10]) | (read > frames[10] + 80_000_000)]
            n = len(read)
            reset = np.zeros(n, np.int64)
            if special == 'reset':
                reset[read > frames[15]] = 1
            force = np.cumsum(rng.integers(-3, 4, (n, 15)), 0).astype(np.float64)
            values = rng.integers(0, 50, (n, 1800)).astype(np.float64)
            interp = np.zeros(n, bool)
            avail = read + 500_000
        else:
            base = frames - 1_000_000
            interp_t = (base[:-1, None] + np.arange(1, 8) * 12_000_000).ravel()
            read = np.sort(np.concatenate([base, interp_t]))
            interp = ~np.isin(read, base)
            n = len(read)
            reset = np.zeros(n, np.int64)
            force = np.cumsum(rng.integers(-3, 4, (n, 15)), 0).astype(np.float64)
            values = np.zeros((n, 1800))
            avail = read.copy()
            avail[interp] += 30_000_000          # interpolation becomes available later
        pd.DataFrame({'read_end_ns': read, 'available_ns': avail,
                      'force': list(force), 'values': list(values),
                      'sequence': np.arange(n, dtype=np.int64), 'reset_sequence': reset,
                      'read_errors': np.zeros(n, np.int64), 'dropped': np.zeros(n, np.int64),
                      'interpolated': interp}).to_parquet(root / f'tactile_raw/episode_{ep:06d}_right.parquet')
        meta.append({'episode_index': ep, 'length': LENGTH, 'suggested_split': split,
                     'source_task': f'task{ep % 3}', 'source_episode_index': ep,
                     'tactile_interpolated': not native, 'caption': f'synthetic task {ep}'})
    pd.DataFrame(meta).to_parquet(root / 'meta/episodes/chunk-000/file-000.parquet')
    pd.DataFrame(rows).to_parquet(root / 'data/chunk-000/file-000.parquet')
    return root
