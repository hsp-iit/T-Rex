"""Train-only statistics (the --taptic_stats file) for a dataset selection. Documented in docs/TOUCH_MODULES.md.

python -I scripts/taptic/make_stats.py --root ROOT [--exclude-sources REPO ...] \
    [--normalization trex_q01_q99_per_step_clipped] [--action-bounds-from PARENT_STATS] --out FILE

The data loader checks that the stats file matches the dataset content (source_identity) and the exact set of
training episodes of the selection, so each (dataset, --taptic_exclude_sources) pair needs its own file.

--action-bounds-from keeps the parent's action and state normalization bounds: a data variant then changes the
data only, not the action scaling (values outside the bounds are clipped, as T-Rex does). Only source_identity,
train_episodes and num_chunks are recomputed; the fresh bounds are reported for comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from taptic.data import NORMALIZATIONS, Source, fit_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--exclude-sources', nargs='*', default=[])
    p.add_argument('--normalization', choices=NORMALIZATIONS, default='trex_q01_q99_per_step_clipped')
    p.add_argument('--action-bounds-from', default='')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    src = Source(a.root)
    e = src.episodes
    excl = set(a.exclude_sources)
    unknown = excl - set(e.source_repo.astype(str))
    if unknown:
        raise SystemExit(f'unknown sources {sorted(unknown)}')
    scope = e[~e.source_repo.astype(str).isin(excl)]
    train = sorted(map(int, scope.index[scope.suggested_split == 'train']))
    st = fit_stats(src, train, normalization=a.normalization)
    blk = st['taptic']
    blk['selection'] = {'root': str(Path(a.root).resolve()), 'exclude_sources': sorted(excl),
                        'train_episodes_by_task': scope[scope.suggested_split == 'train'].source_task.value_counts().to_dict()}
    if a.action_bounds_from:
        parent = json.loads(Path(a.action_bounds_from).read_text())['taptic']
        if parent.get('normalization') != a.normalization:
            raise SystemExit('parent stats use another normalization')
        lo_key, hi_key = ('q01', 'q99') if 'q01' in blk['action'] else ('min', 'max')
        diff = {k: float(np.max(np.abs(np.asarray(blk['action'][k]) - np.asarray(parent['action'][k]))))
                for k in (lo_key, hi_key)}
        blk['fresh_action_bounds_max_abs_diff_vs_parent'] = diff
        blk['action'], blk['state'] = parent['action'], parent['state']
        blk['bounds_from'] = {'path': str(Path(a.action_bounds_from).resolve()), 'source_identity': parent['source_identity']}
        print('fresh vs parent action bounds, max |diff|:', diff)
    Path(a.out).write_text(json.dumps(st, indent=2))
    print(f'STATS_OK {a.out}: identity {blk["source_identity"][:12]}, {len(train)} train episodes, '
          f'{blk["num_chunks"]} chunks, excluded {sorted(excl)}')


if __name__ == '__main__':
    main()
