"""Numerical diagnostics around the executed real-data smoke test.

Runs the existing expert-forward test unchanged; records its tensors at return.
Uses tiny random original Qwen/MoT classes on CPU, never pretrained weights or
training. All outputs go to JSON. Set the same environment as smoke_taptic.sh.
"""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from taptic.actions import chunk_21d, reconstruct_chunk, rotation_angle_error, rotvec_gd_to_matrix


def stats(t):
    dtype = str(t.dtype)
    t = t.detach().float()
    return {'shape': list(t.shape), 'dtype': dtype, 'finite': bool(torch.isfinite(t).all()),
            'min': float(t.min()) if t.numel() else None, 'max': float(t.max()) if t.numel() else None,
            'mean': float(t.mean()) if t.numel() else None,
            'rms': float(t.square().mean().sqrt()) if t.numel() else None}


class StripStrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        return None if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else self.generic_visit(node)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.manual_seed(20261118)
    np.random.seed(20261118)
    spec = importlib.util.spec_from_file_location('taptic_review_tests', ROOT / 'tests/taptic/test_adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setup = module.setup.__wrapped__()
    src, ds, ids = setup
    legacy = next(i for i, (e, _) in enumerate(ds._rows) if e == ids[1])
    # Include a native 76 Hz interval with >1 sample and a legacy interval.
    batch = ds.collate_fn([ds[1], ds[legacy]])
    report = {'branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'seed': 20261118, 'device': 'cpu', 'model': 'random text H64/depth1, vision H32/depth1',
              'episodes': ids, 'batch': {k: stats(v) for k, v in batch.items() if torch.is_tensor(v)},
              'native_stream': {k: stats(v) for k, v in batch['tactile_stream'].items()},
              'errors': {}, 'model_forward': {}, 'official_baseline_comparison': {}}
    for k, v in batch.items():
        if torch.is_tensor(v):
            report['batch'][k]['dtype'] = str(v.dtype)
    report['native_stream_lengths'] = batch['tactile_stream']['lengths'].tolist()
    pe, re, he, ne, be = [], [], [], [], []
    for ep in ids:
        frames = src.frames(ep)
        state, action = np.stack(frames['observation.state']), np.stack(frames.action)
        anchors = src.anchors(ep)
        for t in anchors[::max(1, len(anchors)//20)]:
            c = chunk_21d(state, action, int(t))
            back = reconstruct_chunk(c, state[t, 34:37], state[t, 37:40])
            target = action[t:t+16]
            pe.append(np.max(np.abs(back[:, :3] - target[:, :3])))
            re.append(rotation_angle_error(back[:, 3:6], target[:, 3:6]).max())
            he.append(np.max(np.abs(back[:, 6:] - target[:, 6:])))
            norm = ds.normalize_actions(c)
            ne.append(np.max(np.abs(ds.denormalize_actions(norm) - c)))
            bf = torch.tensor(norm, dtype=torch.bfloat16).float().numpy()
            be.append(np.max(np.abs(ds.denormalize_actions(bf) - c)))
    report['errors'] = {'reconstructed_chunks': len(pe), 'position_max_abs_m': float(max(pe)),
                        'rotation_max_geodesic_rad': float(max(re)), 'hand_max_abs_rad': float(max(he)),
                        'normalization_float32_max_abs_mixed_units': float(max(ne)),
                        'normalization_bfloat16_max_abs_mixed_units': float(max(be))}
    urdf = os.environ.get('TAPTIC_PANDA_URDF')
    if urdf:
        joints = {j.attrib['name']: j for j in ET.parse(urdf).getroot().findall('joint')}
        fkpos, fkrot = [], []
        for ep in src.episodes.groupby('source_task').head(1).index:
            states = np.stack(src.frames(ep)['observation.state'])
            for s in states[::max(1, len(states)//10)]:
                T = np.eye(4)
                for i in range(1, 9):
                    j = joints[f'panda_joint{i}']
                    origin = j.find('origin')
                    O, Q = np.eye(4), np.eye(4)
                    O[:3, 3] = np.fromstring(origin.attrib['xyz'], sep=' ')
                    O[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib['rpy'], sep=' ')).as_matrix()
                    if i <= 7:
                        Q[:3, :3] = Rotation.from_rotvec(np.fromstring(j.find('axis').attrib['xyz'], sep=' ') * np.deg2rad(s[i-1])).as_matrix()
                    T = T @ O @ Q
                fkpos.append(np.max(np.abs(T[:3, 3] - s[34:37])))
                fkrot.append((Rotation.from_matrix(T[:3, :3]).inv() * Rotation.from_matrix(rotvec_gd_to_matrix(s[37:40]))).magnitude())
        report['fk'] = {'source_tasks': 9, 'frames': len(fkpos), 'position_max_abs_m': float(max(fkpos)),
                        'rotation_max_geodesic_rad': float(max(fkrot))}

    capture_key = 'model_forward'

    def trace(frame, event, arg):
        if frame.f_code.co_name != 'test_original_action_and_tactile_forwards':
            return None
        if event == 'return':
            loc = frame.f_locals
            report[capture_key] = {}
            for key in ['emb', 'slow', 'fast', 'pos', 'state', 'x', 'velocity', 'vel', 'xt']:
                report[capture_key][key] = stats(loc[key])
            report[capture_key]['tactile_input'] = stats(loc['b']['tactile_f6s_delayed'])
            report[capture_key]['tau_split'] = loc['tau']
            report[capture_key]['n_action_in_cache'] = loc['nact']
            report[capture_key]['action_mse'] = float((loc['vel'].float() - loc['b']['target'].float()).square().mean())
            target = loc['b']['eps_r'].float() - loc['b']['norm_actions'].float()
            report[capture_key]['tactile_mse'] = float((loc['velocity'].float() - target).square().mean())
            report[capture_key]['cache_length_after_tactile'] = loc['cache'].get_seq_length()
            report[capture_key]['batch_shapes'] = {k: list(v.shape) for k, v in loc['b'].items() if torch.is_tensor(v)}
        return trace

    sys.settrace(trace)
    try:
        module.test_original_action_and_tactile_forwards(setup)
        # The existing test chooses frame 1, which has zero tactile counts.
        # Also execute it unchanged with a real frame containing contact data.
        _, _, _, times, sm, available = ds._episode(ids[0])
        active = None
        for ep, t in ds._rows:
            if ep != ids[0]:
                continue
            i = np.searchsorted(available, times[t], side='right') - 1
            if i >= 0 and np.any(sm.iloc[i]['values']):
                active = (ep, t)
                break
        if active is None:
            raise ValueError('no nonzero native tactile frame in diagnostic scope')
        active_ds = copy.copy(ds)
        active_ds._rows = [active, active]
        capture_key = 'contact_model_forward'
        report['contact_frame'] = {'episode': active[0], 'frame': active[1]}
        module.test_original_action_and_tactile_forwards((src, active_ds, ids))
        assert report[capture_key]['tactile_input']['rms'] > 0
    finally:
        sys.settrace(None)
    for f in ['qwen_vla/modeling_vla.py', 'qwen_vla/modeling_qwen3vl_mot.py',
              'qwen_vla/lerobot_dataset.py', 'utils/lerobot_common.py']:
        ref = subprocess.check_output(['git', 'show', 'full-pipeline:' + f], text=True)
        now = (ROOT / f).read_text()
        norm = lambda s: ast.dump(StripStrings().visit(ast.parse(s)))
        report['official_baseline_comparison'][f] = {'identical_bytes': ref == now,
            'identical_AST_ignoring_docstrings_and_standalone_strings': norm(ref) == norm(now)}
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'errors': report['errors'], 'fk': report.get('fk'),
                      'model_forward': report['model_forward'], 'contact_model_forward': report['contact_model_forward'], 'baseline': report['official_baseline_comparison']}, indent=2))


if __name__ == '__main__':
    main()
