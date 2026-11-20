"""Build the data of the gate review tool (blind human annotation vs risk gate).

For each selected episode it writes
  video/ep_XXXXXX.mp4  both cameras side by side (fixed camera | wrist), 10 fps,
                       one video frame per recorded frame (recorder pauses skipped)
  data/ep_XXXXXX.json  real frame times, per-finger |F| (SDK counts), M / R / R_pos /
                       R_neg / V / r on a 50 ms grid (causal, same code as training),
                       and the gate decisions of two r variants evaluated at every
                       native read (~76 Hz):
                         new = vector R  (docs/taptic_ours/risk_stats.json)
                         old = positive-only R (docs/taptic_ours/risk_stats_positive.json)
  index.json           the episode list.

python scripts/build_taptic_gate_review.py --root ROOT --out DIR [--episodes 599 93 ...]
"""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from taptic.data import CAMERAS, Source  # noqa: E402
from taptic.force_history import build_force_track, read_force_samples  # noqa: E402
from taptic.risk import RiskConfig, RiskStats, combine_risk, risk_features  # noqa: E402
from taptic.risk_gate import GateConfig, RiskGate  # noqa: E402
from taptic.video import VideoReader  # noqa: E402

MS = 1_000_000


def pick_episodes(src, max_len=700):
    e = src.episodes
    v = e[(e.suggested_split == 'val') & ~e.tactile_interpolated.astype(bool)]
    out = []
    for _, g in v.groupby('source_task'):
        cand = g[g.length <= max_len]
        out.append(int((cand.length - g.length.median()).abs().idxmin()))
    return out


def encode_video(src, ep, path, size=(320, 240)):
    row = src.episodes.loc[ep]
    reader = VideoReader()
    n = int(row['length'])
    out = av.open(str(path), mode='w', options={'movflags': '+faststart'})
    st = out.add_stream('libx264', rate=10)
    st.width, st.height, st.pix_fmt = 2 * size[0], size[1], 'yuv420p'
    st.options = {'crf': '30', 'preset': 'medium', 'g': '10'}
    for t in range(n):
        tiles = []
        for cam in CAMERAS:
            vp = src.info['video_path'].format(video_key=cam, chunk_index=int(row[f'videos/{cam}/chunk_index']),
                                               file_index=int(row[f'videos/{cam}/file_index']))
            ts = float(row[f'videos/{cam}/from_timestamp']) + t / 10
            tiles.append(cv2.resize(reader.frame(str(src.root / vp), ts, 10), size, interpolation=cv2.INTER_AREA))
        fr = av.VideoFrame.from_ndarray(np.hstack(tiles), format='rgb24')
        fr.pts, fr.time_base = t, Fraction(1, 10)
        for pkt in st.encode(fr):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    reader.close()
    return n


def segments(t_ms, bins):
    segs, start = [], 0
    for j in range(1, len(bins) + 1):
        if j == len(bins) or bins[j] != bins[start]:
            segs.append([round(t_ms[start], 1), round(t_ms[j], 1) if j < len(bins) else round(t_ms[-1], 1),
                         int(bins[start])])
            start = j
    return segs


def episode_data(src, ep, stats_new, stats_old, h, dwell_ms):
    frames = src.clocks(ep)
    t0, t1 = int(frames[0]), int(frames[-1])
    sm, av_ = read_force_samples(src.root, ep)
    tr = build_force_track(sm, av_, frames)
    cfg_new, cfg_old = RiskConfig(), RiskConfig(rate_mode='positive')
    grid = np.arange(t0, t1 + 1, 50 * MS, dtype=np.int64)
    f = risk_features(tr, grid, stats_new, cfg_new)
    r_old = np.where(f['risk_valid'], combine_risk(f['M'], f['R_pos'], f['V'], cfg_new.coefficients), 0.)
    rd = lambda x, d=3: np.round(np.asarray(x, float), d).tolist()
    ok = tr['segment'] >= 0
    ticks = tr['available_ns'][ok & (tr['available_ns'] >= t0) & (tr['available_ns'] <= t1)]
    gates = {}
    for name, stats, cfg in (('new', stats_new, cfg_new), ('old', stats_old, cfg_old)):
        g = RiskGate(GateConfig.from_risk_stats(stats, hysteresis=h, min_dwell_ms=dwell_ms), keep_log=False)
        ff = risk_features(tr, ticks, stats, cfg)
        bins = [{.2: 0, .4: 1, .6: 2}[g.select(int(t), ff['risk'][j], ff['risk_valid'][j])['selected_split']]
                for j, t in enumerate(ticks)]
        tm = (ticks - t0) / 1e6
        gates[name] = {'segments': segments(tm, bins), 'thresholds': list(GateConfig.from_risk_stats(stats).thresholds),
                       'occupancy': np.bincount(bins, minlength=3).tolist(), 'ticks': len(bins),
                       'transitions': int(np.sum(np.diff(bins) != 0))}
    e = src.episodes.loc[ep]
    return {'episode': ep, 'task': str(e.source_task), 'caption': str(e.caption),
            'duration_ms': round((t1 - t0) / 1e6, 1), 'frame_ms': rd((frames - t0) / 1e6, 1),
            'grid': {'t_ms': rd((grid - t0) / 1e6, 1),
                     'F': [rd(f['magnitude'][:, k], 1) for k in range(5)],
                     'M': rd(f['M']), 'R': rd(f['R']), 'R_pos': rd(f['R_pos']), 'R_neg': rd(f['R_neg']),
                     'V': rd(f['V']), 'r_new': rd(f['risk'], 4), 'r_old': rd(r_old, 4),
                     'valid': f['risk_valid'].astype(int).tolist()},
            'gate': {'hysteresis': h, 'min_dwell_ms': dwell_ms, **gates},
            'video': f'video/ep_{ep:06d}.mp4'}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, nargs='*')
    p.add_argument('--risk_stats', default=str(REPO / 'docs/taptic_ours/risk_stats.json'))
    p.add_argument('--risk_stats_old', default=str(REPO / 'docs/taptic_ours/risk_stats_positive.json'))
    p.add_argument('--hysteresis', type=float, default=.005)
    p.add_argument('--min_dwell_ms', type=float, default=50.)
    a = p.parse_args()
    src = Source(a.root)
    out = Path(a.out)
    (out / 'video').mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    stats_new, stats_old = RiskStats.load(a.risk_stats), RiskStats.load(a.risk_stats_old)
    eps = a.episodes or pick_episodes(src)
    index = []
    for ep in eps:
        if src.episodes.loc[ep, 'suggested_split'] != 'val':
            print(f'warning: episode {ep} is not a validation episode')
        n = encode_video(src, ep, out / 'video' / f'ep_{ep:06d}.mp4')
        d = episode_data(src, ep, stats_new, stats_old, a.hysteresis, a.min_dwell_ms)
        (out / 'data' / f'ep_{ep:06d}.json').write_text(json.dumps(d, separators=(',', ':')))
        index.append({k: d[k] for k in ('episode', 'task', 'caption', 'duration_ms')} | {'frames': n})
        print(ep, d['task'], n, 'frames', {k: d['gate'][k]['occupancy'] for k in ('new', 'old')})
    (out / 'index.json').write_text(json.dumps(index, indent=1))


if __name__ == '__main__':
    main()
