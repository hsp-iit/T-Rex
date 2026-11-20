"""Offline, causal contact-dynamics exploration; never used by model forward.

Use recorded SDK aggregates, not taxel sums. Units are SDK counts, not N.
Event summaries are retrospective diagnostics, not policy inputs or risk labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taptic.constants import FINGERS
from taptic.data import Source
from taptic.timing import build_quality

WINDOWS = (50, 100, 200)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def quant(x):
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    return dict(zip(('p00', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'),
                    np.quantile(x, (0, .1, .25, .5, .75, .9, .95, .99, 1)).tolist())) if len(x) else {}


def causal_features(t, force, break_before):
    """Arrays keep every read, including equal payloads. Invalid rates are zero + mask.

    Rolling means integrate the past zero-order-held magnitude in physical time.
    No future interpolation. Windows (t-W,t] must fit wholly inside one segment.
    """
    t, force = np.asarray(t, np.int64), np.asarray(force, float)
    if force.shape != (len(t), 5, 3) or np.any(np.diff(t) <= 0):
        raise ValueError('expected strictly increasing times and force [N,5,3]')
    if not np.isfinite(force).all():
        raise ValueError('nonfinite force must be excluded and segmented first')
    n = len(t)
    breaks = np.asarray(break_before, bool).copy()
    if len(breaks) != n:
        raise ValueError('break mask length')
    if n:
        breaks[0] = True
    mag = np.linalg.norm(force, axis=2)
    delta = np.zeros_like(mag)
    rate = np.zeros_like(mag)
    valid = ~breaks
    if n > 1:
        delta[1:] = np.diff(mag, axis=0)
        delta[~valid] = 0
        rate[1:] = delta[1:] / (np.diff(t)[:, None] / 1e9)
    result = dict(magnitude=mag, magnitude_change=delta,
                  positive_change=np.maximum(delta, 0), negative_change=np.maximum(-delta, 0), signed_rate=rate,
                  positive_rate=np.maximum(rate, 0), negative_rate=np.maximum(-rate, 0),
                  derivative_valid=valid, max_magnitude=mag.max(axis=1),
                  max_positive_rate=np.maximum(rate, 0).max(axis=1), windows={})
    starts = np.flatnonzero(breaks)
    for ms in WINDOWS:
        w = ms / 1000
        out = {k: np.zeros_like(mag) for k in ('mean_magnitude', 'peak_magnitude',
                'positive_variation_rate', 'negative_variation_rate', 'secant_rate', 'peak_positive_rate')}
        full = np.zeros(n, bool)
        for s, e in zip(starts, np.r_[starts[1:], n]):
            ts, m = t[s:e], mag[s:e]
            left = ts - ms * 1_000_000
            j = np.searchsorted(ts, left, side='right') - 1
            ok = j >= 0
            j = np.maximum(j, 0)
            # Integral of the held past value, exactly clipped at the left boundary.
            integral = np.vstack((np.zeros((1, 5)), np.cumsum(m[:-1] * (np.diff(ts)[:, None]/1e9), axis=0)))
            ileft = integral[j] + m[j] * ((left-ts[j])[:, None]/1e9)
            out['mean_magnitude'][s:e] = (integral - ileft) / w
            out['secant_rate'][s:e] = (m-m[j]) / w
            for direction, key in ((1, 'positive_variation_rate'), (-1, 'negative_variation_rate')):
                increments = np.maximum(direction * delta[s:e], 0)
                prefix = np.cumsum(increments, axis=0)
                out[key][s:e] = (prefix-prefix[j]) / w
            index = pd.to_timedelta(ts-ts[0], unit='ns')
            out['peak_magnitude'][s:e] = pd.DataFrame(m, index=index).rolling(
                pd.Timedelta(milliseconds=ms), closed='right').max().to_numpy()
            # Include a value read before the left edge and held inside the window.
            out['peak_magnitude'][s:e] = np.maximum(out['peak_magnitude'][s:e], m[j])
            out['peak_positive_rate'][s:e] = pd.DataFrame(np.maximum(rate[s:e], 0), index=index).rolling(
                pd.Timedelta(milliseconds=ms), closed='right').max().to_numpy()
            full[s:e] = ok
        for v in out.values():
            v[~full] = 0
        result['windows'][ms] = dict(out, valid=full)
    return result


def contact_events(t, magnitude, breaks, high):
    """Hysteretic activity proxies; high from positive TRAIN quantiles, low=high/2.

    Peak/duration/rise-time need later observations and are OFFLINE descriptors.
    Censored events are retained and explicitly flagged.
    """
    rows = []
    starts = np.flatnonzero(breaks)
    for s, e in zip(starts, np.r_[starts[1:], len(t)]):
        for f in range(5):
            start = None
            for i in range(s, e):
                if start is None and magnitude[i, f] >= high[f]:
                    start = i
                if start is not None and (magnitude[i, f] <= high[f]/2 or i == e-1):
                    end = i
                    p = start + int(np.argmax(magnitude[start:end+1, f]))
                    left_censored = start == s
                    right_censored = i == e-1 and magnitude[i, f] > high[f]/2
                    baseline = magnitude[start-1, f] if not left_censored else magnitude[start, f]
                    excursion = magnitude[p, f]-baseline
                    rise1090 = None
                    if not left_censored and excursion > 0:
                        y = magnitude[start:p+1, f]
                        a = start+int(np.flatnonzero(y >= baseline+.1*excursion)[0])
                        b = start+int(np.flatnonzero(y >= baseline+.9*excursion)[0])
                        rise1090 = (int(t[b])-int(t[a])) / 1e9
                    rows.append(dict(finger=FINGERS[f], start_ns=int(t[start]), end_ns=int(t[end]),
                                     peak_ns=int(t[p]), peak=float(magnitude[p, f]),
                                     duration_s=(int(t[end])-int(t[start]))/1e9,
                                     rise_to_peak_s=(int(t[p])-int(t[start]))/1e9,
                                     rise_10_90_s=rise1090, left_censored=left_censored,
                                     right_censored=right_censored))
                    start = None
    return rows


def weighted_quantile(x, weights, q):
    order = np.argsort(x)
    x, weights = np.asarray(x)[order], np.asarray(weights)[order]
    return float(x[min(np.searchsorted(np.cumsum(weights), q*weights.sum()), len(x)-1)])


def fit_normalization(records):
    """Train-only scales; sample-pooled and equal-episode-weight q99 alternatives."""
    train = [r for r in records if r['split'] == 'train']
    m = np.concatenate([r['features']['magnitude'] for r in train])
    d = np.concatenate([r['features']['positive_rate'][r['features']['derivative_valid']] for r in train])
    weights = np.concatenate([np.full(len(r['t']), 1/len(r['t'])) for r in train])
    result = dict(train_episodes=[r['episode'] for r in train], magnitude={}, positive_rate={}, activity_thresholds={})
    for name, a in (('magnitude', m), ('positive_rate', d)):
        result[name]['pooled_q99'] = max(float(np.quantile(a, .99)), 1.)
        result[name]['per_finger_q99'] = np.maximum(np.quantile(a, .99, axis=0), 1.).tolist()
        result[name]['positive_only_per_finger_q99'] = [max(float(np.quantile(a[a[:, f] > 0, f], .99)), 1.)
                if np.any(a[:, f] > 0) else 1. for f in range(5)]
    result['magnitude']['equal_episode_q99'] = [max(weighted_quantile(m[:, f], weights, .99), 1.) for f in range(5)]
    for q in (.1, .25, .5):
        result['activity_thresholds'][str(q)] = [float(np.quantile(m[m[:, f] > 0, f], q))
                if np.any(m[:, f] > 0) else 1. for f in range(5)]
    result['windows'] = {}
    for ms in WINDOWS:
        a = np.concatenate([r['features']['windows'][ms]['positive_variation_rate'][
            r['features']['windows'][ms]['valid']] for r in train])
        result['windows'][str(ms)] = {'positive_variation_rate_positive_q99': [
            max(float(np.quantile(a[a[:, f] > 0, f], .99)), 1.) if np.any(a[:, f] > 0) else 1.
            for f in range(5)]}
    return result


def normalized(x, scale, log=False):
    return np.clip(np.log1p(x)/np.log1p(scale) if log else x/scale, 0, 1)


def load_records(source):
    records, audit, hashes = [], [], {}
    seen = {}
    for ep, row in source.episodes.iterrows():
        path = source.root/f'tactile_raw/episode_{ep:06d}_right.parquet'
        fp = source.root/f'tactile_raw/episode_{ep:06d}_frames.parquet'
        hashes[str(path.relative_to(source.root))] = digest(path)
        hashes[str(fp.relative_to(source.root))] = digest(fp)
        native = not bool(row.tactile_interpolated)
        # Legacy audit reads no WEART data and never promotes repeats to native data.
        if not native:
            counts = pd.read_parquet(path, columns=['interpolated'])
            audit.append(dict(episode=int(ep), native=False, rows=len(counts),
                              synthetic_rows=int(counts.interpolated.sum())))
            continue
        sm, available = source.tactile(ep)
        clock = source.clocks(ep)
        quality = build_quality(sm, available, True)
        t = quality['read_end_ns']
        force = np.stack(sm.force).reshape(-1, 5, 3).astype(float)
        finite = np.isfinite(force).all(axis=(1, 2))
        keep = finite & (t >= clock[0]) & (t <= clock[-1])
        pause = np.zeros(len(t), bool)
        for i in np.flatnonzero(np.diff(clock) > 150_000_000):
            pause |= (t > clock[i]) & (t < clock[i+1])
        keep &= ~pause
        # A source recording can be split into multiple object/action episodes.
        key = (row.source_repo, row.source_task, int(row.source_episode_index))
        used = seen.setdefault(key, set())
        duplicate = np.array([int(v) in used for v in t], bool) & keep
        keep &= ~duplicate
        idx = np.flatnonzero(keep)
        used.update(map(int, t[idx]))
        breaks = quality['break_before'][idx].copy()
        if len(idx):
            breaks[0] = True
            breaks[1:] |= np.diff(idx) != 1
        valid = ~breaks
        dt = np.diff(t)[~quality['break_before'][1:]]/1e6
        audit.append(dict(episode=int(ep), native=True, rows=len(sm), kept=len(idx),
            nonfinite=int((~finite).sum()), pre_post_roll=int(((t < clock[0]) | (t > clock[-1])).sum()),
            noninteger_components=int(np.count_nonzero(force[finite] != np.rint(force[finite]))),
            pause_rows=int(pause.sum()), duplicate_rows=int(duplicate.sum()),
            resets=int(np.count_nonzero(np.diff(sm.reset_sequence))),
            counter_error_changes=int(np.count_nonzero(np.diff(sm.read_errors))),
            counter_drop_changes=int(np.count_nonzero(np.diff(sm.dropped))),
            read_gaps=int(np.count_nonzero(np.diff(t) > 50_000_000)),
            availability_delayed_rows=int(np.count_nonzero(available > t)),
            segments=int(breaks.sum()), interval_ms=quant(dt)))
        if not len(idx):
            continue
        # Current data have read completion as availability; reject an unsupported
        # delayed/out-of-order stream rather than claim these windows are causal.
        if np.any(available[idx] != t[idx]):
            raise ValueError('delayed availability requires arrival-ordered feature analysis')
        features = causal_features(t[idx], force[idx], breaks)
        records.append(dict(episode=int(ep), task=row.source_task, object=row['object'],
            split=row.suggested_split, source_episode=int(row.source_episode_index),
            t=t[idx], breaks=breaks, force=force[idx], features=features,
            duration_s=float(np.diff(t[idx])[valid[1:]].sum()/1e9) if len(idx)>1 else 0.,
            repeated=int((np.all(force[idx][1:] == force[idx][:-1], axis=(1, 2)) & valid[1:]).sum())))
    return records, audit, hashes


def summarize(records, norm, out):
    episode_rows, finger_rows, window_rows, event_rows, comparisons, threshold_rows = [], [], [], [], [], []
    scales_m = np.asarray(norm['magnitude']['per_finger_q99'])
    for r in records:
        f = r['features']
        valid = f['derivative_valid']
        base = {k: r[k] for k in ('episode', 'task', 'object', 'split', 'source_episode')}
        episode_rows.append(dict(base, samples=len(r['t']), duration_s=r['duration_s'],
            derivative_valid_pairs=int(valid.sum()),
            repeated_fraction=r['repeated']/max(int(valid.sum()), 1), segments=int(r['breaks'].sum()),
            **{'read_interval_ms_'+k:v for k,v in quant(np.diff(r['t'])[valid[1:]]/1e6).items()},
            **{'magnitude_'+k:v for k,v in quant(f['max_magnitude']).items()},
            **{'rate_'+k:v for k,v in quant(f['max_positive_rate'][valid]).items()}))
        for j, finger in enumerate(FINGERS):
            finger_rows.append(dict(base, finger=finger,
                nonzero_fraction=float(np.mean(f['magnitude'][:, j]>0)),
                **{'magnitude_'+k:v for k,v in quant(f['magnitude'][:, j]).items()},
                **{'positive_change_'+k:v for k,v in quant(np.maximum(f['magnitude_change'][valid,j],0)).items()},
                **{'negative_change_'+k:v for k,v in quant(np.maximum(-f['magnitude_change'][valid,j],0)).items()},
                **{'rate_'+k:v for k,v in quant(f['signed_rate'][valid, j]).items()}))
        for ms, w in f['windows'].items():
            mask = w['valid']
            if not mask.any():
                continue
            # M uses current magnitude, R uses variation/W. Both causal and normalized per finger.
            M = normalized(f['magnitude'][mask], scales_m).max(axis=1)
            scales_r = np.asarray(norm['windows'][str(ms)]['positive_variation_rate_positive_q99'])
            R = normalized(w['positive_variation_rate'][mask], scales_r).max(axis=1)
            C = 1-(1-M)*(1-R)  # comparison score, not calibrated probability
            high = np.asarray(norm['activity_thresholds']['0.25'])
            active = (f['magnitude'][mask] >= high).any(axis=1)
            wrow = dict(base, window_ms=ms, full_window_samples=int(mask.sum()),
                magnitude_rate_spearman=float(pd.Series(M).corr(pd.Series(R), method='spearman'))
                    if np.std(M)>0 and np.std(R)>0 else None,
                sustained_active_fraction=float(np.mean(active & (w['positive_variation_rate'][mask].max(axis=1)==0))),
                rising_fraction=float(np.mean(w['positive_variation_rate'][mask].max(axis=1)>0)))
            for name in ('mean_magnitude', 'peak_magnitude', 'positive_variation_rate', 'negative_variation_rate',
                         'secant_rate', 'peak_positive_rate'):
                wrow.update({name+'_'+k:v for k,v in quant(w[name][mask]).items()})
                wrow.update({'max_'+name+'_'+k:v for k,v in quant(w[name][mask].max(axis=1)).items()})
            for name, values in (('M', M), ('R', R), ('combined', C)):
                wrow.update({name+'_'+k:v for k,v in quant(values).items()})
            window_rows.append(wrow)
        # Compare normalization on the same instantaneous magnitudes; these cutoffs
        # are diagnostics, not proposed tau thresholds.
        mag = f['magnitude']
        for name, scale, log in [('pooled_linear', norm['magnitude']['pooled_q99'], False),
                ('finger_linear', scales_m, False), ('finger_log', scales_m, True),
                ('episode_balanced_linear', np.asarray(norm['magnitude']['equal_episode_q99']), False)]:
            score = normalized(mag, scale, log).max(axis=1)
            comparisons.append(dict(base, normalization=name, mean=float(score.mean()),
                median=float(np.median(score)), fraction_ge_half=float(np.mean(score >= .5)),
                saturated_fraction=float(np.mean(score >= 1))))
        for q, thresholds in norm['activity_thresholds'].items():
            active = mag >= np.asarray(thresholds)
            # Occupancy weighted by valid interval duration of the held past value.
            dt = np.diff(r['t'])/1e9
            weights = dt*valid[1:]
            ev = contact_events(r['t'], mag, r['breaks'], np.asarray(thresholds))
            threshold_rows.append(dict(base, quantile=q, events=len(ev),
                onsets=sum(not x['left_censored'] for x in ev),
                releases=sum(not x['right_censored'] for x in ev),
                sample_active_fraction=float(active.any(axis=1).mean()),
                time_active_fraction=float(np.sum(active[:-1].any(axis=1)*weights)/weights.sum()) if weights.sum() else None))
            if q == '0.25':
                event_rows.extend(dict(base, **x) for x in ev)
    frames = {}
    for name, rows in [('episodes', episode_rows), ('per_episode_finger', finger_rows), ('windows', window_rows),
                       ('events', event_rows), ('normalization_comparison', comparisons), ('threshold_sensitivity', threshold_rows)]:
        frames[name] = pd.DataFrame(rows)
        frames[name].to_csv(out/f'{name}.csv', index=False)
    # Exact pooled per-task/per-object distributions, no capped plotting sample.
    for grouping in ('task', 'object'):
        rows = []
        for key in sorted({r[grouping] for r in records}):
            group = [r for r in records if r[grouping] == key]
            a = np.concatenate([r['features']['magnitude'] for r in group])
            d = np.concatenate([r['features']['signed_rate'][r['features']['derivative_valid']] for r in group])
            for j, finger in enumerate(FINGERS):
                rows.append({grouping:key, 'finger':finger, 'episodes':len(group), 'samples':len(a),
                    **{'magnitude_'+k:v for k,v in quant(a[:,j]).items()},
                    **{'signed_rate_'+k:v for k,v in quant(d[:,j]).items()},
                    **{'positive_rate_'+k:v for k,v in quant(np.maximum(d[:,j],0)).items()}})
        frames['by_'+grouping+'_finger'] = pd.DataFrame(rows)
        frames['by_'+grouping+'_finger'].to_csv(out/f'by_{grouping}_finger.csv', index=False)
    return frames


def figures(records, norm, frames, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'axes.grid':True, 'grid.alpha':.2, 'figure.dpi':120})
    figdir = out/'figures'
    figdir.mkdir(exist_ok=True)
    representatives = []
    for task in sorted({r['task'] for r in records}):
        group = [r for r in records if r['task']==task and r['split']=='train']
        if task == 'clearing_table':
            # Three longest examples with distinct annotated objects.
            for r in sorted(group, key=lambda r:r['duration_s'], reverse=True):
                if r['object'] not in [x['object'] for x in representatives if x['task']==task]:
                    representatives.append(r)
                if sum(x['task']==task for x in representatives)==3:
                    break
        else:
            representatives.append(max(group, key=lambda r:r['duration_s']))
            # Also show a typical-force example among episodes with >= median duration.
            # Keep the longest even if almost entirely quiet, rather than hide it.
            longer=[r for r in group if r['duration_s'] >= np.median([x['duration_s'] for x in group])]
            target=np.median([np.quantile(r['features']['max_magnitude'],.95) for r in longer])
            typical=min(longer,key=lambda r:abs(np.quantile(r['features']['max_magnitude'],.95)-target))
            if typical['episode'] != representatives[-1]['episode']:
                representatives.append(typical)
    feature_dir=out/'representative_features'
    feature_dir.mkdir(exist_ok=True)
    for r in representatives:
        f = r['features']
        arrays=dict(read_end_ns=r['t'],available_ns=r['t'],sdk_force=r['force'],
                    break_before=r['breaks'],activity_q25=f['magnitude']>=np.asarray(norm['activity_thresholds']['0.25']))
        arrays.update({k:v for k,v in f.items() if k!='windows'})
        for ms,w in f['windows'].items():
            arrays.update({f'w{ms}_{k}':v for k,v in w.items()})
        np.savez_compressed(feature_dir/f"episode_{r['episode']:06d}.npz",**arrays)
        # NaN breaks only for plotting; never connect across reset/gap boundaries.
        t = (r['t']-r['t'][0])/1e9
        mag, rate = f['magnitude'].copy(), f['signed_rate'].copy()
        mag[r['breaks']] = np.nan
        rate[~f['derivative_valid']] = np.nan
        fig, axes = plt.subplots(4,1,figsize=(12,9),sharex=True)
        axes[0].plot(t, mag, label=list(FINGERS)); axes[0].legend(ncol=5)
        axes[0].set_ylabel('SDK magnitude [counts]')
        axes[1].plot(t, rate); axes[1].set_ylabel('Signed rate [counts/s]')
        for ms, w in f['windows'].items():
            y = w['positive_variation_rate'].max(axis=1).copy(); y[~w['valid']] = np.nan
            axes[2].plot(t, y,label=f'{ms} ms')
        axes[2].legend(); axes[2].set_ylabel('Max positive variation/W')
        w = f['windows'][100]; mask=w['valid']
        M=normalized(f['magnitude'],np.asarray(norm['magnitude']['per_finger_q99'])).max(axis=1)
        R=normalized(w['positive_variation_rate'],np.asarray(norm['windows']['100']['positive_variation_rate_positive_q99'])).max(axis=1)
        for name,y in [('Magnitude',M),('Rate (100 ms)',R),('Combined',1-(1-M)*(1-R))]:
            y=y.copy();y[~mask]=np.nan;axes[3].plot(t,y,label=name)
        axes[3].legend();axes[3].set_ylabel('Exploratory score');axes[3].set_xlabel('Elapsed recorded time [s]')
        fig.suptitle(f"Episode {r['episode']} / {r['task']} / {r['object']} (train)")
        fig.tight_layout();fig.savefig(figdir/f"episode_{r['episode']:06d}.png");plt.close(fig)
    tasks=sorted({r['task'] for r in records})
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for task in tasks:
        group=[r for r in records if r['task']==task]
        for ax,key in zip(axes,('magnitude','positive_rate')):
            data=np.concatenate([r['features'][key][r['features']['derivative_valid']] for r in group]).ravel()
            data=np.sort(data[data>0]); ax.plot(data,np.arange(1,len(data)+1)/len(data),label=task)
    for ax,label in zip(axes,('Magnitude [counts], positive only','Positive magnitude rate [counts/s], positive only')):
        ax.set_xscale('log');ax.set_xlabel(label);ax.set_ylabel('Empirical CDF')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'task_distributions.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,column in zip(axes,('magnitude_p95','positive_rate_p99')):
        table=frames['by_task_finger'].pivot(index='task',columns='finger',values=column).reindex(columns=FINGERS)
        im=ax.imshow(np.log1p(table.values),aspect='auto');ax.set_xticks(range(5),FINGERS)
        ax.set_yticks(range(len(table)),table.index);ax.set_title(column+' (color=log1p)')
        for i in range(len(table)):
            for j in range(5):ax.text(j,i,f'{table.iloc[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white')
        fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(figdir/'finger_task_distributions.png');plt.close(fig)
    # Event examples from different tasks, selected for excursion, not outcome.
    ev=frames['events']; complete=ev[~ev.left_censored & ~ev.right_censored]
    examples=[]
    for task in tasks:
        e=complete[complete.task==task].sort_values('peak',ascending=False)
        if len(e):examples.append(e.iloc[0])
    fig,axes=plt.subplots(len(examples),1,figsize=(12,2.6*len(examples)),squeeze=False)
    high=np.asarray(norm['activity_thresholds']['0.25'])
    for ax,e in zip(axes[:,0],examples):
        r=next(r for r in records if r['episode']==e.episode);j=FINGERS.index(e.finger)
        # Restrict plot to event segment, never visually connect a reset.
        segment=np.cumsum(r['breaks']);k=np.searchsorted(r['t'],e.start_ns)
        select=(r['t']>=e.start_ns-200_000_000)&(r['t']<=e.end_ns+200_000_000)&(segment==segment[k])
        ax.step((r['t'][select]-e.start_ns)/1e9,r['features']['magnitude'][select,j],where='post')
        ax.axhline(high[j],color='orange',ls='--',label='Activity on threshold')
        ax.axhline(high[j]/2,color='gray',ls=':',label='Activity off threshold')
        ax.axvspan(0,e.duration_s,alpha=.1,color='orange');ax.set_title(f'{e.task}, episode {e.episode}, {e.finger}, peak={e.peak:.1f}, duration={e.duration_s:.3f}s')
        ax.set_ylabel('SDK counts');ax.set_xlabel('Seconds from candidate onset')
    if examples:axes[0,0].legend()
    fig.tight_layout();fig.savefig(figdir/'contact_event_examples.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for task,g in frames['windows'].groupby('task'):
        # Equal episode mean of per-episode p95, explicitly not pooled quantiles.
        y=g.groupby('window_ms').max_positive_variation_rate_p95.mean()
        axes[0].plot(y.index,y.values,marker='o',label=task)
    axes[0].set_xlabel('Causal window [ms]');axes[0].set_ylabel('Mean episode p95 positive variation/W [counts/s]');axes[0].legend(fontsize=7)
    nc=frames['normalization_comparison'].groupby(['task','normalization']).fraction_ge_half.mean().unstack()
    nc.plot.bar(ax=axes[1]);axes[1].set_ylabel('Mean episode fraction score >= 0.5 (diagnostic)')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/'feature_sensitivity.png');plt.close(fig)
    # Scores are feature comparisons only; 0.5 has no physical or tau meaning.
    w=frames['windows'];table=w[w.window_ms==100].groupby('task')[['M_p95','R_p95','combined_p95']].median()
    ax=table.plot.bar(figsize=(11,5));ax.set_ylabel('Median episode p95 score (100 ms)')
    ax.figure.tight_layout();ax.figure.savefig(figdir/'candidate_score_comparison.png');plt.close(ax.figure)
    objects={r['object'] for r in records if r['task']=='clearing_table'}
    obj=frames['by_object_finger'];table=obj[obj.object.isin(objects)].pivot(index='object',columns='finger',values='magnitude_p95').reindex(columns=FINGERS)
    table=table.loc[table.max(axis=1).sort_values().index]
    ax=table.plot.barh(figsize=(10,8));ax.set_xlabel('Pooled magnitude p95 [SDK counts]');ax.set_title('clearing_table: annotated object groups')
    ax.figure.tight_layout();ax.figure.savefig(figdir/'clearing_objects.png');plt.close(ax.figure)
    return [r['episode'] for r in representatives]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True);p.add_argument('--out',default='docs/taptic_risk_analysis')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    root=Path(args.root)
    meta_paths=list((root/'meta/episodes').rglob('*.parquet'))+[root/'meta/info.json',root/'tactile_raw/info.json']
    meta_hashes={str(path.relative_to(root)):digest(path) for path in meta_paths}
    source=Source(root)
    cols=['source_task','source_episode_index','object','object_needs_review','suggested_split','tactile_interpolated','length','caption']
    inventory=source.episodes[cols].reset_index()
    inventory.to_csv(out/'inventory.csv',index=False)
    records,audit,hashes=load_records(source)
    print(f'Loaded {len(records)} native episodes / {sum(len(r["t"]) for r in records)} retained reads',flush=True)
    norm=fit_normalization(records)
    (out/'train_normalization.json').write_text(json.dumps(norm,indent=2))
    frames=summarize(records,norm,out)
    reps=figures(records,norm,frames,out)
    pd.DataFrame(audit).drop(columns=['interval_ms']).to_csv(out/'quality.csv',index=False)
    # Rehash all inspected sidecars and metadata: no modification of native or legacy data.
    hashes.update(meta_hashes)
    unchanged=all(digest(source.root/path)==value for path,value in hashes.items())
    summary=dict(root=str(source.root),source_identity=source.identity,all_source_files_unchanged=unchanged,
        source_sha256=hashes,native_episodes=len(records),legacy_episodes=sum(not a['native'] for a in audit),
        synthetic_rows_excluded=sum(a.get('synthetic_rows',0) for a in audit),
        retained_reads=sum(len(r['t']) for r in records),valid_duration_s=sum(r['duration_s'] for r in records),
        derivative_valid_pairs=sum(int(r['features']['derivative_valid'].sum()) for r in records),
        repeated_pairs=sum(r['repeated'] for r in records),
        retained_read_interval_ms=quant(np.concatenate([np.diff(r['t'])[r['features']['derivative_valid'][1:]]/1e6 for r in records])),
        representative_episodes=reps,quality=audit,
        definitions=dict(units='SDK counts / SDK counts per second; not newtons',
            axes='SDK xyz; physical axes/surface normals unverified',
            derivative='signed difference of vector magnitude / actual read_end interval; masked at boundaries',
            windows='(t-W,t], causal held integrals and past increments; full same-segment context required',
            events='positive-training-q25 hysteretic activity proxy, off=on/2; retrospective peak/duration/rise',
            risk='exploratory uncalibrated scores; no success/failure/slip/danger labels',
            repeated='retained; neither invalid nor evidence of hardware acquisition',
            weighting='sample pooled distributions; normalization sensitivity also equal episode weight; occupancy also duration-weighted',
            legacy='inventory/quality only, excluded from native distributions and fitting'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    assert unchanged
    print(json.dumps({k:summary[k] for k in ('native_episodes','legacy_episodes','synthetic_rows_excluded','retained_reads','valid_duration_s','representative_episodes','all_source_files_unchanged')},indent=2))


if __name__=='__main__':
    main()
