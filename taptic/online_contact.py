"""Causal SDK-count contact descriptors. No hardware force or danger calibration.

The online window sums are the offline analysis's positive variation / W.
Samples are processed in availability order; late older reads are explicitly
discarded. Partial windows have zero placeholders and separate validity masks.
"""
from collections import deque
from dataclasses import dataclass, field
import heapq
import json
from numbers import Integral
from pathlib import Path

import numpy as np

WINDOWS = (50, 100, 200)


@dataclass(frozen=True)
class NativeSample:
    episode: int
    read_ns: int
    available_ns: int
    force: np.ndarray = field(compare=False)
    sequence: int = 0
    reset_sequence: int = 0
    read_errors: int = 0
    dropped: int = 0
    valid: bool = True
    native: bool = True
    boundary: bool = False
    taxels: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,Integral) for v in (self.read_ns,self.available_ns)) or self.available_ns < self.read_ns or self.read_ns < 0:
            raise ValueError('availability must be >= nonnegative read time')
        f = np.array(self.force, dtype=np.float64, copy=True)
        if f.shape != (5, 3):
            raise ValueError('SDK force must have shape [5,3]')
        f.flags.writeable = False
        object.__setattr__(self, 'force', f)
        if self.taxels is not None:
            x = np.array(self.taxels, dtype=np.float32, copy=True).reshape(5,120,3)
            x.flags.writeable = False
            object.__setattr__(self, 'taxels', x)


@dataclass(frozen=True)
class ContactNormalization:
    magnitude: tuple
    variation: tuple  # 50/100/200 ms, each five per-finger scales
    train_episodes: tuple = ()
    source_identity: str = ''

    def __post_init__(self):
        m = np.asarray(self.magnitude, dtype=float)
        v = np.asarray(self.variation, dtype=float)
        if m.shape != (5,) or v.shape != (3,5) or not np.isfinite(m).all() or not np.isfinite(v).all() or (m<=0).any() or (v<=0).any():
            raise ValueError('normalization requires positive finite [5] and [3,5] scales')
        object.__setattr__(self, 'magnitude', tuple(m.tolist()))
        object.__setattr__(self, 'variation', tuple(tuple(x) for x in v.tolist()))
        object.__setattr__(self, 'train_episodes', tuple(self.train_episodes))

    @classmethod
    def from_analysis(cls, path, episodes, source_identity):
        """Load frozen offline scales; verify provenance against actual metadata."""
        a = json.loads(Path(path).read_text())
        ids = a['train_episodes']
        if not ids or len(ids)!=len(set(ids)) or any(i not in episodes.index or episodes.loc[i,'suggested_split']!='train' or bool(episodes.loc[i,'tactile_interpolated']) for i in ids):
            raise ValueError('contact scales must be fitted on native training episodes only')
        # The analysis recorded source identity separately; require that evidence.
        evidence = json.loads((Path(path).parent/'summary.json').read_text())
        if evidence['source_identity'] != source_identity:
            raise ValueError('contact normalization source mismatch')
        return cls(tuple(a['magnitude']['per_finger_q99']),
                   tuple(tuple(a['windows'][str(w)]['positive_variation_rate_positive_q99']) for w in WINDOWS),
                   tuple(ids), source_identity)


class OnlineContactExtractor:
    def __init__(self, normalization, *, gap_ns=50_000_000, stale_ns=50_000_000, keep_samples=64):
        if gap_ns<=0 or stale_ns<=0 or keep_samples<2:
            raise ValueError('invalid quality limits')
        self.normalization = normalization
        self.gap_ns, self.stale_ns, self.keep_samples = gap_ns, stale_ns, keep_samples
        self.start_episode(None)

    def start_episode(self, episode):
        self.episode = episode
        self.pending, self.history, self.seen = [], deque(), {}
        self.latest = None
        self.last_tick = None
        self.segment = 0
        self.reason = 'episode_start'
        self.late_discarded = self.duplicates = self.invalid_samples = 0
        self._order = 0

    def ingest(self, samples):
        for sample in samples:
            if sample.episode != self.episode:
                raise ValueError('sample belongs to another episode; start_episode explicitly')
            key = (sample.read_ns, sample.sequence)
            if key in self.seen:
                old = self.seen[key]
                if (old.available_ns!=sample.available_ns or old.reset_sequence!=sample.reset_sequence
                        or old.valid!=sample.valid or old.native!=sample.native
                        or old.read_errors!=sample.read_errors or old.dropped!=sample.dropped
                        or not np.array_equal(old.force,sample.force,equal_nan=True)):
                    raise ValueError('conflicting duplicate SDK sample')
                self.duplicates += 1
                continue
            self.seen[key] = sample
            heapq.heappush(self.pending,(sample.available_ns,sample.read_ns,self._order,sample))
            self._order += 1

    def at(self, tick_ns):
        if isinstance(tick_ns,bool) or not isinstance(tick_ns,Integral) or tick_ns<0:
            raise ValueError('physical timestamps must be nonnegative integer nanoseconds')
        if self.episode is None:
            raise ValueError('start_episode before evaluating')
        if self.last_tick is not None and tick_ns < self.last_tick:
            raise ValueError('physical ticks must be nondecreasing')
        self.last_tick = tick_ns
        while self.pending and self.pending[0][0] <= tick_ns:
            sample = heapq.heappop(self.pending)[-1]
            prev = self.latest
            if prev is not None and sample.read_ns <= prev.read_ns:
                self.late_discarded += 1
                continue
            valid = sample.valid and sample.native and np.isfinite(sample.force).all()
            reason = None
            if not valid:
                reason = 'invalid_sample'
                self.invalid_samples += 1
            elif not self.history:
                reason = 'history_start'
            elif sample.boundary:
                reason = 'explicit_boundary'
            elif prev is not None:
                if sample.reset_sequence != prev.reset_sequence:
                    reason = 'reset'
                elif sample.read_errors!=prev.read_errors or sample.dropped!=prev.dropped:
                    reason = 'error_or_drop'
                elif sample.sequence != prev.sequence+1:
                    reason = 'sequence'
                elif sample.read_ns-prev.read_ns > self.gap_ns:
                    reason = 'gap'
            if reason:
                self.history.clear()
                self.segment += 1
                self.reason = reason
            self.latest = sample
            if valid:
                mag = np.linalg.norm(sample.force,axis=1)
                delta = np.maximum(mag-self.history[-1][1],0) if self.history else np.zeros(5)
                self.history.append((sample,mag,delta))
        # Keep a predecessor of the longest window AND sufficient native rows
        # for existing 16-sample VQ histories. No synthetic rows enter features.
        left = tick_ns-200_000_000
        while len(self.history)>self.keep_samples and self.history[1][0].read_ns<=left:
            self.history.popleft()
        # Old duplicates will already be rejected as late reads; do not retain
        # entire old taxel packets for the lifetime of a long episode.
        if self.latest is not None:
            cutoff=self.history[0][0].read_ns if self.history else self.latest.read_ns
            self.seen={key:value for key,value in self.seen.items() if key[0]>=cutoff}
        latest = self.latest
        age = tick_ns-latest.read_ns if latest else None
        fresh = bool(self.history and latest is self.history[-1][0] and age<=self.stale_ns)
        mag = self.history[-1][1].copy() if fresh else np.zeros(5)
        values = {'M':float(np.clip(mag/np.asarray(self.normalization.magnitude),0,1).max())}
        masks, raw, coverage = {'M':fresh}, {}, {}
        for j,w in enumerate(WINDOWS):
            start = tick_ns-w*1_000_000
            full = bool(self.history and self.history[0][0].read_ns<=start)
            coverage[str(w)] = min(1.,max(0.,(tick_ns-self.history[0][0].read_ns)/(w*1e6))) if self.history else 0.
            rate = sum((row[2] for row in self.history if row[0].read_ns>start),np.zeros(5))/(w/1000)
            ok = fresh and full
            raw[str(w)] = rate.tolist() if ok else [0.]*5
            values['R'+str(w)] = float(np.clip(rate/np.asarray(self.normalization.variation[j]),0,1).max()) if ok else 0.
            masks['R'+str(w)] = ok
        quality = dict(fresh=fresh,age_ns=age,available_age_ns=tick_ns-latest.available_ns if latest else None,
                       latest_read_ns=latest.read_ns if latest else None,latest_available_ns=latest.available_ns if latest else None,
                       feature_valid=masks,coverage=coverage,segment=self.segment,boundary_reason=self.reason,
                       late_discarded=self.late_discarded,duplicates=self.duplicates,invalid_samples=self.invalid_samples)
        return dict(episode=self.episode,tick_ns=int(tick_ns),features=values,quality=quality,
                    magnitude=mag.tolist(),positive_variation_rate=raw)

    def force_history(self, length):
        """Existing model's sample-count history; padding is not contact evidence."""
        if not self.history or length<1:
            raise ValueError('missing force history')
        rows = list(self.history)[-length:]
        force = np.stack([row[0].force for row in rows]).astype(np.float32)
        return np.concatenate([np.repeat(force[:1],max(0,length-len(force)),axis=0),force]),len(force)
