"""Non-learned inference-time split gate driven by the fresh fast-tick risk r_F.

Monotone mapping (before hysteresis/dwell), thresholds lo < hi:

    r_F <  lo        -> tau_split 0.2   (8 Action + 2 Tactile Euler steps)
    lo <= r_F < hi   -> tau_split 0.4   (6 + 4, the baseline handover)
    r_F >= hi        -> tau_split 0.6   (4 + 6)

Higher measured contact risk hands more of the flow to the Tactile Expert.
Thresholds are initialized from TRAINING-data quantiles of r (RiskStats,
default q50/q85, floored at 1e-3 so exactly-zero risk maps to 0.2). This
mapping is an experimental starting point: it is NOT claimed optimal and it is
NOT a safety calibration.

  * mode 'fixed'    : always `fixed_split` (fixed-split ablations, default 0.4).
  * hysteresis h    : leave bin b upward only if r >= thr[b] + h, downward only
                      if r < thr[b-1] - h.
  * min dwell       : a risk-driven change is held back until `min_dwell_ms` has
                      elapsed since the previous change.
  * missing data    : invalid/non-finite r_F -> `fallback_split` (0.4) at once
                      (overrides dwell); the first valid tick after a fallback is
                      not dwell-limited.
Every decision is returned (and kept in `log`) with r_F, M/R/V, validity,
selected split, previous split, transition flag and reason.
"""
import csv
import json
import math
from dataclasses import dataclass

LEVELS = (.2, .4, .6)
MIN_THRESHOLD = 1e-3


@dataclass(frozen=True)
class GateConfig:
    mode: str = 'fixed'
    fixed_split: float = .4
    thresholds: tuple = (.015, .2)
    hysteresis: float = 0.
    min_dwell_ms: float = 0.
    fallback_split: float = .4

    def __post_init__(self):
        if self.mode not in ('fixed', 'adaptive'):
            raise ValueError("gate mode must be 'fixed' or 'adaptive'")
        if self.fixed_split not in LEVELS or self.fallback_split not in LEVELS:
            raise ValueError('splits must be one of {0.2, 0.4, 0.6}')
        t = tuple(float(x) for x in self.thresholds)
        if len(t) != 2 or not (0 <= t[0] < t[1] <= 1):
            raise ValueError('thresholds must satisfy 0 <= lo < hi <= 1')
        if not (math.isfinite(self.hysteresis) and 0 <= self.hysteresis < (t[1] - t[0]) / 2 + 1e-12):
            raise ValueError('hysteresis must be in [0, (hi - lo) / 2)')
        if self.mode == 'adaptive' and self.hysteresis > 0 and not t[0] - self.hysteresis > 0:
            # r >= 0, so with lo - h <= 0 the gate could never return to 0.2.
            raise ValueError(f'hysteresis {self.hysteresis} >= lower threshold {t[0]}: tau 0.2 would be unreachable')
        if not (math.isfinite(self.min_dwell_ms) and self.min_dwell_ms >= 0):
            raise ValueError('min_dwell_ms must be >= 0')
        object.__setattr__(self, 'thresholds', t)

    @classmethod
    def from_risk_stats(cls, stats, **overrides):
        """Thresholds from the train-only quantiles stored in RiskStats."""
        if stats.gate_thresholds is None:
            raise ValueError('risk statistics carry no gate thresholds')
        lo, hi = stats.gate_thresholds
        return cls(**{'mode': 'adaptive', 'thresholds': (max(lo, MIN_THRESHOLD), max(hi, 2 * MIN_THRESHOLD)),
                      **overrides})

    def to_dict(self):
        return {'mode': self.mode, 'fixed_split': self.fixed_split, 'thresholds': list(self.thresholds),
                'hysteresis': self.hysteresis, 'min_dwell_ms': self.min_dwell_ms,
                'fallback_split': self.fallback_split, 'levels': list(LEVELS),
                'note': 'initial monotone mapping; not optimal, not a safety calibration'}


def raw_bin(r, thresholds):
    """Monotone bin index (0, 1, 2) of r without hysteresis/dwell."""
    lo, hi = thresholds
    return 0 if r < lo else 1 if r < hi else 2


class RiskGate:
    def __init__(self, config: GateConfig = GateConfig(), keep_log=True):
        self.config, self.keep_log = config, keep_log
        self.reset()

    def reset(self):
        self.bin = self.last_change_ns = self.last_tick_ns = None
        self.last_reason = None
        self.log = []

    def select(self, tick_ns, risk, risk_valid, features=None):
        c = self.config
        tick_ns = int(tick_ns)
        if self.last_tick_ns is not None and tick_ns < self.last_tick_ns:
            raise ValueError('gate ticks must be nondecreasing')
        prev = self.bin
        valid = bool(risk_valid) and risk is not None and math.isfinite(float(risk))
        r = float(risk) if valid else None
        if c.mode == 'fixed':
            target, reason = LEVELS.index(c.fixed_split), 'fixed'
        elif not valid:
            target, reason = LEVELS.index(c.fallback_split), 'missing_risk_fallback'
        else:
            lo, hi = c.thresholds
            target, h = raw_bin(r, c.thresholds), c.hysteresis
            if prev is not None and h > 0 and self.last_reason != 'missing_risk_fallback':
                target = prev
                while target < 2 and r >= (lo, hi)[target] + h:
                    target += 1
                while target > 0 and r < (lo, hi)[target - 1] - h:
                    target -= 1
            reason = 'risk'
            if (prev is not None and target != prev and self.last_change_ns is not None
                    and self.last_reason != 'missing_risk_fallback'
                    and tick_ns - self.last_change_ns < c.min_dwell_ms * 1e6):
                target, reason = prev, 'dwell_hold'
        transition = prev is not None and target != prev
        if prev is None or transition:
            self.last_change_ns = tick_ns
        self.bin, self.last_tick_ns, self.last_reason = target, tick_ns, reason
        rec = {'tick_ns': tick_ns, 'selected_split': LEVELS[target],
               'previous_split': LEVELS[prev] if prev is not None else None,
               'transition': transition, 'reason': reason, 'risk': r, 'risk_valid': valid,
               'action_steps': 10 - round(LEVELS[target] * 10), 'tactile_steps': round(LEVELS[target] * 10)}
        if features:
            rec.update({k: features[k] for k in ('M', 'R', 'V', 'R_pos', 'R_neg', 'M_valid', 'R_valid',
                                                 'V_valid', 'force_age_ns', 'joint_age_ns') if k in features})
        if self.keep_log:
            self.log.append(rec)
        return rec

    def summary(self):
        n = len(self.log)
        occ = {str(l): sum(r['selected_split'] == l for r in self.log) for l in LEVELS}
        return {'ticks': n, 'transitions': sum(r['transition'] for r in self.log),
                'occupancy': occ, 'fallbacks': sum(r['reason'] == 'missing_risk_fallback' for r in self.log),
                'dwell_holds': sum(r['reason'] == 'dwell_hold' for r in self.log), 'config': self.config.to_dict()}

    def write_log(self, path):
        path = str(path)
        if path.endswith('.jsonl'):
            with open(path, 'w') as f:
                for r in self.log:
                    f.write(json.dumps(r) + '\n')
        else:
            keys = sorted({k for r in self.log for k in r})
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.log)
