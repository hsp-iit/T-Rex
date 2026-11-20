"""Provisional non-learned inference selector. No physical danger interpretation."""
import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    mode: str = 'fixed'
    fixed_split: float = .4
    thresholds: tuple = (.3,.7)
    levels: tuple = (.2,.4,.6)
    rate_windows: tuple = (100,)
    ema_alpha: float = 1.
    hysteresis: float = 0.
    min_dwell_ns: int = 0
    fallback_split: float = .4

    def __post_init__(self):
        if self.mode not in ('fixed','magnitude','rate','combined'):
            raise ValueError('unsupported gate mode')
        if self.fixed_split not in (.2,.4,.6) or self.fallback_split not in (.2,.4,.6) or len(self.levels)!=3 or set(self.levels)!={.2,.4,.6}:
            raise ValueError('invalid split levels')
        if (len(self.thresholds)!=2 or not 0<=self.thresholds[0]<self.thresholds[1]<=1
                or not 0<self.ema_alpha<=1 or not np.isfinite(self.hysteresis) or self.hysteresis<0
                or self.min_dwell_ns<0 or not self.rate_windows or any(w not in (50,100,200) for w in self.rate_windows)):
            raise ValueError('invalid gate configuration')
        object.__setattr__(self,'levels',tuple(self.levels))
        object.__setattr__(self,'thresholds',tuple(self.thresholds))
        object.__setattr__(self,'rate_windows',tuple(self.rate_windows))


class SplitSelector:
    def __init__(self,config=GateConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.ema = self.bin = self.last_change = self.last_tick = None
        self.last_input = self.last_result = None
        self.last_segment = None

    def select_split(self,features,quality,tick_ns):
        c = self.config
        fingerprint = (tuple(sorted(features.items())),quality['fresh'],tuple(sorted(quality['feature_valid'].items())),
                       quality.get('segment'),quality.get('latest_read_ns'),quality.get('latest_available_ns'),
                       quality.get('model_observation_valid',True))
        if self.last_tick is not None:
            if tick_ns<self.last_tick:
                raise ValueError('selector ticks must be nondecreasing')
            if tick_ns==self.last_tick:
                if fingerprint!=self.last_input:
                    raise ValueError('changed observations at an already evaluated tick')
                return copy.deepcopy(self.last_result)
        segment=quality.get('segment')
        if self.last_segment is not None and segment!=self.last_segment:
            self.ema=self.bin=self.last_change=None
        self.last_segment=segment
        previous = c.levels[self.bin] if self.bin is not None else None
        required = ([] if c.mode=='fixed' else ['M'] if c.mode=='magnitude' else
                    ['R'+str(w) for w in c.rate_windows]+(['M'] if c.mode=='combined' else []))
        valid = quality['fresh'] and quality.get('model_observation_valid',True) and all(quality['feature_valid'].get(k,False) and np.isfinite(features[k]) for k in required)
        fallback = c.mode!='fixed' and not valid
        raw = None
        if c.mode=='fixed':
            selected,reason = c.fixed_split,'fixed'
            self.bin = c.levels.index(selected)
        elif fallback:
            selected,reason = c.fallback_split,'invalid_features'
            self.bin,self.ema = c.levels.index(selected),None
        else:
            rate = max(features['R'+str(w)] for w in c.rate_windows) if c.mode in ('rate','combined') else 0.
            raw = features['M'] if c.mode=='magnitude' else rate if c.mode=='rate' else 1-(1-features['M'])*(1-rate)
            self.ema = raw if self.ema is None else c.ema_alpha*raw+(1-c.ema_alpha)*self.ema
            target = int(np.searchsorted(c.thresholds,self.ema,side='right'))
            if self.bin is not None and c.hysteresis:
                target = self.bin
                while target<2 and self.ema>=c.thresholds[target]+c.hysteresis:
                    target += 1
                while target>0 and self.ema<c.thresholds[target-1]-c.hysteresis:
                    target -= 1
            reason = 'score'
            if self.bin is not None and target!=self.bin and self.last_change is not None and tick_ns-self.last_change<c.min_dwell_ns:
                target,reason = self.bin,'dwell'
            self.bin = target
            selected = c.levels[target]
        transition = previous is not None and selected!=previous
        if previous is None or transition:
            self.last_change = tick_ns
        result = dict(selected_split=selected,raw_score=raw,score=self.ema,fallback=fallback,
                      reason=reason,transition=transition,state=dict(bin=self.bin,ema=self.ema,last_change_ns=self.last_change),
                      provisional=True,features=dict(features))
        self.last_tick,self.last_input,self.last_result = tick_ns,fingerprint,copy.deepcopy(result)
        return result


def select_split(contact_features,quality,gate_config,*,tick_ns,state=None):
    """Scalar convenience API; pass a SplitSelector to retain EMA/dwell state.

    Full debugging output is available as state.last_result. A call without
    state intentionally creates a fresh selector and has no temporal smoothing.
    """
    selector=state if state is not None else SplitSelector(gate_config)
    if selector.config!=gate_config:
        raise ValueError('selector state/configuration mismatch')
    return selector.select_split(contact_features,quality,tick_ns)['selected_split']
