"""Approach A: an inference-only selector around immutable T-Rex cache ladders.

No code here is imported by the training loop. The returned 21D chunk remains
in the measured slow-anchor delta-base frame, never a new fast-tick target.
"""
from dataclasses import dataclass
import time
from types import MappingProxyType

import numpy as np
import torch

from .data import denormalize
from .split_gate import SplitSelector


@dataclass(frozen=True)
class SlowAnchor:
    episode: int
    slow_ns: int
    observation_ns: int
    context_available_ns: int
    ready_ns: int
    expires_ns: int
    ref_position: tuple
    ref_rotvec_gd: tuple

    def __post_init__(self):
        if not (self.observation_ns<=self.context_available_ns<=self.slow_ns<=self.ready_ns<self.expires_ns):
            raise ValueError('invalid slow context/cache availability or expiry')
        for key in ('ref_position','ref_rotvec_gd'):
            a=np.asarray(getattr(self,key),float)
            if a.shape!=(3,) or not np.isfinite(a).all():
                raise ValueError('invalid slow reference pose')
            object.__setattr__(self,key,tuple(a.tolist()))


class TaxelObservation:
    def __call__(self,extractor,batch,device):
        sample=extractor.latest
        if sample is None or sample.taxels is None or not np.isfinite(sample.taxels).all():
            return None
        x=sample.taxels/np.asarray([128.,128.,255.],np.float32)
        return {'tactile_f6':torch.tensor(x.reshape(5,360),device=device).unsqueeze(0).repeat(batch,1,1)}


class SDKForceObservation:
    """Reuse a frozen model force normalizer and existing raw VQ history contract."""
    def __init__(self,force_stats,history_length=16,use_history=False):
        self.stats,self.length,self.use_history=force_stats,history_length,use_history

    def __call__(self,extractor,batch,device):
        history,_=extractor.force_history(self.length)
        obs={'tactile_f6':torch.tensor(self.stats.normalize(history[-1]),device=device).unsqueeze(0).repeat(batch,1,1)}
        if self.use_history:
            obs['tactile_f6_history']=torch.tensor(history,device=device).unsqueeze(0).repeat(batch,1,1,1)
        return obs


class ApproachAInference:
    def __init__(self,model,extractor,gate_config,action_stats,observation_builder=None):
        self.model,self.extractor=model,extractor
        self.selector=SplitSelector(gate_config)
        # Freeze normalization so caller mutation cannot change an active run.
        self.action_stats={k:np.array(action_stats[k],dtype=np.float32,copy=True) for k in ('min','max')}
        if any(v.shape!=(21,) or not np.isfinite(v).all() for v in self.action_stats.values()) or np.any(self.action_stats['max']<=self.action_stats['min']):
            raise ValueError('invalid frozen action statistics')
        for v in self.action_stats.values():v.flags.writeable=False
        self.action_stats=MappingProxyType(self.action_stats)
        self.observation_builder=observation_builder or TaxelObservation()
        self.anchor=self.ladder=None
        self.generation=self.last_fast=None
        self.cache_id=0
        self.invalidated=False

    def _model_signature(self):
        if not hasattr(self.model,'parameters'):
            return (id(self.model),)
        return (id(self.model),tuple((p.data_ptr(),p._version,str(p.device),str(p.dtype)) for p in self.model.parameters()))

    def start_episode(self,episode):
        self.extractor.start_episode(episode)
        self.selector.reset()
        self.anchor=self.ladder=None
        self.last_fast=None
        self.invalidated=False

    @torch.no_grad()
    def slow_tick(self,anchor,**context):
        if anchor.episode!=self.extractor.episode:
            raise ValueError('start_episode before building a different episode cache')
        if self.last_fast is not None and anchor.slow_ns<self.last_fast:
            raise ValueError('cannot build a cache in the physical past')
        self.extractor.at(anchor.slow_ns)
        started=time.perf_counter()
        ladder=self.model.forward_flow_action_ladder(**context)
        self.ladder,self.anchor=ladder,anchor
        self.model_signature=self._model_signature()
        self.generation=self.extractor.segment
        self.invalidated=False
        self.cache_id+=1
        self.build_ms=(time.perf_counter()-started)*1000
        return {'cache_id':self.cache_id,'anchor_ns':anchor.slow_ns,'build_ms':self.build_ms,
                'action_euler_updates':8,'action_forwards':9}

    @torch.no_grad()
    def fast_tick(self,episode,tick_ns,samples=()):
        if episode!=self.extractor.episode:
            raise ValueError('fast tick belongs to another episode')
        if self.last_fast is not None and tick_ns<self.last_fast:
            raise ValueError('out-of-order physical fast tick')
        if self.anchor is not None and tick_ns<self.anchor.slow_ns:
            raise ValueError('fast tick precedes slow anchor')
        self.extractor.ingest(samples)
        started=time.perf_counter()
        contact=self.extractor.at(tick_ns)
        obs=None
        if self.ladder is not None and contact['quality']['fresh']:
            context=self.ladder[.4]  # dimensions/device only; not split selection
            obs=self.observation_builder(self.extractor,context._x_tau.shape[0],context._x_tau.device)
            if obs is not None and not all(torch.isfinite(v).all() for v in obs.values()):
                obs=None
            contact['quality']['model_observation_valid']=obs is not None
        decision=self.selector.select_split(contact['features'],contact['quality'],tick_ns)
        self.last_fast=tick_ns
        result=dict(tick_ns=int(tick_ns),slow_anchor_ns=self.anchor.slow_ns if self.anchor else None,
            cache_id=self.cache_id,contact=contact,gate=decision,actions=None,norm_actions=None,
            action_steps=10-round(decision['selected_split']*10),tactile_steps=round(decision['selected_split']*10),
            slow_action_euler_updates=8 if self.ladder is not None else 0,
            feature_gate_ms=(time.perf_counter()-started)*1000,tactile_ms=None,
            action_error_supported=False,frame='slow_anchor_delta_base',status='ok')
        result['cache_identity']=f'{episode}:{self.anchor.slow_ns}:{self.cache_id}' if self.anchor else None
        if self.ladder is None:
            result['status']='no_cache'
        elif tick_ns<self.anchor.ready_ns:
            result['status']='cache_not_ready'
        elif tick_ns>=self.anchor.expires_ns:
            result['status']='cache_expired'
        elif self._model_signature()!=self.model_signature:
            self.invalidated=True
            result['status']='model_changed_requires_scheduled_slow_tick'
        elif self.invalidated or self.extractor.segment!=self.generation:
            self.invalidated=True
            result['status']='sensor_boundary_requires_scheduled_slow_tick'
        elif not contact['quality']['fresh']:
            result['status']='missing_or_stale_tactile'
        else:
            if obs is None:
                result['status']='missing_model_tactile'
            else:
                started=time.perf_counter()
                norm=self.model.tactile_flow_continue_from_ladder(self.ladder,decision['selected_split'],**obs)
                actions=denormalize(norm.float().cpu().numpy(),self.action_stats)
                if not np.isfinite(actions).all():
                    raise ValueError('nonfinite inference actions')
                result.update(actions=actions,norm_actions=norm,tactile_ms=(time.perf_counter()-started)*1000)
        if self.anchor:
            result['reference']=dict(observation_ns=self.anchor.observation_ns,
                position=self.anchor.ref_position,rotvec_gd=self.anchor.ref_rotvec_gd,
                original_chunk_times='slow-anchor 10Hz slots; not shifted to fast time')
        return result

    @staticmethod
    def action_error(*args,**kwargs):
        raise NotImplementedError('fast-tick action targets are not established; inference diagnostics only')
