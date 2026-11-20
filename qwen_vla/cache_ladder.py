"""Owned inference snapshots for one slow tick. No gate or learned parameters.

Public tensor/cache properties return copies. Private storage is used only by
the model's continuation, which clones the cache before any crop/update.
"""
import copy
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Callable

import torch

from .flow_matching import Handover, flow_time


def handover_for_split(split):
    """Accept nominal split levels and their FP32 representations, only."""
    if isinstance(split, bool) or not isinstance(split, Real) or not math.isfinite(split):
        raise ValueError('selected_split must be one of {0.2, 0.4, 0.6}')
    for level in (.2, .4, .6):
        if math.isclose(float(split), level, rel_tol=0, abs_tol=1e-7):
            return Handover(10-round(level*10))
    raise ValueError('selected_split must be one of {0.2, 0.4, 0.6}')


@dataclass(frozen=True)
class ActionHandoverSnapshot:
    """Owned [latent | action] KV at a handover; not a physical fast tick.

    x_tau matches the existing BF16 single-split API boundary. The FP32 solver
    accumulator is retained separately; snapshot capture never rounds the
    continuing Action Expert trajectory.
    """
    handover: Handover
    n_latent: int
    n_fast: int
    n_state: int
    n_chunk: int
    action_dim: int
    _x_tau: torch.Tensor = field(repr=False, compare=False)
    _integration_state_fp32: torch.Tensor = field(repr=False, compare=False)
    _cache: object = field(repr=False, compare=False)
    _latent_positions: torch.Tensor = field(repr=False, compare=False)
    _action_positions: torch.Tensor = field(repr=False, compare=False)
    _clone_cache: Callable = field(repr=False, compare=False)

    @classmethod
    def capture(cls, *, x_fp32, cache, position_ids, action_position_ids,
                handover, n_fast, n_state, clone_cache):
        if x_fp32.dtype != torch.float32:
            raise ValueError('ladder capture requires FP32 integration state')
        n_latent = position_ids.shape[-1]
        n_chunk, action_dim = x_fp32.shape[1:]
        n_action = n_fast+n_state+1+n_chunk
        owned = clone_cache(cache)
        if (owned.get_seq_length() != n_latent+n_action
                or any(layer.keys.shape[-2] != n_latent+n_action
                       or layer.values.shape[-2] != n_latent+n_action for layer in owned.layers)):
            raise ValueError('incomplete handover cache')
        if action_position_ids.shape[-1] != n_action:
            raise ValueError('invalid handover action positions')
        owned._trex_handover = dict(action_steps=handover.action_steps,
            total_steps=handover.total_steps, tau=handover.tau, precision='fp32',
            length=owned.get_seq_length(), n_action=n_action, coherent=True,
            n_latent=n_latent, n_fast=n_fast, n_state=n_state, n_chunk=n_chunk,
            action_dim=action_dim, integration_dtype='float32', x_tau_dtype='bfloat16',
            dt=-.1, time_grid=tuple((10-i)/10 for i in range(11)),
            fp32_time_grid=tuple(flow_time(i).item() for i in range(11)),
            action_step_indices=tuple(range(handover.action_steps)),
            tactile_step_indices=tuple(range(handover.action_steps,10)))
        return cls(handover, n_latent, n_fast, n_state, n_chunk, action_dim,
                   x_fp32.detach().to(torch.bfloat16).clone(), x_fp32.detach().clone(),
                   owned, position_ids.detach().clone(), action_position_ids.detach().clone(), clone_cache)

    @property
    def tau(self):
        return self.handover.tau

    @property
    def n_action_in_cache(self):
        return self.n_fast+self.n_state+1+self.n_chunk

    @property
    def x_tau(self):
        return self._x_tau.clone()

    @property
    def integration_state_fp32(self):
        return self._integration_state_fp32.clone()

    @property
    def cached_kv(self):
        cache = self._clone_cache(self._cache)
        cache._trex_handover = self.cache_metadata
        return cache

    @property
    def cache_metadata(self):
        return copy.deepcopy(self._cache._trex_handover)

    @property
    def latent_position_ids(self):
        return self._latent_positions.clone()

    @property
    def action_position_ids(self):
        return self._action_positions.clone()

    @property
    def position_ids(self):
        return torch.cat((self._latent_positions, self._action_positions), dim=-1)

    @property
    def cache_position(self):
        return torch.arange(self.n_latent+self.n_action_in_cache, device=self._x_tau.device)

    @property
    def token_counts(self):
        return dict(latent=self.n_latent, fast=self.n_fast, state=self.n_state,
                    time=1, chunk=self.n_chunk, action=self.n_action_in_cache,
                    total=self.n_latent+self.n_action_in_cache)


@dataclass(frozen=True)
class ActionCacheLadder:
    snapshots: tuple[ActionHandoverSnapshot, ...]

    def __post_init__(self):
        object.__setattr__(self, 'snapshots', tuple(self.snapshots))
        if tuple(s.tau for s in self.snapshots) != (.6, .4, .2):
            raise ValueError('ladder requires snapshots ordered at 0.6, 0.4, 0.2')

    @property
    def levels(self):
        return tuple(s.tau for s in self.snapshots)

    def __getitem__(self, selected_split):
        handover = handover_for_split(selected_split)
        return next(s for s in self.snapshots if s.handover == handover)
