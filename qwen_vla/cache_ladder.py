"""Immutable Action-Expert handover snapshots for one slow tick (cache ladder).

One slow tick integrates ONE Action trajectory (8 Euler updates + 1 refresh
forward = 9 forwards) and captures coherent snapshots at tau = 0.6, 0.4, 0.2:

    snapshot(tau) = x_tau, complete [latent | fast | state | risk_S | t | chunk] KV
                    computed at (x_tau, tau), position ids, token counts and the
                    handover schedule.

The saved caches are conditioned on the slow risk r_S (when the model has risk
tokens). A fast tick only reads a snapshot: the tactile continuation clones it
before appending/cropping tactile KV, so selecting a split, changing r_F or
repeating fast ticks never modifies the ladder. Reviewed and adapted from the
earlier experimental `feat/test1` ladder: same capture points and BF16/FP32
state convention, plus the risk-token layout and r_S provenance.
"""
import copy
import hashlib
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable, Optional

import torch

from .flow_matching import Handover, flow_time

LADDER_LEVELS = (.6, .4, .2)


def handover_for_split(split):
    """Nominal split levels (or their FP32 representations) -> Handover."""
    if isinstance(split, bool) or not isinstance(split, Real) or not math.isfinite(split):
        raise ValueError('selected_split must be one of {0.2, 0.4, 0.6}')
    for level in (.2, .4, .6):
        if math.isclose(float(split), level, rel_tol=0, abs_tol=1e-7):
            return Handover(10 - round(level * 10))
    raise ValueError('selected_split must be one of {0.2, 0.4, 0.6}')


@dataclass(frozen=True)
class ActionHandoverSnapshot:
    """Owned KV at one handover tau; not a physical fast tick."""
    handover: Handover
    n_latent: int
    n_fast: int
    n_state: int
    n_risk: int
    n_chunk: int
    action_dim: int
    risk_S: Optional[tuple]
    risk_S_valid: Optional[tuple]
    slow_tick_ns: Optional[int]
    _x_tau: torch.Tensor = field(repr=False, compare=False)
    _integration_state_fp32: torch.Tensor = field(repr=False, compare=False)
    _cache: object = field(repr=False, compare=False)
    _latent_positions: torch.Tensor = field(repr=False, compare=False)
    _action_positions: torch.Tensor = field(repr=False, compare=False)
    _clone_cache: Callable = field(repr=False, compare=False)

    @classmethod
    def capture(cls, *, x_fp32, cache, position_ids, action_position_ids, handover, n_fast, n_state,
                n_risk, clone_cache, risk_S=None, risk_S_valid=None, slow_tick_ns=None):
        if x_fp32.dtype != torch.float32:
            raise ValueError('ladder capture requires the FP32 integration state')
        n_latent = position_ids.shape[-1]
        n_chunk, action_dim = x_fp32.shape[1:]
        n_action = n_fast + n_state + n_risk + 1 + n_chunk
        owned = clone_cache(cache)
        if (owned.get_seq_length() != n_latent + n_action
                or any(layer.keys.shape[-2] != n_latent + n_action
                       or layer.values.shape[-2] != n_latent + n_action for layer in owned.layers)):
            raise ValueError('incomplete handover cache')
        if action_position_ids.shape[-1] != n_action:
            raise ValueError('invalid handover action positions')
        owned._trex_handover = dict(
            action_steps=handover.action_steps, total_steps=handover.total_steps, tau=handover.tau,
            precision='fp32', length=owned.get_seq_length(), n_action=n_action, n_risk=n_risk,
            risk_S=risk_S, risk_S_valid=risk_S_valid, slow_tick_ns=slow_tick_ns, coherent=True,
            n_latent=n_latent, n_fast=n_fast, n_state=n_state, n_chunk=n_chunk, action_dim=action_dim,
            integration_dtype='float32', x_tau_dtype='bfloat16', dt=-.1,
            time_grid=tuple((10 - i) / 10 for i in range(11)),
            fp32_time_grid=tuple(flow_time(i).item() for i in range(11)),
            action_step_indices=tuple(range(handover.action_steps)),
            tactile_step_indices=tuple(range(handover.action_steps, 10)))
        return cls(handover, n_latent, n_fast, n_state, n_risk, n_chunk, action_dim, risk_S, risk_S_valid,
                   slow_tick_ns, x_fp32.detach().to(torch.bfloat16).clone(), x_fp32.detach().clone(),
                   owned, position_ids.detach().clone(), action_position_ids.detach().clone(), clone_cache)

    tau = property(lambda self: self.handover.tau)
    n_action_in_cache = property(lambda self: self.n_fast + self.n_state + self.n_risk + 1 + self.n_chunk)
    x_tau = property(lambda self: self._x_tau.clone())
    integration_state_fp32 = property(lambda self: self._integration_state_fp32.clone())
    latent_position_ids = property(lambda self: self._latent_positions.clone())
    action_position_ids = property(lambda self: self._action_positions.clone())

    @property
    def cached_kv(self):
        cache = self._clone_cache(self._cache)
        cache._trex_handover = self.cache_metadata
        return cache

    @property
    def cache_metadata(self):
        return copy.deepcopy(self._cache._trex_handover)

    @property
    def position_ids(self):
        return torch.cat((self._latent_positions, self._action_positions), dim=-1)

    @property
    def token_counts(self):
        return dict(latent=self.n_latent, fast=self.n_fast, state=self.n_state, risk=self.n_risk, time=1,
                    chunk=self.n_chunk, action=self.n_action_in_cache,
                    total=self.n_latent + self.n_action_in_cache)

    def kv_fingerprint(self):
        """SHA-256 over the exact bytes of every owned K/V tensor and of x_tau."""
        h = hashlib.sha256()
        for t in [self._x_tau, self._integration_state_fp32] + [
                x for layer in self._cache.layers for x in (layer.keys, layer.values)]:
            h.update(str((tuple(t.shape), str(t.dtype))).encode())
            h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class ActionCacheLadder:
    snapshots: tuple

    def __post_init__(self):
        object.__setattr__(self, 'snapshots', tuple(self.snapshots))
        if tuple(s.tau for s in self.snapshots) != LADDER_LEVELS:
            raise ValueError('ladder requires snapshots ordered at 0.6, 0.4, 0.2')
        keys = {(s.n_latent, s.n_fast, s.n_state, s.n_risk, s.n_chunk, s.risk_S, s.risk_S_valid,
                 s.slow_tick_ns) for s in self.snapshots}
        if len(keys) != 1:
            raise ValueError('ladder snapshots come from different contexts')

    levels = property(lambda self: tuple(s.tau for s in self.snapshots))
    risk_S = property(lambda self: self.snapshots[0].risk_S)
    risk_S_valid = property(lambda self: self.snapshots[0].risk_S_valid)
    slow_tick_ns = property(lambda self: self.snapshots[0].slow_tick_ns)

    def __getitem__(self, selected_split):
        handover = handover_for_split(selected_split)
        return next(s for s in self.snapshots if s.handover == handover)
