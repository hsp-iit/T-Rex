"""Numerical flow schedules and batch-wide handovers, independent of robot time."""
from dataclasses import dataclass
import torch
import torch.distributed as dist

"""
La classe Handover collega τ al numero di passi dei due esperti.
La funzione [`select_handover`] supporta:
- fixed: mantiene uno split fisso, con baseline a 0.4;
- multi: estrae uniformemente uno split tra quelli configurati.
Nel training distribuito, il rank 0 estrae lo split e lo comunica agli altri rank: tutto il batch distribuito usa la stessa scelta.
"""
@dataclass(frozen=True)
class Handover:
    action_steps: int
    total_steps: int = 10

    def __post_init__(self):
        if self.total_steps != 10 or self.action_steps not in (4, 6, 8):
            raise ValueError('handover requires 10 total steps and 4, 6 or 8 action steps')

    @property
    def tactile_steps(self):
        return self.total_steps - self.action_steps

    @property
    def tau(self):
        return self.tactile_steps / self.total_steps


def flow_time(step, total_steps=10, *, device=None):
    """Integer-indexed FP32 grid. No recurrence and no physical tick units."""
    if not 0 <= step <= total_steps or total_steps <= 0:
        raise ValueError('invalid Euler step index')
    return torch.tensor((total_steps - step) / total_steps, dtype=torch.float32, device=device)


def validate_split_config(cfg):
    total = getattr(cfg, 'cascaded_total_steps', 10)
    Handover(getattr(cfg, 'cascaded_split_step', 6), total)
    mode = getattr(cfg, 'cascaded_split_mode', 'fixed')
    if mode not in ('fixed', 'multi'):
        raise ValueError('cascaded_split_mode must be fixed or multi')
    splits = tuple(getattr(cfg, 'cascaded_splits', (.2, .4, .6)))
    if not splits or len(set(splits)) != len(splits) or any(t not in (.2, .4, .6) for t in splits):
        raise ValueError('cascaded_splits must be distinct members of {0.2,0.4,0.6}')
    if getattr(cfg, 'flow_time_precision', 'fp32') not in ('fp32', 'legacy_bf16'):
        raise ValueError('invalid flow_time_precision')


def select_handover(cfg, device='cpu', generator=None):
    """One split per batch, synchronized when torch.distributed is initialized."""
    validate_split_config(cfg)
    if getattr(cfg, 'cascaded_split_mode', 'fixed') == 'fixed':
        return Handover(getattr(cfg, 'cascaded_split_step', 6))
    levels = tuple(getattr(cfg, 'cascaded_splits', (.2, .4, .6)))
    rank = dist.get_rank() if dist.is_initialized() else 0
    choice = int(torch.randint(len(levels), (), generator=generator)) if rank == 0 else 0
    selected = torch.tensor(choice, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.broadcast(selected, src=0)
    tau = levels[int(selected.item())]
    return Handover(10 - round(tau * 10))


def tactile_training_pair(actions, noise, time_r, handover, precision='fp32'):
    """Original Beta draw scaled to (0,tau_split]; unchanged velocity target."""
    if precision == 'legacy_bf16':
        tau = time_r.to(actions.dtype) * handover.tau
        x = (1-tau[:, None, None])*actions + tau[:, None, None]*noise
    else:
        tau = time_r.float() * handover.tau
        x = (1-tau[:, None, None])*actions.float() + tau[:, None, None]*noise.float()
    return x, tau, noise - actions
