"""Slow/fast-tick inference engine: risk-conditioned cache ladder + adaptive gate.

Slow tick S (10 Hz, after the visual/language context is embedded):
    r_S   <- OnlineRiskExtractor at S (measurements available by S only)
    ladder <- model.forward_flow_action_ladder(context, r_S)      (immutable)
Fast tick F >= S (any rate, e.g. every native tactile read):
    1. latest valid tactile observation available by F
    2. updated VQ-VAE force history and r_F (same extractor, same code as training)
    3. split <- RiskGate(r_F)                                    (non-learned)
    4. snapshot <- ladder[split]                                  (read only)
    5. Tactile Expert continuation for the remaining 10*split Euler steps with
       the current force tokens, VQ codes and r_F
    6. return the action chunk (normalized; denormalized if action stats given)
A new r_F never changes r_S nor rebuilds the ladder. Works for every variant:
a model without risk tokens simply receives no risk inputs (the gate can still
use r_F: "external risk-based selection"); mode='fixed' reproduces fixed-split
inference. No robot I/O happens here.
"""
import time

import numpy as np
import torch

from .data import denormalize
from .risk import OnlineRiskExtractor, RiskConfig, RiskStats
from .risk_gate import GateConfig, RiskGate

SLOT_PERIOD_NS = 100_000_000


class RiskAdaptivePolicy:
    def __init__(self, model, risk_stats: RiskStats, gate_config: GateConfig, force_stats,
                 risk_cfg: RiskConfig = RiskConfig(), vq_window=16, gap_ms=50., action_stats=None):
        self.model = model
        self.risk_cfg = risk_cfg
        self.extractor = OnlineRiskExtractor(risk_stats, risk_cfg, gap_ms, vq_window)
        self.gate = RiskGate(gate_config)
        self.force_stats = force_stats
        self.action_stats = action_stats
        self.ladder = self.slow = None
        self.n_slow = 0

    def start_episode(self):
        self.extractor.start_episode()
        self.gate.reset()
        self.ladder = self.slow = None

    def ingest(self, samples):
        self.extractor.ingest(samples)

    def _risk_inputs(self, rec, B, device):
        if not getattr(self.model, 'use_risk_tokens', False):
            return None, None
        return (torch.full((B,), float(rec['risk']), device=device),
                torch.full((B,), bool(rec['risk_valid']), dtype=torch.bool, device=device))

    @torch.no_grad()
    def slow_tick(self, slow_ns, *, inputs_embeds, position_ids, noise, attention_mask=None,
                  fast_embeds=None, state_embeds=None):
        rec = self.extractor.at(slow_ns)
        rS, vS = self._risk_inputs(rec, noise.shape[0], noise.device)
        t0 = time.perf_counter()
        self.ladder = self.model.forward_flow_action_ladder(
            inputs_embeds, position_ids, noise, attention_mask=attention_mask, state_embeds=state_embeds,
            fast_embeds=fast_embeds, risk=rS, risk_valid=vS, slow_tick_ns=slow_ns)
        self.n_slow += 1
        self.slow = {'slow_ns': int(slow_ns), 'slow_id': self.n_slow,
                     'risk_S': {k: v for k, v in rec.items() if k != 'force_history'},
                     'build_ms': (time.perf_counter() - t0) * 1e3, 'action_forwards': 9}
        return dict(self.slow)

    @torch.no_grad()
    def fast_tick(self, fast_ns):
        if self.ladder is None:
            raise RuntimeError('fast tick before the first slow tick')
        if fast_ns < self.slow['slow_ns']:
            raise ValueError('fast tick precedes its slow anchor')
        t0 = time.perf_counter()
        rec = self.extractor.at(fast_ns)                                     # 1-2
        decision = self.gate.select(fast_ns, rec['risk'], rec['risk_valid'], rec)   # 3
        split = decision['selected_split']
        snap = self.ladder[split]                                            # 4
        B, device = snap._x_tau.shape[0], snap._x_tau.device
        if rec['force_valid']:
            hist = rec['force_history']
            cur = self.force_stats.normalize(hist[-1]).astype(np.float32)
            status = 'ok'
        else:   # same zero inputs as tactile dropout in training; r_F marked missing
            hist = np.zeros_like(rec['force_history'])
            cur = np.zeros((5, 3), np.float32)
            status = 'tactile_missing_zero_inputs'
        kw = {'tactile_f6': torch.tensor(cur, device=device).unsqueeze(0).repeat(B, 1, 1)}
        if getattr(self.model, 'use_tactile_vqvae', False):
            kw['tactile_f6_history'] = torch.tensor(hist, device=device).unsqueeze(0).repeat(B, 1, 1, 1)
        rF, vF = self._risk_inputs(rec, B, device)
        chunk = self.model.tactile_flow_continue_from_ladder(self.ladder, split, risk=rF, risk_valid=vF, **kw)  # 5
        slots = self.slow['slow_ns'] + SLOT_PERIOD_NS * np.arange(snap.n_chunk)
        out = {'fast_ns': int(fast_ns), 'slow_ns': self.slow['slow_ns'], 'slow_id': self.slow['slow_id'],
               'status': status, 'gate': decision, 'norm_actions': chunk,
               'risk_S': self.slow['risk_S']['risk'], 'risk_S_valid': self.slow['risk_S']['risk_valid'],
               'risk_F': rec['risk'], 'risk_F_valid': rec['risk_valid'],
               'features_F': {k: rec[k] for k in ('M', 'R', 'V', 'R_pos', 'R_neg', 'M_valid', 'R_valid',
                                                  'V_valid', 'force_age_ns', 'joint_age_ns')},
               'first_unelapsed_slot': int(np.searchsorted(slots, fast_ns, side='left')),
               'frame': 'slow-anchor delta-base; nominal 100 ms slots from S',
               'fast_ms': (time.perf_counter() - t0) * 1e3}
        if self.action_stats is not None:
            out['actions'] = denormalize(chunk.float().cpu().numpy(), self.action_stats)
        return out                                                            # 6
