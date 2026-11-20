"""Learned risk-token embedders (one per expert stream).

A risk token is a single hidden-size vector inserted in an expert's routed
sequence immediately before that expert's flow-time token:

    Action  stream : [fast | state | risk_S | t | noisy chunk]
    Tactile stream : [VQ codes (5) | force (5) | risk_F | tau | noisy chunk]

Input = the scalar r in [0, 1] plus an explicit validity bit v. Features:

    phi(r, v) = [sin(k*pi*r) * v, cos(k*pi*r) * v]_{k=1..K}, r*v, v, 1-v

so a missing measurement (v = 0) has its own learned output through the 1-v
channel and is never encoded as genuine zero risk (r = 0, v = 1).
"""
import math

import torch
import torch.nn as nn

RISK_TOKEN_VERSION = 1


class RiskEmbedder(nn.Module):
    def __init__(self, hidden_size: int, n_frequencies: int = 8):
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.n_frequencies + 3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def features(self, risk: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        r = risk.reshape(-1).float()
        v = valid.reshape(-1).to(device=r.device, dtype=torch.bool)
        if r.shape != v.shape:
            raise ValueError('risk and risk_valid must have the same batch size')
        if bool(v.any()) and not bool(torch.isfinite(r[v]).all()):
            raise ValueError('nonfinite risk marked valid')
        if bool(v.any()) and (bool((r[v] < 0).any()) or bool((r[v] > 1).any())):
            raise ValueError('valid risk must lie in [0, 1]')
        vf = v.float()
        r = torch.where(v, r, torch.zeros_like(r))
        k = torch.arange(1, self.n_frequencies + 1, device=r.device, dtype=torch.float32) * math.pi
        ang = r[:, None] * k[None]
        return torch.cat([torch.sin(ang) * vf[:, None], torch.cos(ang) * vf[:, None],
                          (r * vf)[:, None], vf[:, None], (1. - vf)[:, None]], dim=-1)

    def forward(self, risk: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """[B] risk, [B] bool validity -> [B, hidden]."""
        w = self.mlp[0].weight
        return self.mlp(self.features(risk.to(w.device), valid.to(w.device)).to(w.dtype))
