"""TactileVQVAE — full encoder + EMA quantizer + decoder over F6 windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import F6Encoder, F6PerFingerEncoder
from .decoder import F6Decoder, F6PerFingerDecoder
from .quantizer import VQEMAQuantizer


@dataclass
class TactileVQVAEConfig:
    window: int = 16
    in_channels: int = 30                 # 5 fingers × 6 dims (per-hand mode)
    hidden_channels: int = 128
    bottleneck_channels: int = 256
    embed_dim: int = 256
    n_strided_blocks: int = 2
    codebook_size: int = 1024
    commitment_weight: float = 0.25
    decay: float = 0.99
    revive_freq: int = 200
    revive_threshold: float = 1.0
    # Recon weighting
    use_magnitude_weight: bool = True
    weight_alpha: float = 2.0             # max extra weight on top of 1.0
    weight_tau: float = 4.0               # F6 magnitude scale (≈sqrt(T*5*6) for unit-norm)
    # Granularity: "hand" → 1 code per (hand, window); "finger" → 5 codes per
    # (hand, window).  Existing checkpoints default to "hand" via from_dict.
    granularity: str = "hand"
    n_fingers: int = 5
    per_finger_dim: int = 6
    # Hands in the raw history passed to the VLA (2 = original bimanual F6
    # [T,10,6]; 1 = TAPTIC right hand [T,5,3]).  Sizes the normalization stats.
    n_hands: int = 2
    # Provenance of the input signal (documentation + guards, not used in forward).
    input_signal: str = "f6"
    # What the decoder reconstructs.  "abs": the whole normalized window.
    # "delta": the window minus its last sample (x - x[:, -1:]); the last sample
    # is the current reading, which the policy already receives as a token, so
    # the code spends its bits on the within-window dynamics instead of the level.
    # The encoder input is the same absolute window in both modes.
    recon_target: str = "abs"
    # Residual VQ stages per code slot.  1 = plain VQ (one index per slot).
    # S > 1: stage s quantizes the residual left by stages < s; encode() returns
    # an extra trailing dim of size S.  Checkpoints without these fields load as
    # ("abs", 1), i.e. exactly the previous model.
    n_quantizers: int = 1

    @property
    def stats_dim(self) -> int:
        return self.n_hands * self.n_fingers * self.per_finger_dim
    # Misc
    init_mode: str = "uniform"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "TactileVQVAEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TactileVQVAE(nn.Module):
    def __init__(self, cfg: TactileVQVAEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.granularity not in ("hand", "finger"):
            raise ValueError(
                f"Unknown granularity {cfg.granularity!r}; expected 'hand' or 'finger'.")
        self.is_per_finger = (cfg.granularity == "finger")
        if cfg.recon_target not in ("abs", "delta"):
            raise ValueError(f"Unknown recon_target {cfg.recon_target!r}; expected 'abs' or 'delta'.")
        if cfg.n_quantizers < 1:
            raise ValueError("n_quantizers must be >= 1")

        if self.is_per_finger:
            self.encoder = F6PerFingerEncoder(
                window=cfg.window,
                per_finger_dim=cfg.per_finger_dim,
                n_fingers=cfg.n_fingers,
                hidden_channels=cfg.hidden_channels,
                bottleneck_channels=cfg.bottleneck_channels,
                embed_dim=cfg.embed_dim,
                n_strided_blocks=cfg.n_strided_blocks,
            )
            self.decoder = F6PerFingerDecoder(
                window=cfg.window,
                per_finger_dim=cfg.per_finger_dim,
                n_fingers=cfg.n_fingers,
                hidden_channels=cfg.hidden_channels,
                bottleneck_channels=cfg.bottleneck_channels,
                embed_dim=cfg.embed_dim,
                n_strided_blocks=cfg.n_strided_blocks,
            )
        else:
            self.encoder = F6Encoder(
                window=cfg.window,
                in_channels=cfg.in_channels,
                hidden_channels=cfg.hidden_channels,
                bottleneck_channels=cfg.bottleneck_channels,
                embed_dim=cfg.embed_dim,
                n_strided_blocks=cfg.n_strided_blocks,
            )
            self.decoder = F6Decoder(
                window=cfg.window,
                out_channels=cfg.in_channels,
                hidden_channels=cfg.hidden_channels,
                bottleneck_channels=cfg.bottleneck_channels,
                embed_dim=cfg.embed_dim,
                n_strided_blocks=cfg.n_strided_blocks,
                n_fingers=cfg.n_fingers,
            )

        def make_quantizer():
            return VQEMAQuantizer(
                codebook_size=cfg.codebook_size,
                embed_dim=cfg.embed_dim,
                commitment_weight=cfg.commitment_weight,
                decay=cfg.decay,
                revive_freq=cfg.revive_freq,
                revive_threshold=cfg.revive_threshold,
                init_mode=cfg.init_mode,
            )

        # Stage 0 keeps its historical name so old state_dicts load unchanged.
        self.quantizer = make_quantizer()
        self.residual_quantizers = nn.ModuleList(
            [make_quantizer() for _ in range(cfg.n_quantizers - 1)])

    @property
    def quantizers(self):
        return [self.quantizer, *self.residual_quantizers]

    def _quantize(self, z: torch.Tensor):
        """Residual VQ over [N, D].  Returns z_q [N, D], indices [N, S], loss, stage-0 info."""
        residual, z_q, idx, vq_loss, info0 = z, 0, [], 0, None
        for s, q in enumerate(self.quantizers):
            q_st, i, loss, info = q(residual)
            z_q = z_q + q_st
            residual = residual - q_st.detach()
            idx.append(i)
            vq_loss = vq_loss + loss
            if s == 0:
                info0 = info
        return z_q, torch.stack(idx, dim=-1), vq_loss, info0

    def _quantize_eval(self, z: torch.Tensor) -> torch.Tensor:
        """Indices [N, S] without EMA updates."""
        residual, idx = z, []
        for q in self.quantizers:
            i = q.encode_only(residual)
            residual = residual - q.embed[i]
            idx.append(i)
        return torch.stack(idx, dim=-1)

    def _embed_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """[..., S] indices → [..., D] sum of the per-stage codebook vectors."""
        return sum(q.embed[idx[..., s]] for s, q in enumerate(self.quantizers))

    def _squeeze_stage(self, idx: torch.Tensor) -> torch.Tensor:
        """Plain VQ keeps the historical index shape (no trailing stage dim)."""
        return idx.squeeze(-1) if self.cfg.n_quantizers == 1 else idx

    def _add_reference(self, recon: torch.Tensor, last: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.recon_target == "abs":
            return recon
        if last is None:
            raise ValueError("recon_target='delta' needs the window's last sample to rebuild it")
        return recon + last

    def _recon_weight(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Per-sample weight = 1 + α·sigmoid(magnitude/τ - 1).

        magnitude is the L2 norm of the *raw* (un-normalized) F6 window.
        Free-air windows (magnitude ≈ 0) get weight ≈ 1; high-contact windows
        approach 1 + α. Cheap, monotonic, and avoids hard-thresholding which
        can cause loss-discontinuity artifacts at the boundary.
        """
        cfg = self.cfg
        return 1.0 + cfg.weight_alpha * torch.sigmoid(magnitude / cfg.weight_tau - 1.0)

    def forward(
        self,
        f6: torch.Tensor,
        magnitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        f6        : [B, T, 5, 6]  normalized F6 window
        magnitude : [B]            raw F6 magnitude (for loss weighting). Optional.

        Returns dict:
            recon, indices, recon_loss, vq_loss, total_loss, plus quantizer info.

        Indices shape:
            - granularity="hand"   → [B]
            - granularity="finger" → [B, F=5]
        """
        z_e = self.encoder(f6)                                        # [B, D] or [B, F, D]
        if self.is_per_finger:
            B, F, D = z_e.shape
            z_q_flat, idx_flat, vq_loss, qinfo = self._quantize(z_e.reshape(B * F, D))
            z_q = z_q_flat.reshape(B, F, D)
            indices = idx_flat.reshape(B, F, -1)
        else:
            z_q, indices, vq_loss, qinfo = self._quantize(z_e)
        indices = self._squeeze_stage(indices)
        # "delta": the decoder predicts x - x_last; adding x_last back makes the
        # loss below identical in form for both targets.
        recon = self._add_reference(self.decoder(z_q), f6[:, -1:])   # [B, T, 5, D]

        per_sample = (recon - f6).pow(2).mean(dim=[1, 2, 3])          # [B]
        if self.cfg.use_magnitude_weight and magnitude is not None:
            w = self._recon_weight(magnitude.to(per_sample.device))
            recon_loss = (per_sample * w).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = per_sample.mean()

        total_loss = recon_loss + vq_loss

        return {
            "recon":          recon,
            "indices":        indices,
            "recon_loss":     recon_loss,
            "vq_loss":        vq_loss,
            "total_loss":     total_loss,
            "perplexity":     torch.tensor(qinfo["perplexity"], device=f6.device),
            "active_codes":   torch.tensor(qinfo["active_codes"], device=f6.device),
            "revived":        torch.tensor(qinfo["revived"], device=f6.device),
            "per_sample_recon": per_sample.detach(),
        }

    @torch.no_grad()
    def encode(self, f6: torch.Tensor) -> torch.Tensor:
        """Inference helper.
        Hand   mode: f6 [B, T, 5, D] → indices [B]     ([B, S] with n_quantizers=S>1).
        Finger mode: f6 [B, T, 5, D] → indices [B, 5]  ([B, 5, S] with S>1).
        """
        z_e = self.encoder(f6)
        if self.is_per_finger:
            B, F, D = z_e.shape
            idx = self._quantize_eval(z_e.reshape(B * F, D)).reshape(B, F, -1)
        else:
            idx = self._quantize_eval(z_e)
        return self._squeeze_stage(idx)

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor,
                       last: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Indices as returned by encode() → recon [B, T, 5, D].

        last : [B, 1, 5, D] normalized last sample of the window; required when
               recon_target='delta' (the codes only describe the change).
        """
        if self.cfg.n_quantizers == 1:
            indices = indices.unsqueeze(-1)
        return self._add_reference(self.decoder(self._embed_indices(indices)), last)
