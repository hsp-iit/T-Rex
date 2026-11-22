"""
Qwen3VLVLAModel – the full VLA model.

Architecture
────────────
  visual             : Qwen3VL ViT (frozen during visual feature extraction)
  model              : Qwen3VLModelMoT – three-expert Qwen3-VL text backbone
  x_embedder         : noisy actions  → hidden_size
  t_embedder         : flow time      → hidden_size
  tacf6_embed        : F6 tactile vec → hidden_size   (optional)
  deform_proj        : tactile deform → hidden_size   (optional)
  state_embed        : robot state    → hidden_size   (optional)
  final_layer        : hidden_size    → action_dim    (action expert velocity)
  final_layer_tactile: hidden_size    → action_dim    (tactile expert velocity)
  risk_embedder_action : (r_S, valid) → hidden_size   (optional, OURS)
  risk_embedder_tactile: (r_F, valid) → hidden_size   (optional, OURS)

Cascaded flow matching
──────────────────────
The action expert handles the upper segment τ ∈ [τ_split, 1] and produces an
intermediate state x_split plus cached [latent | action] KV.  The tactile
expert continues the same flow on the lower segment τ ∈ [0, τ_split] from
x_split using the cached KV plus fresh tactile observations, and its
integrated output IS the executed action chunk (no Â + Δa addition).
Both experts predict the same velocity target ε − A_demo on disjoint
sub-intervals; the action expert is additionally trained on the full [0, 1]
range so it can run standalone if tactile drops out.

Risk tokens (use_risk_tokens=True)
──────────────────────────────────
One learned token per expert stream, inserted before that stream's flow-time
token (the causal rule, expert routing and prediction slices are unchanged —
the chunk tokens stay last):
  Action  : [latent | fast | state | risk_S | t | x_t(16)]      r_S at slow tick S
  Tactile : [cache  | tac_obs(10) | risk_F | tau | x_tau(16)]  r_F at fast tick F
r_S is baked into the Action KV cache; r_F is fresh at every fast tick and
never overwrites the cached r_S. Missing risk = explicit validity bit, never r=0.
When use_risk_tokens=False every method is numerically identical to the
baseline and passing risk inputs is an error (no silent ignore).

Methods:
  forward_flow_action_partial    — slow tick (action expert, upper segment).
  forward_flow_action_ladder     — slow tick, snapshots at tau 0.6/0.4/0.2.
  tactile_flow_continue          — fast tick (tactile expert, lower segment).
  tactile_flow_continue_from_ladder — fast tick from a selected snapshot.
  tactile_flow_train_step        — training-time single tactile forward.
"""

import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from .modeling_qwen3vl_mot import Qwen3VLModelMoT
from diffusion import ActionEmbedder, TimestepEmbedder, FinalLayer
from .flow_matching import flow_time, Handover
from .cache_ladder import ActionCacheLadder, ActionHandoverSnapshot
from .risk_tokens import RiskEmbedder, RISK_TOKEN_VERSION
from models import DeformEncoder


# ── Default image-pad token id (matches Qwen3-VL / Qwen2-VL) ────────────────
_DEFAULT_IMAGE_TOKEN_ID = 151655   # <|image_pad|>


class Qwen3VLVLAModel(nn.Module):
    """
    Full VLA model wrapping the Qwen3-VL visual encoder and the
    three-expert (MoT) Qwen3 text backbone.
    """

    def __init__(
        self,
        config,
        action_dim: int = 29,
        action_chunk:        int   = 8,
        tacf6_dim:           int   = 6,
        use_tactile_deform:  bool  = False,
        use_robot_state:     bool  = False,
        image_token_id:      int   = _DEFAULT_IMAGE_TOKEN_ID,
        tactile_intermediate_size: int = None,
        n_flare_tokens_per_frame: int = 0,
        n_flare_steps:           int = 0,
        flare_layer_index:       int = -1,
        use_tactile_code:        bool = False,
        vqvae_codebook_size:     int  = 64,
        use_tactile_vqvae:       bool = False,
        vqvae_config:            dict = None,
        use_risk_tokens:         bool = False,
        risk_n_frequencies:      int  = 8,
        risk_future_head:        bool = False,
        risk_future_probe:       bool = False,
    ):
        super().__init__()
        self.config             = config
        self.action_dim         = action_dim
        self.action_chunk       = action_chunk
        self.tacf6_dim          = tacf6_dim
        self.use_tactile_deform = use_tactile_deform
        self.use_robot_state    = use_robot_state
        self.image_token_id     = image_token_id
        self.tactile_intermediate_size = tactile_intermediate_size
        self.n_flare_tokens_per_frame = n_flare_tokens_per_frame
        self.n_flare_steps            = n_flare_steps
        self.n_flare_tokens           = n_flare_tokens_per_frame * n_flare_steps
        self.flare_layer_index        = flare_layer_index
        # Embedded (on-the-fly) VQ-VAE turns raw F6 history → discrete codes
        # inside the model.  It implies use_tactile_code (the code embedder is
        # still what maps an integer code → a hidden token).  When disabled the
        # model expects pre-computed codes from the dataloader (legacy path).
        self.use_tactile_vqvae        = use_tactile_vqvae
        self.use_tactile_code         = bool(use_tactile_code or use_tactile_vqvae)
        self.vqvae_codebook_size      = vqvae_codebook_size
        self.vqvae_config             = vqvae_config
        self.use_risk_tokens          = bool(use_risk_tokens)
        self.risk_n_frequencies       = int(risk_n_frequencies)
        self.risk_future_head         = bool(risk_future_head)
        self.risk_future_probe        = bool(risk_future_probe)

        self.visual = None

        self.model = Qwen3VLModelMoT(config, tactile_intermediate_size=tactile_intermediate_size)

        H = config.hidden_size
        self.x_embedder          = ActionEmbedder(action_dim, H)
        self.t_embedder          = TimestepEmbedder(H)
        self.final_layer         = FinalLayer(H, action_dim)   # action expert → velocity
        self.final_layer_tactile = FinalLayer(H, action_dim)   # tactile expert → residual
        self.tacf6_embedder      = ActionEmbedder(tacf6_dim, H)

        if use_tactile_deform:
            self.deform_encoder = DeformEncoder()
            self.deform_proj    = ActionEmbedder(28800, H) # 128 * 15 * 15

        # VQ-VAE tactile code embedder (one code per hand → 2 tokens, or one
        # code per finger → 2*n_fingers tokens).  Created whenever codes are
        # used (pre-computed OR produced on-the-fly by the embedded VQ-VAE) so
        # toggling everything off gives a graph identical to the pre-feature
        # checkpoint.
        if self.use_tactile_code:
            self.tactile_code_embedder = nn.Embedding(vqvae_codebook_size, H)

        # Embedded VQ-VAE: the encoder + EMA quantizer that map a raw F6
        # history window → discrete codes, run on-the-fly during training and
        # inference.  Frozen (never trained, never EMA-updated — encode() uses
        # encode_only()).  Weights + F6 normalization stats are loaded from a
        # standalone VQ-VAE checkpoint via load_tactile_vqvae() (training) or
        # baked into model.pt by scripts/merge_vqvae_into_ckpt.py (deployment).
        self.tactile_vqvae = None
        if self.use_tactile_vqvae:
            if vqvae_config is None:
                raise ValueError(
                    "use_tactile_vqvae=True requires vqvae_config "
                    "(the TactileVQVAEConfig fields from the VQ-VAE checkpoint).")
            from tactile_vqvae.models.tactile_vqvae import (
                TactileVQVAE, TactileVQVAEConfig)
            vq_cfg = TactileVQVAEConfig.from_dict(vqvae_config)
            self.tactile_vqvae = TactileVQVAE(vq_cfg)
            self.tactile_vqvae.eval()
            for p in self.tactile_vqvae.parameters():
                p.requires_grad = False
            # Min-max normalization stats, registered as buffers so they travel
            # with the checkpoint: [60] = 2 hands × 5 fingers × 6 (original F6),
            # [15] = 1 hand × 5 fingers × 3 (TAPTIC SDK force).
            stats_dim = vq_cfg.stats_dim
            self.register_buffer("tacf6_vqvae_min",
                                 torch.zeros(stats_dim, dtype=torch.float32), persistent=True)
            self.register_buffer("tacf6_vqvae_max",
                                 torch.ones(stats_dim, dtype=torch.float32), persistent=True)
            self.register_buffer("tacf6_vqvae_mask",
                                 torch.ones(stats_dim, dtype=torch.bool), persistent=True)

        if use_robot_state:
            self.state_embedder = ActionEmbedder(action_dim, H)

        # Risk tokens (OURS): one embedder per expert, trained with that expert.
        # Created only when enabled, so the baseline weight schema is unchanged.
        if self.use_risk_tokens:
            self.risk_embedder_action  = RiskEmbedder(H, self.risk_n_frequencies)
            self.risk_embedder_tactile = RiskEmbedder(H, self.risk_n_frequencies)

        # Future-risk heads (auxiliary, never used to produce actions). Per chunk slot k they predict
        # the causal risk r(t_k) at the slot's time, strictly after the observation tick:
        #  - future_risk_head: on the Tactile expert's chunk tokens, trained jointly (aux loss);
        #  - future_risk_probe: linear probe on the Action expert's chunk tokens, fed DETACHED
        #    hidden states (it never changes the model), i.e. how much future risk the slow stream sees.
        if self.risk_future_head:
            self.future_risk_head = nn.Linear(H, 1)
        if self.risk_future_probe:
            self.future_risk_probe = nn.Linear(H, 1)

        # Flare visual prediction tokens for the latent expert
        if self.n_flare_tokens > 0:
            # flare_queries è un parametro addestrabile [1,K,H] K = 32
            self.flare_queries = nn.Parameter(
                torch.randn(1, self.n_flare_tokens, H) * 0.02)
            # flare_proj è una piccola MLP che trasforma le rappresentazioni dei token
            # FLARE nello spazio in cui verranno confrontate con i target visivi.
            self.flare_proj = nn.Sequential(
                nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))

        # Lightweight callable for get_rope_index (NOT an nn.Module — avoids
        # registering the full base model as a submodule).
        # Use object.__setattr__ to bypass nn.Module registration.
        object.__setattr__(self, '_rope_index_fn', None)

    @classmethod
    def from_pretrained_qwen3vl(
        cls,
        pretrained_path: str,
        action_dim:         int  = 29,
        action_chunk:       int  = 8,
        tacf6_dim:          int  = 6,
        use_tactile_deform: bool = False,
        use_robot_state:    bool = False,
        torch_dtype              = torch.bfloat16,
        tactile_intermediate_size: int = None,
        n_flare_tokens_per_frame: int = 0,
        n_flare_steps:            int = 0,
        flare_layer_index:        int = -1,
        use_tactile_code:         bool = False,
        vqvae_codebook_size:      int  = 64,
        use_tactile_vqvae:        bool = False,
        vqvae_config:             dict = None,
        use_risk_tokens:          bool = False,
        risk_n_frequencies:       int  = 8,
        risk_future_head:         bool = False,
        risk_future_probe:        bool = False,
    ) -> "Qwen3VLVLAModel":
        """
        Build a Qwen3VLVLAModel by:
          1. Loading `Qwen3VLForConditionalGeneration` from `pretrained_path`.
          2. Copying the visual tower directly.
          3. Loading the text model weights into Qwen3VLModelMoT (strict=False).
          4. VLA-specific modules are randomly initialised.
        """
        from transformers import AutoConfig

        # Load the pretrained Qwen3-VL model for weight extraction
        # STEP 1: carica il modello Qwen3-VL
        try:
            from transformers import Qwen3VLForConditionalGeneration
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_path, torch_dtype=torch_dtype, trust_remote_code=True
            )
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_path, torch_dtype=torch_dtype, trust_remote_code=True
            )

        config = base_model.config
        # Qwen3-VL text config lives in config.model_config or directly in config
        text_config = getattr(config, "text_config", config)

        image_token_id = getattr(config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID)

        # Construct VLA model
        # STEP 2: costruisce il VLA
        # È in questa fase che vengono creati gli embedder, i Transformer specializzati e le prediction head.   
        vla = cls(
            config = text_config,
            action_dim = action_dim,
            action_chunk = action_chunk,
            tacf6_dim = tacf6_dim,
            use_tactile_deform = use_tactile_deform,
            use_robot_state = use_robot_state,
            image_token_id = image_token_id,
            tactile_intermediate_size = tactile_intermediate_size,
            n_flare_tokens_per_frame = n_flare_tokens_per_frame,
            n_flare_steps = n_flare_steps,
            flare_layer_index = flare_layer_index,
            use_tactile_code = use_tactile_code,
            vqvae_codebook_size = vqvae_codebook_size,
            use_tactile_vqvae = use_tactile_vqvae,
            vqvae_config = vqvae_config,
            use_risk_tokens = use_risk_tokens,
            risk_n_frequencies = risk_n_frequencies,
            risk_future_head = risk_future_head,
            risk_future_probe = risk_future_probe,
        )
        # Capture only the get_rope_index function — do NOT store the full
        # base model as an attribute, because nn.Module.__setattr__ would
        # register it as a submodule (duplicating ~2B parameters and breaking
        # the freeze logic for the visual encoder).
        # In Qwen3-VL, get_rope_index lives on Qwen3VLModel (base_model.model),
        # NOT on Qwen3VLForConditionalGeneration (base_model).
        _inner = base_model.model if hasattr(base_model, "model") else base_model
        if hasattr(_inner, "get_rope_index"):
            object.__setattr__(vla, '_rope_index_fn', _inner.get_rope_index)

        # Copy visual tower (base_model.visual is a @property → base_model.model.visual)
        # STEP 3: Il modello VLA riutilizza il modulo visivo del modello pretrained.
        vla.visual = base_model.visual

        # Load text model weights into Qwen3VLModelMoT (strict=False).
        # In Qwen3-VL (transformers >=4.57), base_model.model is Qwen3VLModel
        # which wraps both visual and language_model.  State dict keys are
        # prefixed: 'visual.*' and 'language_model.*'.
        # In Qwen2-VL (older), base_model.model is the text model directly
        # and keys have no prefix (embed_tokens.*, layers.*, norm.*).
        raw_sd = base_model.model.state_dict()

        # Detect the key structure and extract text-model weights
        lang_prefix = "language_model."
        has_lang_prefix = any(k.startswith(lang_prefix) for k in raw_sd)

        text_sd = {}
        if has_lang_prefix:
            # Qwen3-VL: strip 'language_model.' prefix, skip visual keys
            for k, v in raw_sd.items():
                if k.startswith(lang_prefix):
                    text_sd[k[len(lang_prefix):]] = v
        else:
            # Qwen2-VL / direct text model: use keys as-is
            text_sd = dict(raw_sd)

        # STEP 4: Carica i pesi linguistici nel MoT.
        missing, unexpected = vla.model.load_state_dict(text_sd, strict=False)
        print(f"[Qwen3VLVLAModel] Text model loaded – missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            # Filter out expected _action / _tactile keys for cleaner logging
            truly_missing = [k for k in missing if "_action" not in k and "_tactile" not in k]
            expected_missing = len(missing) - len(truly_missing)
            if truly_missing:
                print(f"  WARNING – unexpected missing base keys: {truly_missing[:10]} ...")
            print(f"  New MoT expert weights (expected missing): {expected_missing}")

        # Free the base model's language_model to save memory.
        # get_rope_index only needs config + image_token_id, not the weights,
        # but it's a bound method so we keep the Qwen3VLModel alive.
        # Deleting language_model parameters frees ~1.5B params.
        if hasattr(_inner, "language_model"):
            del _inner.language_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Qwen3VLVLAModel] Freed base model language_model to save memory.")

        return vla

    def load_deform_encoder_weights(self, ckpt_path: str):
        if not self.use_tactile_deform:
            return
        if not os.path.exists(ckpt_path):
            print(f"Warning: DeformEncoder checkpoint not found at {ckpt_path}")
            return
        print(f"Loading DeformEncoder weights from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder."):]] = v
            elif k in self.deform_encoder.state_dict():
                encoder_sd[k] = v
        missing, _ = self.deform_encoder.load_state_dict(encoder_sd, strict=False)
        if missing:
            print(f"  DeformEncoder missing keys: {missing}")
        print("DeformEncoder weights loaded.")

    def load_tactile_vqvae(self, ckpt_path_or_blob):
        """Load weights + F6 normalization stats into the embedded VQ-VAE.

        Accepts either a path to a standalone VQ-VAE checkpoint (the
        ``{config, model_state, stats}`` blob saved by tactile_vqvae/train.py)
        or an already-loaded blob dict.  Keeps the module frozen + eval.
        """
        if not self.use_tactile_vqvae or self.tactile_vqvae is None:
            print("Warning: load_tactile_vqvae called but use_tactile_vqvae is off.")
            return
        blob = ckpt_path_or_blob
        if isinstance(blob, str):
            blob = torch.load(blob, map_location="cpu", weights_only=False)
        missing, unexpected = self.tactile_vqvae.load_state_dict(
            blob["model_state"], strict=False)
        if missing:
            print(f"  [tactile_vqvae] missing keys: {missing[:6]} ...")
        if unexpected:
            print(f"  [tactile_vqvae] unexpected keys: {unexpected[:6]} ...")
        stats = blob["stats"]
        self.tacf6_vqvae_min.copy_(torch.as_tensor(stats["tacf6_min"], dtype=torch.float32))
        self.tacf6_vqvae_max.copy_(torch.as_tensor(stats["tacf6_max"], dtype=torch.float32))
        self.tacf6_vqvae_mask.copy_(torch.as_tensor(stats["tacf6_mask"], dtype=torch.bool))
        self.tactile_vqvae.eval()
        for p in self.tactile_vqvae.parameters():
            p.requires_grad = False
        print(f"[Qwen3VLVLAModel] embedded VQ-VAE loaded "
              f"(codebook={self.tactile_vqvae.cfg.codebook_size}, "
              f"granularity={self.tactile_vqvae.cfg.granularity}, "
              f"window={self.tactile_vqvae.cfg.window}).")

    @torch.no_grad()
    def encode_tactile_f6_history(self, f6_history_raw: torch.Tensor) -> torch.Tensor:
        """Encode a raw (un-normalized) F6 history window into per-hand codes.

        f6_history_raw : [B, T, 10, 6]  (two hands × 5 fingers; T = VQ-VAE window),
                         or [B, T, 5, 3] for the TAPTIC SDK-force VQ-VAE (one hand).
                         Normalized internally with the embedded VQ-VAE stats,
                         so callers pass raw F6 — exactly what the dataloader /
                         robot reads off the sensor.

        Returns
        -------
        codes : [B, n_codes] int64, flattened to match the pre-computed layout
                consumed by `tactile_code_embedder`:
                  hand   granularity → [B, 2]            (left, right)
                  finger granularity → [B, 2*n_fingers]  (L[5], R[5])
        """
        if self.tactile_vqvae is None:
            raise RuntimeError("encode_tactile_f6_history requires use_tactile_vqvae.")
        vq = self.tactile_vqvae
        w_dtype = next(vq.parameters()).dtype
        x = f6_history_raw.to(device=self.tacf6_vqvae_min.device, dtype=torch.float32)
        B, T, NF, D = x.shape
        n_fingers = vq.cfg.n_fingers
        if D != vq.cfg.per_finger_dim or NF * D != self.tacf6_vqvae_min.numel():
            raise ValueError(
                f"tactile history [*, {T}, {NF}, {D}] does not match the embedded VQ-VAE "
                f"(per_finger_dim={vq.cfg.per_finger_dim}, stats={self.tacf6_vqvae_min.numel()})")
        # Min-max normalize to [-1, 1] (matches TacF6Stats.normalize).
        flat = x.reshape(B, T, NF * D)                                  # [B, T, NF*D]
        denom = (self.tacf6_vqvae_max - self.tacf6_vqvae_min) + 1e-8
        normed = torch.clamp(
            2.0 * (flat - self.tacf6_vqvae_min) / denom - 1.0, -1.0, 1.0)
        normed = torch.where(self.tacf6_vqvae_mask, normed, flat)
        normed = normed.reshape(B, T, NF, D)

        n_hands = NF // n_fingers
        per_hand_codes = []
        for h in range(n_hands):
            wh = normed[:, :, h * n_fingers:(h + 1) * n_fingers, :].to(w_dtype)  # [B, T, 5, D]
            idx = vq.encode(wh)                                          # [B] or [B, 5]
            if idx.dim() == 1:
                idx = idx.unsqueeze(1)                                   # [B, 1]
            per_hand_codes.append(idx.long())
        return torch.cat(per_hand_codes, dim=1)                         # [B, n_codes]

    def initialize_vla_weights(self):
        """Xavier-init all new VLA-specific linear layers.

        Under cascaded flow matching both `final_layer` (action expert) and
        `final_layer_tactile` (tactile expert) are velocity predictors on
        disjoint sub-intervals of the same flow.  Only `final_layer` is
        zero-inited so v_act starts at zero; the tactile head must start
        non-trivial so the lower-segment refinement signal is alive at step 0.
        """
        for m in [self.x_embedder, self.t_embedder, self.final_layer,
                  self.final_layer_tactile, self.tacf6_embedder]:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
        if self.use_robot_state:
            for mm in self.state_embedder.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
        if self.use_tactile_deform:
            for mm in self.deform_proj.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
        if self.use_tactile_code:
            nn.init.normal_(self.tactile_code_embedder.weight, mean=0.0, std=0.02)
        if self.use_risk_tokens:
            self.initialize_risk_weights()
        for name in ('future_risk_head', 'future_risk_probe'):
            if hasattr(self, name):
                nn.init.zeros_(getattr(self, name).weight)
                nn.init.zeros_(getattr(self, name).bias)
        nn.init.zeros_(self.final_layer.mlp.fc2.weight)
        if self.final_layer.mlp.fc2.bias is not None:
            nn.init.zeros_(self.final_layer.mlp.fc2.bias)
        if self.n_flare_tokens > 0:
            for mm in self.flare_proj.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)

    # ── Risk tokens ────────────────────────────────────────────────────────
    def initialize_risk_weights(self):
        """Xavier/zero-bias init of the (new) risk embedders. Returns their param names."""
        if not self.use_risk_tokens:
            return []
        names = []
        for prefix, m in (('risk_embedder_action', self.risk_embedder_action),
                          ('risk_embedder_tactile', self.risk_embedder_tactile)):
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
            names += [f'{prefix}.{n}' for n, _ in m.named_parameters()]
        return names

    @property
    def n_risk_tokens(self) -> int:
        return 1 if self.use_risk_tokens else 0

    def risk_token(self, risk, risk_valid, expert, batch_size, device, dtype):
        """[B, n_risk, H] risk token for `expert` ('action' = r_S, 'tactile' = r_F).

        Strict contract: a risk-conditioned model REQUIRES (risk, risk_valid) on
        every path (missing data = risk_valid False, never omitted); a baseline
        model rejects them, so a configuration mix-up cannot be silently ignored.
        """
        H = self.config.hidden_size
        if not self.use_risk_tokens:
            if risk is not None or risk_valid is not None:
                raise ValueError('this model has no risk tokens; pass risk=None, risk_valid=None')
            return torch.empty((batch_size, 0, H), device=device, dtype=dtype)
        if risk is None or risk_valid is None:
            raise ValueError(f'risk-conditioned model needs {expert} risk and risk_valid '
                             '(use risk_valid=False for missing measurements)')
        risk = torch.as_tensor(risk).reshape(-1)
        risk_valid = torch.as_tensor(risk_valid).reshape(-1)
        if risk.shape[0] != batch_size or risk_valid.shape[0] != batch_size:
            raise ValueError(f'{expert} risk batch {risk.shape[0]} != {batch_size}')
        emb = {'action': self.risk_embedder_action, 'tactile': self.risk_embedder_tactile}[expert]
        return emb(risk, risk_valid).unsqueeze(1).to(device)

    def architecture_contract(self) -> dict:
        """What a checkpoint of this model needs to be reloaded faithfully."""
        return {'family': 'trex_qwen3vl_mot_cascaded', 'action_dim': self.action_dim,
                'action_chunk': self.action_chunk, 'tacf6_dim': self.tacf6_dim,
                'use_robot_state': self.use_robot_state, 'use_tactile_code': self.use_tactile_code,
                'use_tactile_vqvae': self.use_tactile_vqvae, 'n_flare_tokens': self.n_flare_tokens,
                'risk_tokens': self.use_risk_tokens,
                'risk_token_version': RISK_TOKEN_VERSION if self.use_risk_tokens else None,
                'risk_n_frequencies': self.risk_n_frequencies if self.use_risk_tokens else None,
                'risk_future_head': self.risk_future_head, 'risk_future_probe': self.risk_future_probe,
                'action_stream': 'fast|state|' + ('risk_S|' if self.use_risk_tokens else '') + 't|x',
                'tactile_stream': 'tac_obs|' + ('risk_F|' if self.use_risk_tokens else '') + 'tau|x'}

    # 1. Costruisce gli embedding dei token testuali
    # 2. Sostituisce gli embedding dei segnaposto visivi
    def prepare_inputs_embeds(
        self,
        input_ids:       torch.LongTensor,
        pixel_values:    Optional[torch.Tensor]  = None,
        image_grid_thw:  Optional[torch.Tensor]  = None,
    ) -> torch.Tensor:
        """
        Build the combined vision-language embedding sequence.

        input_ids      : [B, L]  (contains <|image_pad|> at image positions)
        pixel_values   : [N_patches, 3, patch_h, patch_w]  (all images flattened)
        image_grid_thw : [N_images, 3]  – (temporal, height, width) grid per image
        """
        # Text embeddings from the MoT backbone's embedding table
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None and self.visual is not None:
            # Run the visual tower
            pixel_values = pixel_values.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            out = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Qwen3-VL returns (hidden_states, deepstack_features) tuple;
            # Qwen2-VL returns a plain tensor.
            image_features = out[0] if isinstance(out, (tuple, list)) else out
            # image_features: [total_merged_tokens, hidden_size]

            # Locate image-pad positions and inject features
            image_mask = (input_ids == self.image_token_id)
            if image_mask.any():
                inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

        return inputs_embeds

    def get_rope_index(
        self,
        input_ids:      torch.LongTensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Delegate M-RoPE index computation to the underlying Qwen3VL base model.
        Returns (position_ids, mrope_position_deltas).

        position_ids shape: [3, batch, latent_seq_len]
        """
        if self._rope_index_fn is not None:
            return self._rope_index_fn(
                input_ids       = input_ids,
                image_grid_thw  = image_grid_thw,
                attention_mask  = attention_mask,
            )
        # Fallback: plain sequential 1-D position ids
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        return position_ids, None

    def _embed_tactile_observations(
        self,
        tactile_f6: Optional[torch.Tensor],
        tactile_deform: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        tactile_codes: Optional[torch.Tensor] = None,
        tactile_f6_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build [B, n_obs, H] tactile observation embedding (codes + vec + deform).

        Codes can arrive two ways:
          * pre-computed `tactile_codes` ([B, n_codes] int64) — legacy path.
          * `tactile_f6_history` ([B, T, 10, 6] raw) — encoded on-the-fly by the
            embedded VQ-VAE.  Used whenever the model carries `tactile_vqvae`.
        Each integer code becomes one prepended token via `tactile_code_embedder`.
        """
        # On-the-fly: derive codes from the raw F6 history if the model has the
        # embedded VQ-VAE and the caller did not pass pre-computed codes.
        if (self.use_tactile_vqvae
                and self.tactile_vqvae is not None
                and tactile_codes is None
                and tactile_f6_history is not None
                and tactile_f6_history.shape[1] > 0):
            tactile_codes = self.encode_tactile_f6_history(tactile_f6_history)

        tac_parts = []
        if (tactile_codes is not None
                and self.use_tactile_code
                and tactile_codes.shape[1] > 0):
            code_emb = self.tactile_code_embedder(tactile_codes.to(device).long())
            tac_parts.append(code_emb.to(dtype))
        if tactile_f6 is not None and tactile_f6.shape[1] > 0:
            tac_parts.append(self.tacf6_embedder(tactile_f6.to(dtype)))
        if (tactile_deform is not None
                and tactile_deform.shape[1] > 0
                and self.use_tactile_deform):
            Bd, nf, C, Hd, Wd = tactile_deform.shape
            flat = tactile_deform.view(-1, C, Hd, Wd).to(dtype)
            feats = self.deform_encoder(flat).view(Bd, nf, -1)
            tac_parts.append(self.deform_proj(feats.to(dtype)))
        if not tac_parts:
            raise ValueError(
                "tactile residual flow requires tactile_f6 or tactile_deform")
        return torch.cat(tac_parts, dim=1)

    @staticmethod
    def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
        """Manual clone of a DynamicCache: torch.deepcopy fails on non-leaf
        tensors, so we copy the per-layer K/V tensors with detach().clone()."""
        new_cache = DynamicCache()
        # Mirror per-layer DynamicLayer instances with cloned K/V tensors.
        if hasattr(cache, "layers") and isinstance(cache.layers, list):
            from transformers.cache_utils import DynamicLayer
            for layer in cache.layers:
                new_layer = DynamicLayer()
                if getattr(layer, "is_initialized", False):
                    new_layer.dtype  = layer.dtype
                    new_layer.device = layer.device
                    new_layer.keys   = layer.keys.detach().clone()
                    new_layer.values = layer.values.detach().clone()
                    new_layer.is_initialized = True
                new_cache.layers.append(new_layer)
        else:
            # Older (pre-4.55) API: key_cache / value_cache lists at top level.
            if hasattr(cache, "key_cache"):
                new_cache.key_cache = [k.detach().clone() for k in cache.key_cache]
                new_cache.value_cache = [v.detach().clone() for v in cache.value_cache]
                new_cache._seen_tokens = getattr(cache, "_seen_tokens", 0)
            else:
                raise NotImplementedError(
                    "Unrecognized DynamicCache layout; cannot clone for "
                    "tactile_flow_continue.")
        return new_cache

    # ──────────────────────────────────────────────────────────────────────
    # Cascaded flow matching
    #
    # The action expert handles the upper segment τ ∈ [τ_split, 1] and the
    # tactile expert continues on the lower segment τ ∈ [0, τ_split].  Both
    # are velocity predictors for the same action-distribution flow target
    # u = ε − A_demo on disjoint sub-intervals, so the tactile expert's
    # integrated output IS the clean action — no Â + Δa addition.  The
    # action expert is still trained on the full τ ∈ [0, 1] range so it
    # can run standalone if tactile drops out.
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_flow_action_full(
        self,
        inputs_embeds: torch.Tensor,            # [B, L_latent, H]
        position_ids: torch.Tensor,             # [3, B, L_latent]
        noise: torch.Tensor,                    # [B, n_chunk, action_dim]
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        flow_time_precision: str = 'fp32',
        risk: Optional[torch.Tensor] = None,
        risk_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Action-expert-only full flow τ ∈ [0, 1].

        Tactile-blind baseline used for the "without tactile expert" ablation.
        The action expert is trained on the full τ ∈ [0, 1] range so it can
        integrate the whole trajectory standalone — this method runs all
        `num_steps` Euler steps using only the action expert and returns the
        clean action chunk directly.  No KV cache is returned because there
        is no downstream tactile pass.

        Returns
        -------
        clean_chunk : [B, n_chunk, action_dim]  the integrated action at τ=0.
        """
        device = noise.device
        dtype  = torch.bfloat16
        if flow_time_precision not in ('fp32', 'legacy_bf16'):
            raise ValueError('invalid flow_time_precision')
        precise = flow_time_precision == 'fp32'
        dt     = torch.tensor(-1.0 / num_steps, dtype=torch.float32 if precise else dtype, device=device)
        x_t    = noise.float() if precise else noise.to(dtype)
        time   = torch.tensor(1.0, dtype=dtype, device=device)
        n_chunk = noise.shape[1]
        B      = inputs_embeds.shape[0]
        H      = inputs_embeds.shape[2]

        if fast_embeds is None:
            fast_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
        if state_embeds is None:
            state_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
        n_state = state_embeds.shape[1]
        L_latent = inputs_embeds.shape[1]
        risk_tok = self.risk_token(risk, risk_valid, 'action', B, device, dtype)   # r_S
        n_risk = risk_tok.shape[1]

        past_kv = None
        n_act = 0
        for step in range(num_steps):
            if precise:
                time = flow_time(step, num_steps, device=device)
            timesteps = self.t_embedder(time.expand(B)).unsqueeze(1)
            noisy_act = self.x_embedder(x_t.to(dtype))
            act_parts = [fast_embeds]
            if n_state > 0:
                act_parts.append(state_embeds)
            if n_risk > 0:
                act_parts.append(risk_tok)
            act_parts += [timesteps, noisy_act]
            act_seq = torch.cat(act_parts, dim=1)
            n_act = act_seq.shape[1]

            if past_kv is None:
                full_embeds = torch.cat([inputs_embeds, act_seq], dim=1)
                outputs = self.model(
                    inputs_embeds=full_embeds,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    latent_indexes=torch.arange(0, L_latent, device=device),
                    action_indexes=torch.arange(L_latent, L_latent + n_act, device=device),
                    tactile_indexes=torch.arange(0, 0, device=device),
                )
            else:
                past_kv.crop(-n_act)
                extended_pos = self.model._extend_position_ids(position_ids, n_act, 0)
                act_pos = extended_pos[..., -n_act:]
                outputs = self.model(
                    inputs_embeds=act_seq,
                    position_ids=act_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                    latent_indexes=torch.arange(0, 0, device=device),
                    action_indexes=torch.arange(0, n_act, device=device),
                    tactile_indexes=torch.arange(0, 0, device=device),
                )

            hidden = outputs.last_hidden_state
            v_act  = self.final_layer(hidden[:, -n_chunk:, :])
            x_t    = x_t + dt * (v_act.float() if precise else v_act)
            if not precise:
                time = time + dt
            past_kv = outputs.past_key_values

        return x_t.to(dtype)


    # il suo output è una tupla di 4 elementi
    # x_t, past_kv, n_action_in_cache, tau_split
    @torch.no_grad()
    def forward_flow_action_partial(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
        noise: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None,
        num_steps_total: int = 10, split_step: int = 6,
        refresh_clean_kv: bool = True, flow_time_precision: str = 'fp32',
        risk: Optional[torch.Tensor] = None, risk_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, "DynamicCache", int, float]:
        """Single-split slow tick (r_S token when the model has risk tokens).
        Numerically identical to the baseline when use_risk_tokens=False."""
        return self._action_trajectory(
            inputs_embeds, position_ids, noise, attention_mask, state_embeds, fast_embeds,
            num_steps_total, split_step, refresh_clean_kv, flow_time_precision, risk, risk_valid)

    @torch.no_grad()
    def forward_flow_action_ladder(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
        noise: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None,
        risk: Optional[torch.Tensor] = None, risk_valid: Optional[torch.Tensor] = None,
        slow_tick_ns: Optional[int] = None,
    ) -> ActionCacheLadder:
        """One slow tick: ONE Action trajectory (8 Euler updates + 1 refresh =
        9 forwards) with coherent snapshots at tau 0.6 / 0.4 / 0.2, all
        conditioned on the same r_S. Requires eval mode, FP32 flow time, latent-only
        position ids and an unpadded context (the continuation forwards after the
        first one do not carry a padding mask — pre-existing T-Rex behaviour)."""
        if self.training:
            raise ValueError('cache ladder requires model.eval()')
        if (position_ids.ndim not in (2, 3)
                or position_ids.shape[-1] != inputs_embeds.shape[1]
                or position_ids.shape[-2] != inputs_embeds.shape[0]
                or (position_ids.ndim == 3 and position_ids.shape[0] != 3)):
            raise ValueError('cache ladder requires latent-only position IDs')
        if noise.shape != (inputs_embeds.shape[0], self.action_chunk, self.action_dim):
            raise ValueError('cache ladder noise shape must be [B, action_chunk, action_dim]')
        if attention_mask is not None and (attention_mask.ndim != 2
                or attention_mask.shape[0] != inputs_embeds.shape[0]
                or not torch.all(attention_mask == 1)):
            raise ValueError('cache ladder requires an unpadded attention context')
        sink = {}
        meta = dict(slow_tick_ns=None if slow_tick_ns is None else int(slow_tick_ns),
                    risk_S=None if risk is None else tuple(torch.as_tensor(risk).reshape(-1).float().tolist()),
                    risk_S_valid=None if risk_valid is None else tuple(
                        bool(v) for v in torch.as_tensor(risk_valid).reshape(-1).tolist()))
        self._action_trajectory(
            inputs_embeds, position_ids, noise, attention_mask, state_embeds, fast_embeds,
            10, 8, True, 'fp32', risk, risk_valid, _snapshot_sink=sink, _snapshot_meta=meta)
        return ActionCacheLadder(tuple(sink[i] for i in (4, 6, 8)))

    @torch.no_grad()
    def _action_trajectory(
        self,
        inputs_embeds: torch.Tensor, # Contesto latent [B,L_latent,H]
        position_ids: torch.Tensor, # # Informazioni posizionali del contesto 
        noise: torch.Tensor, # noise
        attention_mask: Optional[torch.Tensor] = None, # # Maschera del contesto
        state_embeds: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None,
        num_steps_total: int = 10,
        split_step: int = 6,
        refresh_clean_kv: bool = True,
        flow_time_precision: str = 'fp32',
        risk: Optional[torch.Tensor] = None,
        risk_valid: Optional[torch.Tensor] = None,
        *, _snapshot_sink=None, _snapshot_meta=None,
    ) -> Tuple[torch.Tensor, "DynamicCache", int, float]:
        """Cascaded slow-tick: run the action expert for `split_step` of
        `num_steps_total` Euler steps, stopping at τ = 1 − split_step/num_steps_total.

        Returns
        -------
        x_split           : [B, n_chunk, action_dim]  partially-denoised chunk
                                                       at τ = τ_split.
        cached_kv         : DynamicCache with [latent KV | action KV at τ_split].
        n_action_in_cache : int
        tau_split         : float                      ∈ (0, 1)
        """
        if not (0 < split_step < num_steps_total):
            raise ValueError(
                f"split_step must be in (0, num_steps_total); got "
                f"{split_step}/{num_steps_total}.")

        device = noise.device
        dtype  = torch.bfloat16
        if flow_time_precision not in ('fp32', 'legacy_bf16'):
            raise ValueError('invalid flow_time_precision')
        precise = flow_time_precision == 'fp32'
        # Same dt as the full flow, so the cascaded trajectory is consistent
        # with what a 10-step monolithic flow would integrate.
        dt     = torch.tensor(-1.0 / num_steps_total, dtype=torch.float32 if precise else dtype, device=device)
        x_t    = noise.float() if precise else noise.to(dtype)
        time   = torch.tensor(1.0, dtype=dtype, device=device)
        n_chunk = noise.shape[1]
        B      = inputs_embeds.shape[0]
        H      = inputs_embeds.shape[2]

        if fast_embeds is None:
            fast_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
        if state_embeds is None:
            state_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
        n_state = state_embeds.shape[1]
        L_latent = inputs_embeds.shape[1]
        # r_S token: computed once from the slow-tick measurements and baked into
        # every Action forward (and hence into the handover KV cache).
        risk_tok = self.risk_token(risk, risk_valid, 'action', B, device, dtype)
        n_risk = risk_tok.shape[1]

        def capture(step, cache, x_state, act_positions):
            _snapshot_sink[step] = ActionHandoverSnapshot.capture(
                x_fp32=x_state, cache=cache, position_ids=position_ids,
                action_position_ids=act_positions, handover=Handover(step, num_steps_total),
                n_fast=fast_embeds.shape[1], n_state=n_state, n_risk=n_risk,
                clone_cache=self._clone_dynamic_cache, **(_snapshot_meta or {}))

        past_kv = None
        n_act = 0
        # Per 6 steps
        for i in range(split_step):
            if precise:
                time = flow_time(i, num_steps_total, device=device)

            # A ogni step vengono creati nuovamente:
            # - Il token del tempo corrente.
            # - I 16 token delle azioni rumorose correnti.
            # - La sequenza action che contiene anche i token fast e l'eventuale stato.
            timesteps = self.t_embedder(time.expand(B)).unsqueeze(1)
            noisy_act = self.x_embedder(x_t.to(dtype))
            act_parts = [fast_embeds] # importante anche la presenza dei fast tokens
            if n_state > 0:
                act_parts.append(state_embeds)
            if n_risk > 0:
                act_parts.append(risk_tok)   # r_S, before the timestep token
            act_parts += [timesteps, noisy_act]
            act_seq = torch.cat(act_parts, dim=1)
            n_act = act_seq.shape[1]

            # Alla prima iterazione quando i = 0, past_kv è None
            if past_kv is None:
                full_embeds = torch.cat([inputs_embeds, act_seq], dim=1)
                outputs = self.model(
                    inputs_embeds=full_embeds, # full
                    position_ids=position_ids, # full
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True, # ora la cache viene create e salvata
                    latent_indexes=torch.arange(0, L_latent, device=device),
                    action_indexes=torch.arange(L_latent, L_latent + n_act, device=device),
                    tactile_indexes=torch.arange(0, 0, device=device),
                )
            else:
                # se in cache avevamo [KV_L | KV_A(1)]
                # il crop elimina la parte dell'action e lascia [KV_L]
                # così non perdiamo il contesto visivo-linguistico e non va ricalcolato
                # però il transformer deve ricalcolare i token action perché x_t e time sono cambiati
                past_kv.crop(-n_act)
                extended_pos = self.model._extend_position_ids(position_ids, n_act, 0)
                act_pos = extended_pos[..., -n_act:]
                outputs = self.model(
                    inputs_embeds=act_seq,
                    position_ids=act_pos, # Il contesto latent non è più presente tra i nuovi token.
                                          # È già memorizzato in past_kv.
                    past_key_values=past_kv,
                    use_cache=True,
                    latent_indexes=torch.arange(0, 0, device=device),
                    action_indexes=torch.arange(0, n_act, device=device),
                    tactile_indexes=torch.arange(0, 0, device=device),
                )
            # Ladder: this forward's KV is exactly the (x_tau, tau) refresh a
            # single-split run would do at this handover; snapshot BEFORE the
            # Euler update (no extra forward). Clone first: the next iteration crops.
            if _snapshot_sink is not None and i in (4, 6):
                capture(i, outputs.past_key_values, x_t,
                        self.model._extend_position_ids(position_ids, n_act, 0)[..., -n_act:])
            # hidden[:, -n_chunk:, :] seleziona gli hidden states dei 16 token action finali.
            hidden = outputs.last_hidden_state
            # la HEAD self.final_layer(...) li trasforma in v_A
            # !!! il tensore v_act non viene salvato come una sequenza di azioni da eseguire.
            # Viene utilizzato per aggiornare x_t.
            # È x_t, dopo il completamento del processo, a rappresentare il chunk generato.
            v_act  = self.final_layer(hidden[:, -n_chunk:, :])
            # viene aggiornato lo stato
            x_t    = x_t + dt * (v_act.float() if precise else v_act)
            # viene aggiornato il tempo
            if not precise:
                time = time + dt
            past_kv = outputs.past_key_values

        if precise:
            time = flow_time(split_step, num_steps_total, device=device)
        # `time` is now τ_split.  Refresh KV with the partially-denoised state
        # at τ_split so the tactile expert attends to a coherent action context.
        # Refresh per avere cache e stato coerenti
        if refresh_clean_kv and past_kv is not None:
            past_kv.crop(-n_act)
            split_timesteps = self.t_embedder(
                time.expand(B)
            ).unsqueeze(1)
            split_actions = self.x_embedder(x_t.to(dtype))
            clean_parts = [fast_embeds]
            if n_state > 0:
                clean_parts.append(state_embeds)
            if n_risk > 0:
                clean_parts.append(risk_tok)
            clean_parts += [split_timesteps, split_actions]
            clean_seq = torch.cat(clean_parts, dim=1)
            n_act_final = clean_seq.shape[1]
            extended_pos = self.model._extend_position_ids(position_ids, n_act_final, 0)
            act_pos_final = extended_pos[..., -n_act_final:]
            # NUOVO FORWARD
            # Questa volta il Transformer riceveE
            # (x_{0.4},\tau=0.4)
            # e aggiorna la cache.
            # Non viene calcolato un nuovo passo di Eulero.
            _ = self.model(
                inputs_embeds=clean_seq,
                position_ids=act_pos_final,
                past_key_values=past_kv,
                use_cache=True,
                latent_indexes=torch.arange(0, 0, device=device),
                action_indexes=torch.arange(0, n_act_final, device=device),
                tactile_indexes=torch.arange(0, 0, device=device),
            )
            n_action_in_cache = n_act_final
        else:
            n_action_in_cache = n_act

        tau_split = (num_steps_total - split_step) / num_steps_total if precise else float(time.item())
        # n_action_in_cache = L_fast + n_state + 1 + 16
        past_kv._trex_handover = {'action_steps': split_step, 'total_steps': num_steps_total,
            'tau': tau_split, 'precision': flow_time_precision, 'length': past_kv.get_seq_length(),
            'n_action': n_action_in_cache, 'n_risk': n_risk}
        if _snapshot_sink is not None:
            capture(split_step, past_kv, x_t, act_pos_final)
        return x_t.to(dtype), past_kv, n_action_in_cache, tau_split

    @torch.no_grad()
    def tactile_flow_continue(
        self,
        cached_kv,
        latent_position_ids: torch.Tensor,
        n_action_in_cache: int,
        x_split: torch.Tensor,                  # [B, n_chunk, action_dim] at τ=τ_split
        tau_split: float,
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        tactile_codes: Optional[torch.Tensor] = None,
        tactile_f6_history: Optional[torch.Tensor] = None,
        num_steps_total: int = 10,
        split_step: int = 6,
        flow_time_precision: str = 'fp32',
        risk: Optional[torch.Tensor] = None,
        risk_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cascaded fast-tick: continue the flow from x_split at τ=tau_split
        down to τ=0 using the tactile expert.  Returns the final clean action
        chunk (NOT a residual — the tactile expert here is a velocity predictor
        for the action distribution itself, restricted to τ ∈ [0, τ_split]).

        `cached_kv` is cloned so multiple fast ticks within one chunk window
        each start from the same slow-tick snapshot.
        """
        # !!! Il Tactile Expert non utilizza direttamente la cache originale:
        # ne crea una copia indipendente. Perché?
        # Perché, durante i quattro step di denoising, il codice modifica la cache con
        # operazioni di inserimento e rimozione delle KV tattile.
        # Se modificasse direttamente self.cached_kv, non potremmo riutilizzare lo stesso
        # snapshot al tick veloce successivo. La clonazione permette quindi di preservare
        # il contesto prodotto dall'Action Expert.
        expected_tau = (num_steps_total - split_step) / num_steps_total
        if flow_time_precision == 'fp32' and abs(tau_split - expected_tau) > 1e-6:
            raise ValueError('tactile handover time does not match action schedule')
        metadata = getattr(cached_kv, '_trex_handover', None)
        if metadata is not None:
            if (metadata['action_steps'] != split_step or metadata['total_steps'] != num_steps_total
                    or metadata['precision'] != flow_time_precision or metadata['n_action'] != n_action_in_cache
                    or abs(metadata['tau'] - tau_split) > 1e-6):
                raise ValueError('inference handover cache metadata mismatch')
            if cached_kv.get_seq_length() != metadata['length']:
                raise ValueError('inference handover cache has been mutated')
            if metadata.get('n_risk', 0) != self.n_risk_tokens:
                raise ValueError('handover cache risk-token layout does not match this model')
        cache = self._clone_dynamic_cache(cached_kv)
        device = x_split.device
        dtype  = torch.bfloat16
        if flow_time_precision not in ('fp32', 'legacy_bf16'):
            raise ValueError('invalid flow_time_precision')
        precise = flow_time_precision == 'fp32'
        B      = x_split.shape[0]
        n_chunk, action_dim = x_split.shape[1], x_split.shape[2]

        tac_obs = self._embed_tactile_observations(
            tactile_f6, tactile_deform, device, dtype,
            tactile_codes=tactile_codes,
            tactile_f6_history=tactile_f6_history)
        n_obs = tac_obs.shape[1]
        # Fresh r_F token for THIS fast tick; the cached r_S is untouched.
        risk_tok = self.risk_token(risk, risk_valid, 'tactile', B, device, dtype)
        n_risk = risk_tok.shape[1]
        n_tac_seq = n_obs + n_risk + 1 + n_chunk

        extended_pos = self.model._extend_position_ids(
            latent_position_ids, n_action_in_cache, n_tac_seq)
        tac_pos = extended_pos[..., -n_tac_seq:]

        # Same dt as the action expert used in the slow tick — the cascaded
        # trajectory's integration step matches what a monolithic 10-step flow
        # would use.  `remaining` steps complete the integration.
        dt   = torch.tensor(-1.0 / num_steps_total, dtype=torch.float32 if precise else dtype, device=device)
        time = torch.tensor(tau_split, dtype=dtype, device=device)
        remaining = num_steps_total - split_step
        x = x_split.float() if precise else x_split.to(dtype)

        for step in range(remaining):
            if precise:
                time = flow_time(split_step + step, num_steps_total, device=device)
            # Ad ogni iterazione
            # x_{tau-0.1} = x_tau -0.1 v_T(x_tau, tau, F, C), tau -> tau - 0.1


            tau_emb = self.t_embedder(time.expand(B)).unsqueeze(1)
            x_emb   = self.x_embedder(x.to(dtype))
            full_embeds = torch.cat([tac_obs] + ([risk_tok] if n_risk else []) + [tau_emb, x_emb], dim=1)
            if step > 0:
                # Al primo step, la cache clonata contiene soltanto il contesto latent e action.
                # Dopo il primo forward vengono aggiunte le KV dei token tattili.
                cache.crop(-n_tac_seq)

            outputs = self.model(
                inputs_embeds=full_embeds,
                position_ids=tac_pos,
                past_key_values=cache,
                use_cache=True,
                latent_indexes=torch.arange(0, 0, device=device),
                action_indexes=torch.arange(0, 0, device=device),
                tactile_indexes=torch.arange(0, n_tac_seq, device=device),
            )
            hidden = outputs.last_hidden_state
            v = self.final_layer_tactile(hidden[:, -n_chunk:, :])
            x    = x + dt * (v.float() if precise else v)
            if not precise:
                time = time + dt
            
        # Questo x è il chunk denoisato a tau=0, ancora nello spazio normalizzato delle azioni.
        return x.to(dtype)

    @torch.no_grad()
    def tactile_flow_continue_from_ladder(
        self, ladder: ActionCacheLadder, selected_split: float,
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        tactile_codes: Optional[torch.Tensor] = None,
        tactile_f6_history: Optional[torch.Tensor] = None,
        risk: Optional[torch.Tensor] = None,
        risk_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One physical fast tick: restart the selected immutable snapshot with
        the current tactile inputs and fresh r_F, integrating to tau=0. Calls never
        consume or modify ladder levels (the continuation clones the cache)."""
        if self.training:
            raise ValueError('cache ladder requires model.eval()')
        if not isinstance(ladder, ActionCacheLadder):
            raise TypeError('expected ActionCacheLadder')
        snap = ladder[selected_split]
        if snap.n_risk != self.n_risk_tokens:
            raise ValueError('ladder risk-token layout does not match this model')
        return self.tactile_flow_continue(
            snap._cache, snap._latent_positions, snap.n_action_in_cache, snap._x_tau, snap.tau,
            tactile_f6=tactile_f6, tactile_deform=tactile_deform, tactile_codes=tactile_codes,
            tactile_f6_history=tactile_f6_history, num_steps_total=snap.handover.total_steps,
            split_step=snap.handover.action_steps, flow_time_precision='fp32',
            risk=risk, risk_valid=risk_valid)


    # Quindi tactile_flow_train_step non restituisce direttamente un chunk pulito.
    # Restituisce la velocità predetta dal Tactile Expert nel punto di flow campionato.
    # L'aggiornamento con Eulero si farà durante l'inferenza, nella funzione tactile_flow_continue
    def tactile_flow_train_step(
        self,
        cached_kv,
        latent_position_ids: torch.Tensor,
        n_action_in_cache: int,
        x_tau: torch.Tensor,                    # [B, n_chunk, action_dim], full action state
        tau: torch.Tensor,                      # [B] flow times in [0, tau_split]
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        tactile_codes: Optional[torch.Tensor] = None,
        tactile_f6_history: Optional[torch.Tensor] = None,
        flow_time_precision: str = 'fp32',
        risk: Optional[torch.Tensor] = None,
        risk_valid: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Single tactile-only forward at (x_τ, τ) for cascaded training.

        return_hidden=True also returns the Tactile expert's last hidden states of the
        chunk tokens [B, n_chunk, H] (input of the future-risk head).

        Returns predicted velocity v_pred [B, n_chunk, action_dim] suitable for
        L_flow_tactile = MSE(v_pred, ε − A_demo).  Same target as the action
        expert's L_flow loss — both experts are trained as velocity predictors
        for the same flow trajectory, just on disjoint τ ranges.
        """
        metadata = getattr(cached_kv, '_trex_handover', None)
        if metadata is not None:
            if metadata['precision'] != flow_time_precision or metadata['n_action'] != n_action_in_cache:
                raise ValueError('training handover cache metadata mismatch')
            if cached_kv.get_seq_length() != metadata['length']:
                raise ValueError('training handover cache has been mutated')
            limit = metadata['tau'] if flow_time_precision == 'fp32' else (metadata['total_steps']-metadata['action_steps'])/metadata['total_steps']
            tolerance = 1e-6 if flow_time_precision == 'fp32' else 0.002
            if torch.any(tau <= 0) or torch.any(tau.float() > limit + tolerance):
                raise ValueError('tactile training time outside selected handover')
            if metadata.get('n_risk', 0) != self.n_risk_tokens:
                raise ValueError('training handover cache risk-token layout does not match this model')
        cached_kv = self._clone_dynamic_cache(cached_kv)
        device = x_tau.device
        dtype  = torch.bfloat16
        B      = x_tau.shape[0]
        n_chunk = x_tau.shape[1]

        # Costruzione dei token tattili
        # Questa funzione converte gli input tattili in una sequenza di embedding.
        tac_obs = self._embed_tactile_observations(
            tactile_f6, tactile_deform, device, dtype,
            tactile_codes=tactile_codes,
            tactile_f6_history=tactile_f6_history)
        n_obs = tac_obs.shape[1]

        # IMPORTANTE: il Tactile Expert utilizza gli stessi moduli t_embedder e x_embedder dell'Action Expert.
        # Non sono presenti due copie separate di questi embedder nella funzione.
        # Questa condivisione ha conseguenze sui gradienti, come abbiamo visto studiando il training.
        tau_emb = self.t_embedder(tau.to(dtype) if flow_time_precision == 'legacy_bf16' else tau.float()).unsqueeze(1)            # [B, 1, H]
        x_emb   = self.x_embedder(x_tau.to(dtype))                       # [B, 16, H]

        # r_F token (fresh fast-tick risk), trained jointly with the tactile expert.
        risk_tok = self.risk_token(risk, risk_valid, 'tactile', B, device, dtype)

        # Tactile obs (#n_obs), [risk_F (#1)], tau(#1), noisy action(#16)
        full_embeds = torch.cat([tac_obs] + ([risk_tok] if risk_tok.shape[1] else [])
                                + [tau_emb, x_emb], dim=1) # concatenazione
        n_tac_seq = full_embeds.shape[1]

        # Il codice costruisce gli indici posizionali:
        extended_pos = self.model._extend_position_ids(
            latent_position_ids, n_action_in_cache, n_tac_seq)
        # La cache contiene già i token latent e action elaborati precedentemente.
        # I nuovi token tattili devono occupare posizioni successive a quelle della cache.
        tac_pos = extended_pos[..., -n_tac_seq:]

        outputs = self.model(
            inputs_embeds=full_embeds,
            position_ids=tac_pos,
            past_key_values=cached_kv,
            use_cache=True,
            latent_indexes=torch.arange(0, 0, device=device), # indici latent vuoti
            action_indexes=torch.arange(0, 0, device=device), # indici action vuoti
            tactile_indexes=torch.arange(0, n_tac_seq, device=device), # tutti i nuovi token sono assegnati al tactile expert
        ) 

        hidden = outputs.last_hidden_state
        # Questa istruzione estrae gli ultimi 16 token, corrispondenti alle azioni rumorose, e li passa alla prediction head tattile.
        v_pred = self.final_layer_tactile(hidden[:, -n_chunk:, :])
        if return_hidden:
            return v_pred, hidden[:, -n_chunk:, :]
        return v_pred

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)


def split_slow_fast_embeds(
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    image_token_id: int,
    n_slow_img_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split inputs_embeds into slow (latent) and fast (action) portions.

    Splits at the <|vision_start|> of the first fast image: everything
    before it is slow (contiguous prefix), everything from it onward is
    fast (contiguous suffix).  This preserves token order — critical
    because the MoT routes tokens by contiguous index ranges and
    position_ids are computed for the original sequence order.

    Fast embeds include the vision brackets, image patches, AND any
    trailing tokens (``<|im_end|>``, generation prompt) that follow the
    fast images in the original sequence.

    Parameters
    ----------
    inputs_embeds     : [B, L, H]
    input_ids         : [B, L]
    image_token_id    : int — the <|image_pad|> token id (e.g. 151655)
    n_slow_img_tokens : int — total <|image_pad|> count for slow images

    Returns
    -------
    slow_embeds : [B, L_slow, H]  — contiguous prefix (text + slow image)
    fast_embeds : [B, L_fast, H]  — contiguous suffix (fast images + trailing)
    """

    # Il codice ricerca il primo token visivo appartenente alle immagini fast e determina
    # il punto in cui dividere la sequenza.

    #Il suffisso fast comprende anche i token di delimitazione dell'immagine e gli eventuali
    # token successivi previsti dal prompt.

    B = inputs_embeds.shape[0]
    img_pad_mask = (input_ids == image_token_id)             # [B, L]
    img_cumcount = img_pad_mask.long().cumsum(dim=1)         # [B, L]
    fast_pad_mask = img_pad_mask & (img_cumcount > n_slow_img_tokens)

    n_fast_pads = int(fast_pad_mask[0].sum().item())
    if n_fast_pads == 0:
        return inputs_embeds, inputs_embeds[:, :0]

    # Split at <|vision_start|> of the first fast image (one position
    # before the first fast <|image_pad|> token).
    # All samples have the same fast content length (same image sizes +
    # same trailing tokens), so split_pos is identical across the batch.
    fast_pad_pos = fast_pad_mask[0].nonzero(as_tuple=True)[0]
    split_pos = int(fast_pad_pos[0].item()) - 1  # <|vision_start|>

    # slow_embeds.shape  # [B, L_slow, H]
    # fast_embeds.shape  # [B, L_fast, H]
    slow_embeds = inputs_embeds[:, :split_pos]
    fast_embeds = inputs_embeds[:, split_pos:]
    return slow_embeds, fast_embeds


def extend_position_ids_for_flare(
    pos_ids: torch.Tensor,
    n_flare: int,
) -> torch.Tensor:
    """
    Extend M-RoPE position_ids by n_flare sequential positions.

    Parameters
    ----------
    pos_ids : [3, B, L_slow]  M-RoPE positions for slow tokens
    n_flare : int  number of flare query tokens

    Returns
    -------
    [3, B, L_slow + n_flare]  extended positions
    """
    if n_flare == 0:
        return pos_ids
    device = pos_ids.device
    max_pos = pos_ids.max(dim=-1, keepdim=True)[0]  # [3, B, 1]
    offsets = torch.arange(1, n_flare + 1, device=device).view(1, 1, n_flare)
    flare_pos = max_pos + offsets  # [3, B, n_flare]
    return torch.cat([pos_ids, flare_pos], dim=-1)
