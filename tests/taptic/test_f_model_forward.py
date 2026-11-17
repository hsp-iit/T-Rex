"""Test F — forward passes through the Action Expert and Tactile Expert (cascaded),
with the real Qwen3-VL/MoT code at reduced size (see tiny_qwen3vl.py)."""
import pytest
import torch

from qwen_vla import extend_position_ids_for_flare, split_slow_fast_embeds
from qwen_vla.modeling_vla import Qwen3VLVLAModel
from qwen_vla.taptic_dataset import TRexTapticDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def setup(tiny_qwen, vq_ckpt, train_cfg, processor, accel):
    blob = torch.load(vq_ckpt, map_location="cpu", weights_only=False)
    m = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        tiny_qwen, action_dim=21, action_chunk=16, tacf6_dim=3, n_tactile_fingers=5,
        tactile_intermediate_size=64, n_flare_tokens_per_frame=4, n_flare_steps=8,
        use_tactile_vqvae=True, vqvae_config=blob["config"])
    m.initialize_vla_weights(); m.load_tactile_vqvae(blob)
    for n, p in m.named_parameters():               # official midtrain tactile re-init (zeros norms)
        if "_tactile" in n:
            torch.nn.init.xavier_uniform_(p) if p.ndim >= 2 else torch.nn.init.zeros_(p)
    m = m.to(DEV).float().eval()
    ds = TRexTapticDataset(train_cfg, processor, accel)
    b = ds.collate_fn([ds[i] for i in (0, 4000, 9000)])
    b = {k: v.to(DEV) if torch.is_tensor(v) else v for k, v in b.items()}
    return m, b


def _latent(m, b):
    emb = m.prepare_inputs_embeds(b["input_ids"], b["pixel_values"], b["image_grid_thw"])
    B = emb.shape[0]
    g = b["image_grid_thw"]
    n_slow_tok = int(g[0, 0] * (g[0, 1] // 2) * (g[0, 2] // 2))
    slow, fast = split_slow_fast_embeds(emb, b["input_ids"], m.image_token_id, n_slow_tok)
    pos, _ = m.get_rope_index(b["input_ids"], b["image_grid_thw"], b["attention_mask"])
    pos = pos[:, :, :slow.shape[1]]
    K = m.n_flare_tokens
    slow_ext = torch.cat([slow, m.flare_queries.expand(B, -1, -1).to(slow.dtype)], 1)
    return slow_ext, fast, extend_position_ids_for_flare(pos, K)


@torch.no_grad()
def test_action_expert_forward(setup):
    m, b = setup
    with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        slow, fast, pos = _latent(m, b)
        assert fast.shape[1] > 0                                          # wrist image is the fast stream
        ts = m.t_embedder(b["timesteps"].float()).unsqueeze(1)
        na = m.x_embedder(b["noisy_actions"].float())
        full = torch.cat([slow, fast, ts, na], 1)
        L = slow.shape[1]
        out = m.model(inputs_embeds=full, position_ids=pos, attention_mask=b["attention_mask"], use_cache=False,
                      latent_indexes=torch.arange(0, L, device=DEV),
                      action_indexes=torch.arange(L, full.shape[1], device=DEV),
                      tactile_indexes=torch.arange(0, 0, device=DEV))
        start = L + fast.shape[1] + 1
        v = m.final_layer(out.last_hidden_state[:, start:start + 16])
    assert v.shape == (3, 16, 21) and torch.isfinite(v).all()


@torch.no_grad()
def test_cascaded_tactile_expert(setup):
    m, b = setup
    with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        slow, fast, pos = _latent(m, b)
        x_split, kv, n_act, tau_split = m.forward_flow_action_partial(
            slow, pos, torch.randn(3, 16, 21, device=DEV), attention_mask=b["attention_mask"],
            fast_embeds=fast, num_steps_total=10, split_step=6, refresh_clean_kv=True)
        assert abs(tau_split - 0.4) < 1e-2 and x_split.shape == (3, 16, 21)
        assert n_act == fast.shape[1] + 1 + 16
        L_cache = kv.get_seq_length()
        tau = b["time_r"].float() * 0.4
        x_tau = (1 - tau[:, None, None]) * b["norm_actions"].float() + tau[:, None, None] * b["eps_r"].float()
        v = m.tactile_flow_train_step(kv, pos, n_act, x_tau, tau, tactile_f6=b["tactile_f6s_delayed"],
                                      tactile_f6_history=b["tactile_f6_history"])
        assert v.shape == (3, 16, 21) and torch.isfinite(v).all()
        assert kv.get_seq_length() == L_cache + (1 + 5) + 1 + 16           # [code, 5 tips, tau, 16 actions]
        # inference: slow tick once, two fast ticks reuse the same cached KV
        x_split, kv, n_act, tau_split = m.forward_flow_action_partial(
            slow, pos, torch.randn(3, 16, 21, device=DEV), attention_mask=b["attention_mask"], fast_embeds=fast)
        L0 = kv.get_seq_length()
        a1 = m.tactile_flow_continue(kv, pos, n_act, x_split, tau_split, tactile_f6=b["tactile_f6s"],
                                     tactile_f6_history=b["tactile_f6_history"])
        a2 = m.tactile_flow_continue(kv, pos, n_act, x_split, tau_split, tactile_f6=b["tactile_f6s_delayed"],
                                     tactile_f6_history=b["tactile_f6_history"])
        assert kv.get_seq_length() == L0                                    # snapshot untouched (cloned)
        assert a1.shape == a2.shape == (3, 16, 21) and torch.isfinite(a1).all()
        full = m.forward_flow_action_full(slow, pos, torch.randn(3, 16, 21, device=DEV),
                                          attention_mask=b["attention_mask"], fast_embeds=fast)
        assert full.shape == (3, 16, 21)
