"""Test E — TRexTapticDataset: a real batch through a multi-worker DataLoader."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from qwen_vla.taptic_dataset import TRexTapticDataset


def _ds(train_cfg, processor, accel):
    return TRexTapticDataset(train_cfg, processor, accel, split="train")


def test_batch_contract(train_cfg, processor, accel):
    ds = _ds(train_cfg, processor, accel)
    B = 3
    dl = DataLoader(ds, batch_size=B, shuffle=True, num_workers=2, collate_fn=ds.collate_fn,
                    generator=torch.Generator().manual_seed(0))
    b = next(iter(dl))
    shapes = {k: tuple(v.shape) for k, v in b.items() if torch.is_tensor(v)}
    print("\n".join(f"{k:22s} {s} {b[k].dtype}" for k, s in shapes.items()))
    assert shapes["noisy_actions"] == shapes["target"] == shapes["norm_actions"] == (B, 16, 21)
    assert shapes["timesteps"] == (B,) and shapes["time_r"] == (B,) and shapes["eps_r"] == (B, 16, 21)
    assert shapes["tactile_f6s"] == shapes["tactile_f6s_delayed"] == (B, 5, 3)
    assert shapes["tactile_f6_history"] == (B, 16, 5, 3) and b["tactile_f6_history"].dtype == torch.float32
    assert b["state_raw"] is None and b["tactile_deforms"] is None and b["tactile_codes"] is None
    assert shapes["image_grid_thw"] == (2 * B, 3)                       # 1 slow + 1 fast image per sample
    assert shapes["flare_grid_thw"] == (8 * B, 3)                       # 8 future head frames
    assert b["n_slow_images"] == 1
    for k in ("noisy_actions", "target", "norm_actions", "tactile_f6s", "tactile_f6s_delayed", "pixel_values"):
        assert torch.isfinite(b[k].float()).all(), k
    assert b["norm_actions"].abs().max() <= 1 and b["tactile_f6s"].abs().max() <= 1
    # flow-matching identities (official): x_t = t*eps + (1-t)*a, u = eps - a
    t = b["timesteps"].float()[:, None, None]
    eps = b["target"].float() + b["norm_actions"].float()
    assert torch.allclose(b["noisy_actions"].float(), t * eps + (1 - t) * b["norm_actions"].float(), atol=3e-2)
    # the primitive instruction is in the prompt, between the slow and the fast image
    txt = processor.tokenizer.decode(b["input_ids"][0][b["attention_mask"][0].bool()])
    m = b["meta"][0]
    instr = ds._rows[ds._rows.sample_idx == m["sample_idx"]].instruction.iloc[0]
    assert instr in txt and txt.count("<|vision_start|>") == 2
    assert txt.index(instr) > txt.index("<|vision_start|>") and txt.rindex("<|vision_start|>") > txt.index(instr)
    assert all(0 <= x["tactile_age_ms"] < 600 for x in b["meta"])


def test_items_are_causal_and_in_primitive(train_cfg, processor, accel):
    ds = _ds(train_cfg, processor, accel)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(ds), 40, replace=False):
        it = ds[int(i)]
        r = ds._rows.iloc[int(i)]
        assert it["action_chunk"].shape == (16, 21)
        assert r.frame + it["delay_k"] < r.seg_end
        assert it["head"].shape == (480, 640, 3) and len(it["fast"]) == 1
        assert len(it["flare_frames"]) == 8


def test_val_split_from_converter(train_cfg, processor, accel):
    ds = _ds(train_cfg, processor, accel)
    val = ds.create_val_split(0.5, 123)                                  # args ignored: fixed split
    assert set(val._rows.episode).isdisjoint(set(ds._rows.episode))
    assert int(val._cum_frames[-1]) == len(val) and int(ds._cum_frames[-1]) == len(ds)
