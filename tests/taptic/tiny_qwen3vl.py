"""Build a tiny, randomly initialised Qwen3-VL checkpoint with the REAL
architecture classes and the REAL T-REX processor/tokenizer.

Used by Tests F/G: the 2B backbone + action + tactile experts (~5 B params in
bf16, 10 GB) does not fit the 8 GB development GPU, so model-forward and
training-step tests exercise every code path (MoT routing, KV cache, cascaded
flow, VQ-VAE embedding, FLARE) at reduced width/depth.  Only sizes change:
text hidden 128, 2 layers, vision hidden 64, 2 blocks.
"""
from __future__ import annotations

import os
import shutil

TREX_META = os.path.expanduser("~/midtrain_merge/derived/trex_pretrain_meta/checkpoint-0-610000")


def build(out_dir: str, seed: int = 0) -> str:
    import torch
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    if os.path.isfile(os.path.join(out_dir, "model.safetensors")):
        return out_dir
    cfg = AutoConfig.from_pretrained(os.path.join(TREX_META, "config.json"))
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.rope_scaling = {"rope_type": "default", "mrope_section": [6, 5, 5], "mrope_interleaved": True}
    v.hidden_size, v.intermediate_size, v.depth, v.num_heads = 64, 128, 2, 2
    v.out_hidden_size = t.hidden_size
    v.deepstack_visual_indexes = [0]
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    for f in os.listdir(os.path.join(TREX_META, "processor")):
        shutil.copy(os.path.join(TREX_META, "processor", f), out_dir)
    return out_dir


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1]))
