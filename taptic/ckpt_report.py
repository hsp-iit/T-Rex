"""Parameter-loading report for resuming T-REX midtraining from a checkpoint.

Reproduces the official ``scripts/midtrain.py`` resume semantics exactly:
  1. every checkpoint tensor whose name exists in the model with the SAME shape
     is loaded; same-name tensors with a different shape are skipped (they keep
     the fresh initialisation from ``initialize_vla_weights``);
  2. model tensors absent from the checkpoint keep their initialisation;
     checkpoint tensors absent from the model are ignored;
  3. afterwards, with ``resume_source == "pretrain"`` and tactile enabled, every
     parameter whose name contains ``_tactile`` is re-initialised (xavier /
     zeros), even if it was loaded in step 1.
This module only classifies; it never changes weights.

Offline use (no weights needed, model built on the meta device and the
checkpoint index read remotely with taptic.remote_ckpt_index):
    python -m taptic.ckpt_report --ckpt_index model_pt_index.json \
        --qwen_config config.json --vqvae_ckpt latest.pt --out report.json
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple


def _group(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "model" and len(parts) > 3 and parts[1] == "layers":
        sub = parts[3]
        expert = "tactile" if "_tactile" in name else "action" if "_action" in name else "latent"
        return f"model.layers.*.{sub}[{expert}]"
    if parts[0] == "model":
        return f"model.{parts[1]}"
    return parts[0]


def classify(model_shapes: Dict[str, Tuple[int, ...]], ckpt_shapes: Dict[str, Tuple[int, ...]],
             frozen_prefixes: Sequence[str] = ("visual", "tactile_vqvae", "deform_encoder"),
             reinit_tactile: bool = True, vqvae_loaded_separately: bool = True) -> Dict:
    loaded, mismatch, missing, unexpected, reinit = [], [], [], [], []
    for k, shp in model_shapes.items():
        if k in ckpt_shapes:
            if tuple(ckpt_shapes[k]) == tuple(shp):
                loaded.append(k)
            else:
                mismatch.append((k, list(ckpt_shapes[k]), list(shp)))
        else:
            missing.append(k)
    unexpected = [k for k in ckpt_shapes if k not in model_shapes]
    if reinit_tactile:
        reinit = [k for k in model_shapes if "_tactile" in k]
    reinit_set = set(reinit)

    def numel(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n

    status = OrderedDict()
    for k, shp in model_shapes.items():
        if k in reinit_set:
            st = "reinit_tactile_expert"
        elif k in ckpt_shapes and tuple(ckpt_shapes[k]) == tuple(shp):
            st = "loaded"
        elif k in ckpt_shapes:
            st = "shape_mismatch_reinit"
        elif k.startswith("tactile_vqvae") or k.startswith("tacf6_vqvae"):
            st = "from_vqvae_ckpt" if vqvae_loaded_separately else "missing_init"
        else:
            st = "missing_init"
        status[k] = st

    groups: Dict[str, Dict[str, int]] = OrderedDict()
    for k, st in status.items():
        g = groups.setdefault(_group(k), {})
        g[st] = g.get(st, 0) + numel(model_shapes[k])
    totals: Dict[str, int] = {}
    for k, st in status.items():
        totals[st] = totals.get(st, 0) + numel(model_shapes[k])
    trainable = {k: not any(k.startswith(p) for p in frozen_prefixes) for k in model_shapes}
    return {
        "rules": "official midtrain.py: shape-filtered strict=False load, then _tactile re-init "
                 "(resume_source=pretrain)",
        "n_model_tensors": len(model_shapes), "n_ckpt_tensors": len(ckpt_shapes),
        "params_by_status": totals,
        "shape_mismatch": [{"name": k, "ckpt": c, "model": m} for k, c, m in mismatch],
        "missing_in_ckpt": missing,
        "unexpected_in_ckpt": unexpected,
        "loaded_then_reinitialized": sorted(k for k in reinit if k in ckpt_shapes),
        "groups": groups,
        "frozen": sorted(k for k, t in trainable.items() if not t),
        "status": status,
    }


def summarize(rep: Dict) -> List[str]:
    out = [f"model tensors={rep['n_model_tensors']}  ckpt tensors={rep['n_ckpt_tensors']}"]
    for st, n in rep["params_by_status"].items():
        out.append(f"  {st:24s} {n / 1e6:10.2f} M params")
    out.append(f"  shape mismatches ({len(rep['shape_mismatch'])}):")
    out += [f"    {m['name']}: ckpt {m['ckpt']} -> model {m['model']}" for m in rep["shape_mismatch"]]
    miss = [k for k in rep["missing_in_ckpt"] if not k.startswith(("tactile_vqvae", "tacf6_vqvae"))]
    out.append(f"  missing in ckpt (fresh init, excl. VQ-VAE): {miss}")
    out.append(f"  unexpected in ckpt (ignored): {rep['unexpected_in_ckpt']}")
    return out


def _meta_model_shapes(qwen_config: str, vqvae_ckpt: str, action_dim: int, action_chunk: int,
                       tacf6_dim: int, n_tactile_fingers: int, tactile_intermediate_size: int,
                       n_flare_tokens_per_frame: int, n_flare_steps: int) -> Dict[str, Tuple[int, ...]]:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from qwen_vla.modeling_vla import Qwen3VLVLAModel
    cfg = AutoConfig.from_pretrained(qwen_config)
    blob = torch.load(vqvae_ckpt, map_location="cpu", weights_only=False)
    with torch.device("meta"):
        m = Qwen3VLVLAModel(cfg.text_config, action_dim=action_dim, action_chunk=action_chunk,
                            tacf6_dim=tacf6_dim, use_tactile_deform=False, use_robot_state=False,
                            tactile_intermediate_size=tactile_intermediate_size or None,
                            n_flare_tokens_per_frame=n_flare_tokens_per_frame, n_flare_steps=n_flare_steps,
                            use_tactile_vqvae=True, vqvae_config=blob["config"],
                            n_tactile_fingers=n_tactile_fingers)
        vis = Qwen3VLVisionModel._from_config(cfg.vision_config)
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    shapes.update({f"visual.{k}": tuple(v.shape) for k, v in vis.state_dict().items()})
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_index", required=True, help="JSON {name: {shape, dtype}}")
    ap.add_argument("--qwen_config", required=True)
    ap.add_argument("--vqvae_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action_dim", type=int, default=21)
    ap.add_argument("--action_chunk", type=int, default=16)
    ap.add_argument("--tacf6_dim", type=int, default=3)
    ap.add_argument("--n_tactile_fingers", type=int, default=5)
    ap.add_argument("--tactile_intermediate_size", type=int, default=1536)
    ap.add_argument("--n_flare_tokens_per_frame", type=int, default=4)
    ap.add_argument("--n_flare_steps", type=int, default=8)
    a = ap.parse_args()
    with open(a.ckpt_index) as f:
        idx = {k: tuple(v["shape"]) for k, v in json.load(f).items()}
    shapes = _meta_model_shapes(a.qwen_config, a.vqvae_ckpt, a.action_dim, a.action_chunk, a.tacf6_dim,
                                a.n_tactile_fingers, a.tactile_intermediate_size,
                                a.n_flare_tokens_per_frame, a.n_flare_steps)
    rep = classify(shapes, idx)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("\n".join(summarize(rep)))
    print("groups:")
    for g, d in rep["groups"].items():
        print(f"  {g:45s} " + ", ".join(f"{k}={v / 1e6:.2f}M" for k, v in d.items()))


if __name__ == "__main__":
    main()
