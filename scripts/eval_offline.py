"""Offline open-loop evaluation of a T-Rex checkpoint on held-out LeRobot episodes.

For a set of frames from the validation episodes (the same episode split that
pretrain.py uses: --val_ratio / --seed), the model samples a full action chunk
with the action expert only (forward_flow_action_full, as test.py does with
--disable_tactile 1). Predictions and ground truth are denormalized with the
checkpoint's stats_data.json and compared in physical units, per horizon step:

  * pos  - wrist translation error [cm]   (delta-base xyz, metres in the data)
  * rot  - wrist rotation error   [deg]   (geodesic angle between rot6d poses)
  * hand - mean abs hand-joint error [deg]

Two reference baselines are scored the same way:
  * hold - no arm motion (identity delta pose), hand stays at its current joints
  * mean - the mean ground-truth chunk over the evaluated frames

Usage (single GPU):
  python scripts/eval_offline.py \
      --checkpoint_path trex_runs/.../checkpoint-0-3388 \
      --lerobot_root /home/iit.local/lgori/egomix_100gb/retarget

Outputs (in --out_dir, default <checkpoint>/offline_eval):
  metrics.json, summary.md, predictions.npz, horizon_errors.png, examples.png
"""
import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds
from qwen_vla.lerobot_dataset import TRexLeRobotDataset, add_lerobot_args


class _PrintAccelerator:
    """TRexLeRobotDataset only needs accelerator.print."""
    def print(self, *a, **k):
        print(*a, **k, flush=True)


# ── geometry ────────────────────────────────────────────────────────────────
def rot6d_to_mat(r6):
    """[..., 6] (first two columns of R) -> [..., 3, 3] via Gram-Schmidt."""
    a1, a2 = r6[..., :3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True).clip(1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def chunk_errors(pred, gt, n_arms, hand_dim):
    """pred, gt [N, K, D] physical units -> dict of [N, K] errors."""
    pos, rot, hand = [], [], []
    w = 9 + hand_dim
    for a in range(n_arms):
        p, g = pred[..., a * w:(a + 1) * w], gt[..., a * w:(a + 1) * w]
        pos.append(np.linalg.norm(p[..., :3] - g[..., :3], axis=-1) * 100.0)
        Rp, Rg = rot6d_to_mat(p[..., 3:9]), rot6d_to_mat(g[..., 3:9])
        cos = (np.einsum("...ij,...ij->...", Rp, Rg) - 1.0) / 2.0
        rot.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        hand.append(np.degrees(np.abs(p[..., 9:] - g[..., 9:]).mean(-1)))
    return {"pos_cm": np.mean(pos, 0), "rot_deg": np.mean(rot, 0), "hand_deg": np.mean(hand, 0)}


def normalize(x, mask, lo, hi):
    return np.where(mask, np.clip(2 * (x - lo) / (hi - lo + 1e-8) - 1, -1, 1), x)


def denormalize(x, mask, lo, hi):
    return np.where(mask, 0.5 * (x + 1) * (hi - lo) + lo, x)


# ── model ───────────────────────────────────────────────────────────────────
def load_model(args, device):
    ckpt = args.checkpoint_path
    with open(os.path.join(ckpt, "training_args.json")) as f:
        ta = json.load(f)
    base = args.base_model_path or ta["model_path"]
    n_tpf = int(ta.get("n_flare_tokens_per_frame", 0)) if ta.get("use_flare", 0) else 0
    n_steps = int(ta.get("n_flare_steps", 0)) if ta.get("use_flare", 0) else 0
    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        pretrained_path=base,
        action_dim=int(ta.get("action_dim", args.action_dim)),
        action_chunk=int(ta.get("action_chunk", args.action_chunk)),
        use_robot_state=bool(ta.get("use_robot_state", args.use_robot_state)),
        torch_dtype=torch.bfloat16,
        n_flare_tokens_per_frame=n_tpf,
        n_flare_steps=n_steps,
    )
    sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        print(f"  missing: {missing[:10]}\n  unexpected: {unexpected[:10]}", flush=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(os.path.join(ckpt, "processor"), trust_remote_code=True)
    return model, processor, ta


@torch.no_grad()
def predict_chunk(model, batch, device, n_noise, num_steps, action_chunk, action_dim, generator):
    """One observation (B=1) -> [n_noise, chunk, D] normalized action chunks."""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    thw = batch["image_grid_thw"].to(device)

    emb = model.prepare_inputs_embeds(input_ids=input_ids, pixel_values=pv, image_grid_thw=thw)
    n_slow = int(batch["n_slow_images"])
    if thw.shape[0] > n_slow:
        merge = getattr(model.visual, "spatial_merge_size", 2)
        n_slow_tok = sum(int(g[0] * (g[1] // merge) * (g[2] // merge)) for g in thw[:n_slow])
        slow, fast = split_slow_fast_embeds(emb, input_ids, model.image_token_id, n_slow_tok)
    else:
        slow, fast = emb, None

    pos, _ = model.get_rope_index(input_ids=input_ids, image_grid_thw=thw, attention_mask=attn)
    pos = pos[:, :, :slow.shape[1]]
    if model.n_flare_tokens > 0:
        fq = model.flare_queries.to(device=slow.device, dtype=slow.dtype)
        slow = torch.cat([slow, fq.expand(slow.shape[0], -1, -1)], dim=1)
        pos = extend_position_ids_for_flare(pos, model.n_flare_tokens)

    state = None
    if batch.get("state_raw") is not None and getattr(model, "state_embedder", None) is not None:
        state = model.state_embedder(batch["state_raw"].to(device, dtype=torch.bfloat16)).unsqueeze(1)

    n = n_noise
    rep = lambda t: None if t is None else t.expand(n, *t.shape[1:]).contiguous()
    noise = torch.randn(n, action_chunk, action_dim, generator=generator,
                        device=device, dtype=torch.float32).to(torch.bfloat16)
    out = model.forward_flow_action_full(
        inputs_embeds=rep(slow),
        position_ids=pos.expand(pos.shape[0], n, pos.shape[2]).contiguous(),
        attention_mask=rep(attn),
        noise=noise,
        state_embeds=rep(state),
        fast_embeds=rep(fast),
        num_steps=num_steps,
    )
    return out.float().cpu().numpy()


# ── plots ───────────────────────────────────────────────────────────────────
def make_plots(out_dir, errs, pred, gt, meta, n_examples, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = gt.shape[1]
    steps = np.arange(1, K + 1)
    titles = {"pos_cm": "Wrist position error [cm]", "rot_deg": "Wrist rotation error [deg]",
              "hand_deg": "Hand joint error [deg]"}
    styles = {"model": dict(color="#2563eb", lw=2.2), "model_avg": dict(color="#7c3aed", lw=1.8),
              "hold": dict(color="#9ca3af", ls="--", lw=1.6),
              "mean": dict(color="#f59e0b", ls=":", lw=1.6)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key in zip(axes, titles):
        for name, e in errs.items():
            m = e[key].mean(0)
            ax.plot(steps, m, label=name, **styles[name])
            if name == "model":
                ax.fill_between(steps, np.percentile(e[key], 25, 0), np.percentile(e[key], 75, 0),
                                color=styles[name]["color"], alpha=0.15, lw=0)
        ax.set_title(titles[key]); ax.set_xlabel("horizon step (30 fps)"); ax.grid(alpha=0.3)
    axes[0].legend(title="mean (model: IQR band)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "horizon_errors.png"), dpi=130); plt.close(fig)

    # Example chunks: xyz delta trajectories + 4 hand joints, predicted vs GT.
    rng = np.random.RandomState(seed)
    motion = np.linalg.norm(gt[:, -1, :3], axis=-1)
    cand = np.where(motion > np.median(motion))[0]          # pick frames that actually move
    idx = rng.choice(cand if len(cand) >= n_examples else np.arange(len(gt)),
                     size=min(n_examples, len(gt)), replace=False)
    fig, axes = plt.subplots(len(idx), 2, figsize=(12, 2.6 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        ax = axes[r, 0]
        for d, c in zip(range(3), ["#dc2626", "#16a34a", "#2563eb"]):
            ax.plot(steps, gt[i, :, d] * 100, color=c, lw=2, label=f"GT {'xyz'[d]}")
            ax.plot(steps, pred[i, :, d] * 100, color=c, lw=1.5, ls="--", label=f"pred {'xyz'[d]}")
        ax.set_ylabel("cm"); ax.grid(alpha=0.3)
        ax.set_title(f"{meta['source'][i]} ep{meta['episode'][i]} f{meta['frame'][i]}: wrist delta xyz",
                     fontsize=9)
        ax = axes[r, 1]
        hd = gt.shape[-1] - 9
        joints = np.argsort(-gt[i, :, 9:].std(0))[:4]       # the 4 most active joints
        for j, c in zip(joints, ["#7c3aed", "#db2777", "#0891b2", "#65a30d"]):
            ax.plot(steps, np.degrees(gt[i, :, 9 + j]), color=c, lw=2, label=f"GT j{j}")
            ax.plot(steps, np.degrees(pred[i, :, 9 + j]), color=c, lw=1.5, ls="--")
        ax.set_ylabel("deg"); ax.grid(alpha=0.3); ax.set_title(f"hand joints (of {hd}), solid=GT dashed=pred", fontsize=9)
        if r == 0:
            axes[r, 0].legend(fontsize=7, ncol=2); axes[r, 1].legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("horizon step")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "examples.png"), dpi=110); plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--base_model_path", default="", help="Qwen3-VL base; default: training_args.json model_path")
    p.add_argument("--out_dir", default="")
    p.add_argument("--cuda", default="0")
    p.add_argument("--max_samples", type=int, default=1500, help="frames evaluated, spread evenly over val")
    p.add_argument("--n_noise", type=int, default=1, help="action samples per frame (errors averaged)")
    p.add_argument("--num_steps", type=int, default=10, help="Euler flow steps")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42, help="must match training for the same val split")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--n_examples", type=int, default=6)
    # data settings: defaults mirror scripts/pretrain.sh
    p.add_argument("--action_dim", type=int, default=21)
    p.add_argument("--action_chunk", type=int, default=16)
    p.add_argument("--use_robot_state", type=int, default=1)
    p.add_argument("--image_size", type=int, nargs=2, default=[384, 288])
    add_lerobot_args(p)
    p.set_defaults(lerobot_root=["/home/iit.local/lgori/egomix_100gb/retarget"],
                   lerobot_fast_keys=["observation.images.wrist"], lerobot_base="action",
                   lerobot_video_backend="pyav", image_fit="crop", hand_dim=12)
    args = p.parse_args()
    args.use_flare = 0                      # no future frames needed for action eval

    out_dir = args.out_dir or os.path.join(args.checkpoint_path, "offline_eval")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(device)

    model, processor, ta = load_model(args, device)
    for k in ("action_dim", "action_chunk", "use_robot_state"):
        if k in ta and int(ta[k]) != int(getattr(args, k)):
            print(f"WARNING: {k}={getattr(args, k)} but checkpoint was trained with {ta[k]}; using checkpoint value")
            setattr(args, k, int(ta[k]))

    # ── data: same episode split as training, checkpoint normalization ──
    full = TRexLeRobotDataset(args, processor, _PrintAccelerator())
    val = full.create_val_split(val_ratio=args.val_ratio, seed=args.seed)
    with open(os.path.join(args.checkpoint_path, "stats_data.json")) as f:
        block = next(iter(json.load(f).values()))
    A = lambda k, s: np.array(block[k][s])
    a_mask, a_lo, a_hi = A("action", "mask"), A("action", "q01"), A("action", "q99")
    for ds in (val,):
        ds.action_mask, ds.action_min, ds.action_max = a_mask, a_lo, a_hi
        ds.state_mask, ds.state_min, ds.state_max = A("state", "mask"), A("state", "q01"), A("state", "q99")
        ds.te_std = np.zeros_like(ds.te_std)            # no state noise at eval
        ds.te_mean = np.zeros_like(ds.te_mean)

    n_val = len(val)
    n = min(args.max_samples, n_val)
    idx = np.unique(np.linspace(0, n_val - 1, n).round().astype(int))
    src_of = np.searchsorted(val._src_cum, idx, side="right")
    ep_of = np.searchsorted(val._cum_frames, idx, side="right")
    frame_of = idx - np.concatenate([[0], val._cum_frames])[ep_of]
    meta = {"source": np.array([val.sources[s].repo_id for s in src_of]),
            "episode": ep_of, "frame": frame_of, "index": idx}
    print(f"Evaluating {len(idx)} frames from {val._num_episodes} val episodes ({n_val} frames)", flush=True)

    def collate(batch):
        out = val.collate_fn(batch)
        out["gt"] = np.stack([x["action_chunk"] for x in batch])
        out["state_phys"] = np.stack([x["state"] for x in batch])
        return out

    loader = DataLoader(Subset(val, idx.tolist()), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        persistent_workers=False, prefetch_factor=4 if args.num_workers else None)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    preds, gts, states, norm_mse = [], [], [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        pn = predict_chunk(model, batch, device, args.n_noise, args.num_steps,
                           args.action_chunk, args.action_dim, gen)          # [n_noise, K, D]
        gt = batch["gt"][0]
        gn = normalize(gt, a_mask, a_lo, a_hi)
        norm_mse.append(float(((pn - gn[None]) ** 2).mean()))
        preds.append(denormalize(pn, a_mask, a_lo, a_hi))
        gts.append(gt)
        states.append(batch["state_phys"][0])
        if (i + 1) % 50 == 0 or i + 1 == len(idx):
            el = time.time() - t0
            print(f"  {i + 1}/{len(idx)}  {el / (i + 1):.2f}s/frame  eta {el / (i + 1) * (len(idx) - i - 1) / 60:.1f} min",
                  flush=True)

    pred_all = np.stack(preds)                     # [N, n_noise, K, D]
    gt = np.stack(gts)                             # [N, K, D]
    state = np.stack(states)                       # [N, D]
    N, S, K, D = pred_all.shape
    hd = args.hand_dim
    n_arms = D // (9 + hd)

    # Errors for each noise sample, then averaged over samples.
    e_samples = [chunk_errors(pred_all[:, s], gt, n_arms, hd) for s in range(S)]
    errs = {"model": {k: np.mean([e[k] for e in e_samples], 0) for k in e_samples[0]}}
    if S > 1:   # error of the average chunk over the samples (closer to the conditional mean)
        errs["model_avg"] = chunk_errors(pred_all.mean(1), gt, n_arms, hd)

    hold = np.zeros_like(gt)
    for a in range(n_arms):
        o = a * (9 + hd)
        hold[..., o + 3] = 1.0; hold[..., o + 7] = 1.0                    # identity rot6d
        hold[..., o + 9:o + 9 + hd] = state[:, None, o + 9:o + 9 + hd]    # current hand joints
    errs["hold"] = chunk_errors(hold, gt, n_arms, hd)
    errs["mean"] = chunk_errors(np.broadcast_to(gt.mean(0, keepdims=True), gt.shape), gt, n_arms, hd)

    # ── metrics ──
    def summarize(e, sel=slice(None)):
        return {k: {"mean": float(v[sel].mean()), "final_step": float(v[sel][:, -1].mean()),
                    "per_step": [float(x) for x in v[sel].mean(0)]} for k, v in e.items()}

    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint_path),
        "n_frames": int(N), "n_val_episodes": int(val._num_episodes), "n_noise": S,
        "num_flow_steps": args.num_steps,
        "normalized_mse_model": float(np.mean(norm_mse)),
        "gt_motion": {"final_step_pos_cm": float(np.linalg.norm(gt[:, -1, :3], axis=-1).mean() * 100)},
        "overall": {name: summarize(e) for name, e in errs.items()},
        "per_source": {},
    }
    if S > 1:
        metrics["sample_spread_final_pos_cm"] = float(pred_all[:, :, -1, :3].std(1).mean() * 100)
    for src in np.unique(meta["source"]):
        m = meta["source"] == src
        metrics["per_source"][str(src)] = {"n_frames": int(m.sum()),
                                           **{name: summarize(e, m) for name, e in errs.items()}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), pred=pred_all, gt=gt, state=state,
                        **{f"meta_{k}": v for k, v in meta.items()})
    make_plots(out_dir, errs, pred_all[:, 0], gt, meta, args.n_examples, args.seed)

    # ── summary ──
    lines = [f"# Offline eval: {os.path.basename(os.path.normpath(args.checkpoint_path))}", "",
             f"{N} frames from {val._num_episodes} held-out episodes, {S} sample(s)/frame, "
             f"{args.num_steps} flow steps. Normalized-space MSE: {metrics['normalized_mse_model']:.4f}. "
             f"Mean GT wrist displacement at step {K}: {metrics['gt_motion']['final_step_pos_cm']:.2f} cm.", "",
             "| | pos mean [cm] | pos @16 [cm] | rot mean [deg] | rot @16 [deg] | hand mean [deg] | hand @16 [deg] |",
             "|---|---|---|---|---|---|---|"]
    for name in metrics["overall"]:
        o = metrics["overall"][name]
        lines.append(f"| {name} | {o['pos_cm']['mean']:.2f} | {o['pos_cm']['final_step']:.2f} | "
                     f"{o['rot_deg']['mean']:.2f} | {o['rot_deg']['final_step']:.2f} | "
                     f"{o['hand_deg']['mean']:.2f} | {o['hand_deg']['final_step']:.2f} |")
    lines += ["", "Per source (model vs hold, mean over horizon):", "",
              "| source | frames | pos [cm] | rot [deg] | hand [deg] |", "|---|---|---|---|---|"]
    for src, o in metrics["per_source"].items():
        lines.append(f"| {src} | {o['n_frames']} | "
                     + " | ".join(f"{o['model'][k]['mean']:.2f} / {o['hold'][k]['mean']:.2f}"
                                  for k in ("pos_cm", "rot_deg", "hand_deg")) + " |")
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary + f"\n\nWritten to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
