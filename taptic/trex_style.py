"""Re-cut the merged TAPTIC midtrain dataset into T-REX-style episodes.

T-REX midtrain episodes (https://huggingface.co/datasets/zekaiwang/trex_dataset,
paper App. G) are one object x one motor primitive, start with the hand free,
and carry one imperative sentence ("caption") describing the whole motion.
This module builds that structure from the 702 TAPTIC recordings
("version 1": one episode per goal, frames never shared between episodes):

* single-object tasks: one recording -> one episode;
* fruit_in_the_basket / clearing_table: one recording -> one episode per
  object, cut where the release that follows a transport ends.

Recordings or trailing frames without a completed goal are dropped.

Two stages:
    python -m taptic.trex_style plan  --root <merged> --out <plan_dir>
    python -m taptic.trex_style build --root <merged> --plan <plan_dir> --out <dataset_dir>

``plan`` writes ``pieces.csv`` (one row per new episode + dropped spans) and
contact sheets for naming the objects of the multi-object tasks. Object names
go in ``objects.csv`` (piece_id, object), written after looking at the sheets.
``build`` writes a LeRobot v3.0 dataset. The source dataset is only read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

MULTI_OBJECT = {"fruit_in_the_basket", "clearing_table"}

# T-REX primitive, target and caption template per task. {obj} is replaced by
# the object name for the multi-object tasks.
TASKS: Dict[str, Dict[str, str]] = {
    "insert_credit_card": dict(
        primitive="insert", object="credit card", target="wallet",
        caption="Pick up the credit card with your right hand and insert it into the wallet."),
    "insert_game_object_cylinder": dict(
        primitive="insert", object="elliptical cylinder", target="hole",
        caption="Pick up the elliptical cylinder with your right hand and insert it into the matching hole of the wooden box."),
    "grasp_plastic_glass": dict(
        primitive="lift_and_place", object="plastic glass", target="other box",
        caption="Gently pick up the plastic glass with your right hand and place it on the other box."),
    # Adapted from the T-REX paper's "extract" example (last appendix): red -> white cups.
    "split_cups": dict(
        primitive="extract", object="white plastic cup", target="",
        caption=("A stack of white plastic cups sits on the desktop; use the right hand to slide "
                 "out the topmost one, exerting effort to separate it from the rest of the stack.")),
    "dosing_pipette": dict(
        primitive="dispense", object="pipette", target="glass",
        caption=("Pick up the pipette with your right hand, squeeze it to release the liquid "
                 "into the glass, and put it back into the bottle.")),
    "wiping_table": dict(
        primitive="wipe", object="blue cloth", target="whiteboard",
        caption="Grasp the blue cloth with your right hand and wipe the whiteboard with it."),
    "unscrew_the_bottle": dict(
        primitive="unscrew", object="bottle cap", target="",
        caption="Unscrew the bottle cap with your right hand and lift it off."),
    "fruit_in_the_basket": dict(
        primitive="lift_and_place", object="{obj}", target="plastic box",
        caption="Place the {obj} into the plastic box with your right hand."),
    "clearing_table": dict(
        primitive="lift_and_place", object="{obj}", target="plastic box",
        caption="Place the {obj} into the plastic box with your right hand."),
}


# "Version 2": T-REX's two largest primitives, reach and grasp_and_lifting, carved out
# of whole pieces. A carved piece keeps only its first reach (reach) or its first
# reach -> grasp/pinch -> lift (grasp_and_lifting); the rest of the recording is
# dropped, so frames are still never shared. Pinch and grasp are one primitive, as
# in T-REX ("pick up the pipette" and "pick up the backpack" are both
# grasp_and_lifting). Captions follow the T-REX templates.
CARVE_COUNTS: Dict[str, Dict[str, int]] = {
    "grasp_and_lifting": {"fruit_in_the_basket": 20, "clearing_table": 20, "grasp_plastic_glass": 15,
                          "insert_credit_card": 15, "insert_game_object_cylinder": 15,
                          "dosing_pipette": 5},
    "reach": {"clearing_table": 15, "fruit_in_the_basket": 15, "split_cups": 10,
              "insert_game_object_cylinder": 10, "grasp_plastic_glass": 10},
}
CARVE_CAPTIONS = {
    "grasp_and_lifting": "Pick up the {obj} with your right hand.",
    "reach": "Move your right hand toward the {obj}.",
}
# Object as named in a reach/pick-up caption, when it differs from TASKS[...]["object"].
CARVE_OBJECT = {"split_cups": "top cup of the stack"}


def carve(p: pd.DataFrame, frames: pd.DataFrame, counts=CARVE_COUNTS, seed: int = 42) -> pd.DataFrame:
    """Turn randomly chosen kept pieces into reach / grasp_and_lifting episodes.

    Adds a ``primitive`` column ("" = the task's primitive), shortens ``end`` of the
    carved pieces and adds the cut-off tails as dropped rows. piece_id is unchanged.
    """
    p = p.copy()
    p["primitive"] = ""
    rng = np.random.default_rng(seed)
    cut: Dict[int, tuple] = {}   # row index -> (primitive, new_end)
    for prim in ("grasp_and_lifting", "reach"):   # g&l first: it has the stricter pattern
        for task, n in counts[prim].items():
            cands = []
            for i, r in p[(p.task == task) & p.keep].iterrows():
                if i in cut:
                    continue
                lab = frames.label.values[frames.episode_index.values == r.source_episode][r.start:r.end]
                segs = _segments(lab)
                names = [s[0] for s in segs]
                if names[0] != "reach" or len(names) < 2:
                    continue
                if prim == "reach":
                    cands.append((i, r.start + segs[0][2]))
                elif len(names) >= 3 and names[1] in ("grasp", "pinch") and names[2] == "lift":
                    cands.append((i, r.start + segs[2][2]))
            if len(cands) < n:
                raise ValueError(f"{task}: only {len(cands)} clean candidates for {prim}, need {n}")
            for j in rng.choice(len(cands), size=n, replace=False):
                cut[cands[j][0]] = (prim, cands[j][1])
    tails = []
    for i, (prim, new_end) in cut.items():
        r = p.loc[i]
        tails.append(dict(r, start=new_end, end=r.end, keep=False, cycle=-1, piece_id=-1, primitive="",
                          reason=f"after the {prim} cut of piece {r.piece_id}"))
        p.loc[i, ["primitive", "end"]] = [prim, new_end]
    p = pd.concat([p, pd.DataFrame(tails)], ignore_index=True)
    p["length"] = p.end - p.start
    return p.sort_values(["source_episode", "start"]).reset_index(drop=True)


def _segments(labels: np.ndarray):
    ch = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(labels[ch[i]], int(ch[i]), int(ch[i + 1])) for i in range(len(ch) - 1)]  # [s, e)


def load_frames(root: str) -> pd.DataFrame:
    sub = {int(k): v for k, v in json.load(open(os.path.join(root, "meta/subtasks.json")))
           ["index_to_subtask"].items()}
    f = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "subtask_index"])
                   for p in sorted(glob.glob(os.path.join(root, "data/*/*.parquet")))])
    f = f.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    f["label"] = f.subtask_index.map(sub)
    return f


def plan(root: str) -> pd.DataFrame:
    """One row per span of a source recording: kept pieces and dropped spans."""
    merged = json.load(open(os.path.join(root, "meta/merged_from.json")))
    frames = load_frames(root)
    rows: List[dict] = []
    for m in merged:
        task = m["repo_id"].split("/")[-1]
        a, b = m["episodes"]
        for ep in range(a, b):
            lab = frames.label.values[frames.episode_index.values == ep]
            n = len(lab)
            segs = _segments(lab)
            base = dict(task=task, source_episode=ep, local_episode=ep - a,
                        tactile_raw=m["tactile_raw"], source_length=n)
            if task in MULTI_OBJECT:
                start, moved, k = 0, False, 0
                for name, s, e in segs:
                    if name == "move":
                        moved = True
                    if name == "release" and moved:
                        rows.append(dict(base, start=start, end=e, keep=True, cycle=k, reason=""))
                        k += 1
                        start, moved = e, False
                if start < n:
                    rows.append(dict(base, start=start, end=n, keep=False, cycle=-1,
                                     reason="no completed transport->release after this point"))
                continue
            names = [s[0] for s in segs]
            if set(names) <= {"reach"}:
                ok, why = False, "reach only (interrupted)"
            elif task in {"insert_credit_card", "insert_game_object_cylinder", "grasp_plastic_glass"}:
                ok = any(x == "move" for x in names) and "release" in names[names.index("move"):] \
                    if "move" in names else False
                why = "" if ok else "object never transported and released (failed attempt)"
            else:
                ok, why = True, ""
            rows.append(dict(base, start=0, end=n, keep=ok, cycle=0 if ok else -1, reason=why))
    p = pd.DataFrame(rows)
    p["length"] = p.end - p.start
    keep = p[p.keep].copy()
    keep["piece_id"] = np.arange(len(keep))
    p = p.merge(keep[["source_episode", "start", "piece_id"]], on=["source_episode", "start"], how="left")
    p["piece_id"] = p.piece_id.fillna(-1).astype(int)
    return p


def contact_sheets(root: str, p: pd.DataFrame, out: str, per_sheet: int = 12):
    """Wrist + scene frame at the middle of the first lift of each multi-object piece."""
    from PIL import Image, ImageDraw
    from .source import TapticSource
    from .video import VideoReader

    frames = load_frames(root)
    vr = VideoReader()
    os.makedirs(out, exist_ok=True)
    for task in sorted(MULTI_OBJECT):
        src = TapticSource(root, task)
        q = p[(p.task == task) & p.keep].reset_index(drop=True)
        tiles = []
        for _, r in q.iterrows():
            lab = frames.label.values[frames.episode_index.values == r.source_episode][r.start:r.end]
            lifts = np.flatnonzero(lab == "lift")
            moves = np.flatnonzero(lab == "move")
            t = r.start + int(lifts[len(lifts) // 2] if len(lifts) else moves[0])
            ims = []
            for cam in ("observation.images.wrist", "observation.images.left"):
                path, t0, _ = src.video_ref(int(r.source_episode), cam)
                ims.append(Image.fromarray(vr.frame(path, t0 + t / 10 + 0.05, 10)).resize((240, 180)))
            tile = Image.new("RGB", (480, 200), "white")
            tile.paste(ims[0], (0, 20)); tile.paste(ims[1], (240, 20))
            ImageDraw.Draw(tile).text((4, 4), f"piece {r.piece_id}  ep {r.source_episode} f{t}", fill="black")
            tiles.append(tile)
        for i in range(0, len(tiles), per_sheet):
            chunk = tiles[i:i + per_sheet]
            cols = 3
            sheet = Image.new("RGB", (480 * cols, 200 * ((len(chunk) + cols - 1) // cols)), "white")
            for j, t in enumerate(chunk):
                sheet.paste(t, ((j % cols) * 480, (j // cols) * 200))
            sheet.save(os.path.join(out, f"{task}_{i // per_sheet:03d}.jpg"), quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("plan"); a.add_argument("--root", required=True); a.add_argument("--out", required=True)
    a.add_argument("--no_sheets", action="store_true")
    c = sp.add_parser("carve", help="add reach / grasp_and_lifting episodes to a plan (version 2)")
    c.add_argument("--root", required=True); c.add_argument("--plan", required=True)
    c.add_argument("--out", required=True); c.add_argument("--seed", type=int, default=42)
    b = sp.add_parser("build"); b.add_argument("--root", required=True); b.add_argument("--plan", required=True)
    b.add_argument("--out", required=True); b.add_argument("--val_ratio", type=float, default=0.1)
    b.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.cmd == "plan":
        os.makedirs(args.out, exist_ok=True)
        p = plan(args.root)
        p.to_csv(os.path.join(args.out, "pieces.csv"), index=False)
        k = p[p.keep]
        print(k.groupby("task").agg(episodes=("piece_id", "size"), frames=("length", "sum"),
                                    median_s=("length", lambda x: x.median() / 10)).to_string())
        print("kept", len(k), "episodes,", k.length.sum(), "frames; dropped spans",
              (~p.keep).sum(), "frames", p[~p.keep].length.sum())
        if not args.no_sheets:
            contact_sheets(args.root, p, os.path.join(args.out, "sheets"))
    elif args.cmd == "carve":
        import shutil
        os.makedirs(args.out, exist_ok=True)
        p = carve(pd.read_csv(os.path.join(args.plan, "pieces.csv")), load_frames(args.root), seed=args.seed)
        p.to_csv(os.path.join(args.out, "pieces.csv"), index=False)
        shutil.copy2(os.path.join(args.plan, "objects.csv"), os.path.join(args.out, "objects.csv"))
        k = p[p.keep].assign(primitive=lambda d: d.primitive.replace("", "(task)"))
        print(k.groupby(["primitive", "task"]).agg(episodes=("piece_id", "size"),
                                                   median_s=("length", lambda x: x.median() / 10)).to_string())
    else:
        from .trex_style_build import build
        build(args.root, args.plan, args.out, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
