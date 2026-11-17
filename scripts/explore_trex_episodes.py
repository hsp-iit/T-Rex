"""Explore the episodes of the official T-REX dataset (zekaiwang/trex_dataset).

Reads only meta/episodes (one small parquet file, no videos).

Examples:
    python scripts/explore_trex_episodes.py                       # overview: primitives and objects
    python scripts/explore_trex_episodes.py --object knife        # primitives and episodes of an object
    python scripts/explore_trex_episodes.py --primitive reach     # objects and episodes of a primitive
    python scripts/explore_trex_episodes.py --object knife --list # one line per episode
    python scripts/explore_trex_episodes.py --episode 93          # one episode in detail
    python scripts/explore_trex_episodes.py --csv out.csv         # save the table
"""
import argparse

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "zekaiwang/trex_dataset"
COLS = ["episode_index", "motor_primitive", "object", "target", "length", "caption"]


def load() -> pd.DataFrame:
    p = hf_hub_download(REPO, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    e = pd.read_parquet(p)
    e["seconds"] = (e["length"] / 30).round(1)   # 30 fps
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", help="text contained in the object name (case-insensitive)")
    ap.add_argument("--primitive", help="exact motor primitive, e.g. reach, lift_and_place")
    ap.add_argument("--episode", type=int, help="show one episode")
    ap.add_argument("--list", action="store_true", help="print one line per episode")
    ap.add_argument("--csv", help="save the selected episodes to this CSV file")
    a = ap.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", 500)
    e = load()

    if a.episode is not None:
        r = e[e.episode_index == a.episode]
        if r.empty:
            raise SystemExit(f"episode {a.episode} not found")
        for c in COLS + ["seconds"]:
            print(f"{c:16s} {r.iloc[0][c]}")
        return

    s = e
    if a.object:
        s = s[s.object.str.contains(a.object, case=False, na=False)]
    if a.primitive:
        s = s[s.motor_primitive == a.primitive]
    if s.empty:
        raise SystemExit("no episodes match")

    print(f"{len(s)} episodes, {s.length.sum() / 30 / 3600:.1f} h, "
          f"{s.motor_primitive.nunique()} primitives, {s.object.nunique()} objects\n")

    if not a.object and not a.primitive:
        print(e.groupby("motor_primitive").agg(episodes=("episode_index", "size"), objects=("object", "nunique"),
                                               median_s=("seconds", "median"))
              .sort_values("episodes", ascending=False).to_string(), "\n")
        print("Objects with the most primitives:")
        print(e.groupby("object").motor_primitive.agg(["nunique", lambda x: sorted(set(x))])
              .set_axis(["n_primitives", "primitives"], axis=1)
              .sort_values("n_primitives", ascending=False).head(20).to_string())
    else:
        print(s.groupby(["object", "motor_primitive"]).agg(
            episodes=("episode_index", "size"), median_s=("seconds", "median"),
            episode_indices=("episode_index", lambda x: sorted(x)),
            example_caption=("caption", "first")).to_string())

    if a.list:
        print()
        print(s[COLS + ["seconds"]].drop(columns="length").sort_values("episode_index").to_string(index=False))
    if a.csv:
        s[COLS + ["seconds"]].to_csv(a.csv, index=False)
        print(f"\nsaved {len(s)} rows to {a.csv}")


if __name__ == "__main__":
    main()
