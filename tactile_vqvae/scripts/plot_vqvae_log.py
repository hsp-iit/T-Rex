"""Plot tactile VQ-VAE training metrics from a training log (stdout / Slurm log).

    python tactile_vqvae/scripts/plot_vqvae_log.py slurm-413.log            # -> slurm-413.png
    python tactile_vqvae/scripts/plot_vqvae_log.py slurm-413.log --out m.png --csv m.csv

Parses the lines printed by tactile_vqvae/train.py:
    [step    1800 | ep  0] recon=0.0065 vq=0.0034 perp=11.8 active=59/64 revived=0 lr=3.00e-04 elapsed=17s
      [val @ step 2000] val_recon=0.0074 val_vq=0.0063 val_perplexity=7.8452 val_active=15.5400

A log can hold several runs (e.g. the pipeline's short probe, then the full run):
the step counter restarting marks a new run. The last one is plotted (--run).

What to look for:
  recon    reconstruction error             -> goes down, val close to train
  vq       commitment loss                  -> small and stable
  active   codes used per batch (of K)      -> train close to K
  perp     how evenly codes are used        -> high and stable
  revived  dead codes replaced              -> goes to 0
Validation batches are NOT shuffled (256 consecutive windows of one episode),
so val active/perplexity are lower than train by construction: compare them
over time, not against train.
"""
import argparse
import csv
import re
from pathlib import Path

TRAIN_RE = re.compile(
    r"\[step\s+(\d+)\s*\|\s*ep\s+(\d+)\]\s+recon=([\d.eE+-]+)\s+vq=([\d.eE+-]+)\s+"
    r"perp=([\d.eE+-]+)\s+active=(\d+)/(\d+)\s+revived=(\d+)\s+lr=([\d.eE+-]+)\s+elapsed=(\d+)s")
VAL_RE = re.compile(r"\[val @ step (\d+)\]\s+(.*)")

# reference categorical palette, slots 1-2 (validated pair)
TRAIN_C, VAL_C = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse(path):
    runs, cur, last_step = [], None, None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            step = int(m.group(1))
            if cur is None or (last_step is not None and step < last_step):
                cur = {"train": [], "val": []}
                runs.append(cur)
            last_step = step
            cur["train"].append({
                "step": step, "epoch": int(m.group(2)), "recon": float(m.group(3)),
                "vq": float(m.group(4)), "perp": float(m.group(5)),
                "active": int(m.group(6)), "K": int(m.group(7)),
                "revived": int(m.group(8)), "lr": float(m.group(9)),
                "elapsed": int(m.group(10))})
            continue
        m = VAL_RE.search(line)
        if m and cur is not None:
            row = {"step": int(m.group(1))}
            for kv in m.group(2).split():
                k, v = kv.split("=")
                row[k.replace("val_", "")] = float(v)
            cur["val"].append(row)
    return runs


def smooth(y, w):
    if w <= 1 or len(y) < 2:
        return list(y)
    out, acc = [], 0.0
    for i, v in enumerate(y):
        acc += v
        if i >= w:
            acc -= y[i - w]
        out.append(acc / min(i + 1, w))
    return out


def plot(run, out, smooth_w, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, va = run["train"], run["val"]
    K = tr[0]["K"]
    xs = [r["step"] for r in tr]
    vx = [r["step"] for r in va]
    panels = [  # key, title, what to look for, has val, reference line
        ("recon", "Reconstruction error", "should go down", True, None),
        ("vq", "Commitment loss (vq)", "small and stable", True, None),
        ("active", f"Codes used per batch (of {K})", f"train close to {K}", True, K),
        ("perp", "Perplexity", "high and stable", True, None),
        ("revived", "Dead codes revived", "should go to 0", False, None),
        ("lr", "Learning rate", "warmup + cosine decay", False, None),
    ]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True)
    for ax, (key, name, hint, has_val, ref) in zip(axes.flat, panels):
        y = [r[key] for r in tr]
        if key in ("recon", "vq", "perp", "active") and smooth_w > 1:
            ax.plot(xs, y, color=TRAIN_C, lw=0.8, alpha=0.25)
            ax.plot(xs, smooth(y, smooth_w), color=TRAIN_C, lw=2, label="train (smoothed)")
        else:
            ax.plot(xs, y, color=TRAIN_C, lw=2, label="train")
        if has_val and va:
            vk = {"perp": "perplexity"}.get(key, key)
            ax.plot(vx, [r[vk] for r in va], color=VAL_C, lw=2, marker="o", ms=4,
                    markeredgecolor="white", markeredgewidth=1, label="validation")
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
        if key in ("recon", "vq"):
            ax.set_yscale("log")
        ax.set_title(f"{name}\n", loc="left", fontsize=10, color=INK)
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=8, color=INK2)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("optimizer step")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False)
    last = tr[-1]
    fig.suptitle(f"{title}  -  epoch {last['epoch']}, step {last['step']}, "
                 f"{last['elapsed'] / 60:.0f} min", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, "Validation batches are not shuffled (consecutive windows of one episode): "
             "val codes/perplexity are lower than train by construction; compare them over time.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=130)
    print(f"plot: {out}")


def summary(run):
    tr, va = run["train"], run["val"]
    n = max(1, len(tr) // 20)
    first, last = tr[:n], tr[-n:]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    print(f"train points {len(tr)}, val points {len(va)}, last epoch {tr[-1]['epoch']}, "
          f"last step {tr[-1]['step']}")
    for k in ("recon", "vq", "perp", "active", "revived"):
        print(f"  {k:8s} start {avg(first, k):.4g}  ->  end {avg(last, k):.4g}")
    if va:
        print("  val      " + "  ".join(f"{k} {va[0][k]:.4g} -> {va[-1][k]:.4g}"
                                         for k in va[0] if k != "step"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--out", help="PNG path (default: <log>.png)")
    ap.add_argument("--csv", help="also write the parsed train/val points to this CSV")
    ap.add_argument("--run", type=int, default=-1, help="which run in the log (default: last)")
    ap.add_argument("--smooth", type=int, default=8, help="moving-average window for train curves")
    a = ap.parse_args()
    runs = parse(a.log)
    if not runs:
        raise SystemExit(f"no VQ-VAE training lines found in {a.log}")
    run = runs[a.run]
    print(f"{len(runs)} run(s) in log; plotting run {a.run if a.run >= 0 else len(runs) + a.run}")
    summary(run)
    if a.csv:
        keys = sorted({k for r in run["train"] + run["val"] for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split"] + keys)
            w.writeheader()
            for split in ("train", "val"):
                for r in run[split]:
                    w.writerow({"split": split, **r})
        print(f"csv: {a.csv}")
    plot(run, a.out or str(Path(a.log).with_suffix(".png")), a.smooth, Path(a.log).name)
