"""Pre-start checks for the TAPTIC pipeline. Run on the GPU node, inside the allocation.

- the expected number of GPUs is visible (respects CUDA_VISIBLE_DEVICES);
- each GPU has enough memory, supports bf16, and this torch build has kernels for it;
- each GPU is idle (nobody else is using it): never share a GPU with another job;
- deepspeed imports;
- the output filesystem has the required space and stays <= 90% used.
"""
import argparse
import shutil
from pathlib import Path


def check_gpus(n, min_mem_gib):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("PREFLIGHT FAIL: no CUDA GPU visible")
    count = torch.cuda.device_count()
    if count != n:
        raise SystemExit(f"PREFLIGHT FAIL: {count} GPUs visible, expected {n} "
                         "(set CUDA_VISIBLE_DEVICES / request the right allocation)")
    arch = set(torch.cuda.get_arch_list())
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        sm = f"sm_{major}{minor}"
        if sm not in arch:
            raise SystemExit(f"PREFLIGHT FAIL: GPU {i} {name} is {sm}; torch {torch.__version__} "
                             f"has kernels for {sorted(arch)}. A different torch build is needed: "
                             "stop and ask (do not change versions silently).")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {i} {name} has no bf16 support")
        free, total = torch.cuda.mem_get_info(i)
        gib = 2 ** 30
        if total / gib < min_mem_gib:
            raise SystemExit(f"PREFLIGHT FAIL: GPU {i} {name} has {total / gib:.0f} GiB, "
                             f"need >= {min_mem_gib}")
        # our own fresh context uses ~0.5 GiB; more than ~2 GiB means someone else is on it
        if (total - free) / gib > 2.0:
            raise SystemExit(f"PREFLIGHT FAIL: GPU {i} {name} already uses "
                             f"{(total - free) / gib:.1f} GiB: busy, refusing to share it")
        print(f"GPU OK: {i} {name} {sm} {total / gib:.0f} GiB", flush=True)


def check_deepspeed():
    import deepspeed
    print(f"deepspeed OK: {deepspeed.__version__}", flush=True)


def check_disk(path, need_gib, max_used_pct=90.0):
    p = Path(path).resolve()
    a = p
    while not a.exists():
        a = a.parent
    u = shutil.disk_usage(a)
    need = need_gib * 2 ** 30
    if u.free < need or (u.used + need) > max_used_pct / 100 * u.total:
        raise SystemExit(f"PREFLIGHT FAIL: {a}: {u.free / 2**30:.0f} GiB free; need {need_gib} GiB "
                         f"while keeping the disk <= {max_used_pct:g}% used")
    print(f"Storage OK: {p} ({u.free / 2**30:.0f} GiB free, {100 * u.used / u.total:.0f}% used)",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gpus", type=int, required=True)
    ap.add_argument("--min-mem-gib", type=float, default=40)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--need-gib", type=float, required=True)
    ap.add_argument("--max-used-pct", type=float, default=90.0,
                    help="disk usage ceiling after the job (lab rule: 90)")
    a = ap.parse_args()
    check_gpus(a.n_gpus, a.min_mem_gib)
    check_deepspeed()
    check_disk(a.storage, a.need_gib, a.max_used_pct)
    print("PREFLIGHT OK", flush=True)
