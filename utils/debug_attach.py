"""Optional VS Code (debugpy) attach point for the training scripts.

Does nothing unless TREX_DEBUGPY_PORT is set. When it is, the selected ranks
listen on <TREX_DEBUGPY_HOST>:<TREX_DEBUGPY_PORT + local_rank> and block until
VS Code attaches ("Attach to T-Rex job" in .vscode/launch.json).

Environment:
  TREX_DEBUGPY_PORT   base port, e.g. 5678 (unset = debugging off)
  TREX_DEBUGPY_HOST   interface to listen on (default 0.0.0.0, reachable from the login node)
  TREX_DEBUGPY_RANKS  comma-separated local ranks that listen (default "0"; "all" = every rank).
                      Ranks that do not listen keep running and wait at the first collective.
  TREX_DEBUGPY_WAIT   1 (default) = block until the debugger attaches; 0 = keep running.

Breakpoints inside Dataset.__getitem__ / collate_fn are only hit with --num_workers 0,
because DataLoader workers are separate processes.
"""
import os
import socket


def maybe_wait_for_debugger() -> None:
    port = os.environ.get("TREX_DEBUGPY_PORT")
    if not port:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ranks = os.environ.get("TREX_DEBUGPY_RANKS", "0").strip()
    if ranks != "all" and local_rank not in {int(r) for r in ranks.split(",") if r.strip()}:
        return

    import debugpy

    host = os.environ.get("TREX_DEBUGPY_HOST", "0.0.0.0")
    port = int(port) + local_rank
    debugpy.listen((host, port))
    print(f"[debugpy] rank {local_rank} listening on {socket.gethostname()}:{port} "
          f"(attach VS Code with host={socket.gethostname()}, port={port})", flush=True)
    if os.environ.get("TREX_DEBUGPY_WAIT", "1") != "0":
        print(f"[debugpy] rank {local_rank} waiting for the debugger to attach...", flush=True)
        debugpy.wait_for_client()
        print(f"[debugpy] rank {local_rank} debugger attached", flush=True)
