"""Lazy frame access for LeRobot v3.0 concatenated AV1 videos.

Videos are GOP-2 AV1 (``video.g = 2``), so seek + decode of one frame costs
~2 ms; no per-episode frame cache is needed (the official loader decodes whole
episodes into RAM, which does not scale to 640x480 x 2 cameras x 900 frames).
"""
from __future__ import annotations

from collections import OrderedDict

import av
import numpy as np


class VideoReader:
    def __init__(self, max_open: int = 8):
        self._open: "OrderedDict[str, av.container.InputContainer]" = OrderedDict()
        self.max_open = max_open

    def _container(self, path: str):
        c = self._open.pop(path, None)
        if c is None:
            c = av.open(path)
            if len(self._open) >= self.max_open:
                _, old = self._open.popitem(last=False)
                old.close()
        self._open[path] = c
        return c

    def frame(self, path: str, ts: float, fps: int) -> np.ndarray:
        """RGB uint8 [H,W,3] of the frame whose pts is ``ts`` seconds (+- half a frame)."""
        c = self._container(path)
        s = c.streams.video[0]
        c.seek(int(max(0.0, ts - 0.5 / fps) / s.time_base), stream=s, backward=True, any_frame=False)
        last = None
        for fr in c.decode(s):
            if fr.time is None:
                continue
            if fr.time >= ts - 0.5 / fps:
                if fr.time > ts + 0.5 / fps:
                    raise RuntimeError(f"{path}: no frame at {ts:.3f}s (next {fr.time:.3f}s)")
                return fr.to_ndarray(format="rgb24")
            last = fr
        raise RuntimeError(f"{path}: frame at {ts:.3f}s not found (last {getattr(last, 'time', None)})")

    def close(self):
        for c in self._open.values():
            c.close()
        self._open.clear()

    def __getstate__(self):              # DataLoader workers: never pickle open containers
        return {"max_open": self.max_open, "_open": OrderedDict()}
