"""Read the {key: (shape, dtype)} index of a remote torch ``model.pt`` (zip format)
via HTTP range requests, without downloading the tensor data.

torch.save writes a zip whose ``<prefix>/data.pkl`` holds the pickled state
dict with tensors as persistent-id references.  We fetch only the zip central
directory and that pickle, then unpickle with stubs that record shapes.

Usage:
    python -m taptic.remote_ckpt_index <url> <out.json>
"""
from __future__ import annotations

import io
import json
import pickle
import struct
import sys

import requests


def _get(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, allow_redirects=True, timeout=120)
    r.raise_for_status()
    return r.content


def _size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _central_directory(url: str, size: int):
    tail = _get(url, max(0, size - 65536), size - 1)
    base = size - len(tail)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("zip end-of-central-directory not found")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:          # ZIP64
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = _get(url, cd_off, cd_off + cd_size - 1) if cd_off < base else tail[cd_off - base:cd_off - base + cd_size]
    entries, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        loff = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q < len(extra):                                  # zip64 extra field
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for flag in (usize, csize, loff):
                    if flag == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                    else:
                        vals.append(flag)
                usize, csize, loff = vals
            q += 4 + hl
        entries[name] = (loff, csize)
        p += 46 + nlen + elen + clen
    return entries


class _Stub:
    def __init__(self, *a, **k):
        self.a = a


def _rebuild(storage, offset, size, stride, *rest):
    return {"shape": list(size), "dtype": storage}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a: data
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch":
            return name                                        # e.g. "BFloat16Storage" / dtype name
        return _Stub

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        st = pid[1]
        return str(st if isinstance(st, str) else getattr(st, "__name__", st))


def remote_state_dict_index(url: str) -> dict:
    size = _size(url)
    entries = _central_directory(url, size)
    pkl = [n for n in entries if n.endswith("/data.pkl") or n == "data.pkl"][0]
    loff, csize = entries[pkl]
    hdr = _get(url, loff, loff + 29)
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    data = _get(url, loff + 30 + nlen + elen, loff + 30 + nlen + elen + csize - 1)
    sd = _Unpickler(io.BytesIO(data)).load()
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k: v for k, v in sd.items() if isinstance(v, dict) and "shape" in v}


if __name__ == "__main__":
    idx = remote_state_dict_index(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(idx, f, indent=0)
    print(f"{len(idx)} tensors -> {sys.argv[2]}")
