#!/usr/bin/env python3
"""Inspect tensors in remote HF-hosted GGUFs via HTTP range requests (no download).

Parses only each shard's header (a few MB) to report a tensor's quant type,
size and dims - handy for checking how a quantiser treated a specific tensor
(e.g. Qwen3.8-Flash-Next's per_layer_token_embd) across quant tiers before
committing to a 100GB download.

Usage:
  gguf-remote-probe.py <repo> [tensor-name] [quant-dir-filter ...]

  gguf-remote-probe.py bartowski/Qwen3.8-Flash-Next-GGUF
  gguf-remote-probe.py unsloth/Qwen3.8-Flash-Next-GGUF per_layer_token_embd.weight UD-Q4_K_XL UD-Q5_K_XL

Needs gguf-py on the path for the quant block sizes (falls back to the
~/git/llama.cpp checkout).
"""
import json
import struct
import sys
import urllib.request

try:
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
except ModuleNotFoundError:
    sys.path.insert(0, "/Users/samm/git/llama.cpp/gguf-py")
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

UA = {"User-Agent": "gguf-header-probe"}


class RemoteReader:
    CHUNK = 4 * 1024 * 1024

    def __init__(self, url):
        self.url = url
        self.buf = b""
        self.pos = 0

    def _fetch(self, upto):
        while len(self.buf) < upto:
            start = len(self.buf)
            req = urllib.request.Request(self.url, headers={
                **UA, "Range": f"bytes={start}-{start + self.CHUNK - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = b""
                while True:
                    part = r.read(65536)
                    if not part:
                        break
                    data += part
            if not data:
                raise EOFError("no more data")
            self.buf += data

    def read(self, n):
        self._fetch(self.pos + n)
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", "replace")


SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def skip_value(r, vtype):
    if vtype in SCALAR:
        r.read(SCALAR[vtype])
    elif vtype == 8:
        r.string()
    elif vtype == 9:
        et = r.u32()
        n = r.u64()
        if et in SCALAR:
            r.read(SCALAR[et] * n)
        elif et == 8:
            for _ in range(n):
                r.string()
        else:
            raise ValueError(f"nested array type {et}")
    else:
        raise ValueError(f"kv type {vtype}")


def probe(url, tensor_name):
    r = RemoteReader(url)
    if r.read(4) != b"GGUF":
        raise ValueError("not a gguf")
    r.u32()  # version
    n_tensors = r.u64()
    n_kv = r.u64()
    for _ in range(n_kv):
        r.string()
        skip_value(r, r.u32())
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        r.u64()  # offset
        if name == tensor_name:
            qt = GGMLQuantizationType(ttype)
            blk, tsize = GGML_QUANT_SIZES[qt]
            nelem = 1
            for d in dims:
                nelem *= d
            return f"{qt.name:8s} {nelem // blk * tsize / 2**30:7.1f} GiB  dims={dims}"
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    repo = sys.argv[1]
    tensor_name = sys.argv[2] if len(sys.argv) > 2 else "per_layer_token_embd.weight"
    filters = sys.argv[3:]

    tree_req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", headers=UA)
    tree = json.load(urllib.request.urlopen(tree_req, timeout=60))
    by_quant = {}
    for f in tree:
        p = f.get("path", "")
        if p.endswith(".gguf"):
            by_quant.setdefault(p.split("/")[0] if "/" in p else p, []).append(p)

    base = f"https://huggingface.co/{repo}/resolve/main"
    for quant in sorted(by_quant):
        if filters and not any(x in quant for x in filters):
            continue
        found = None
        for path in sorted(by_quant[quant]):
            try:
                found = probe(f"{base}/{path}", tensor_name)
            except Exception as e:
                print(f"{path}: error {e}", file=sys.stderr)
                continue
            if found:
                print(f"{quant:28s} {found}  ({path.split('/')[-1]})")
                break
        if not found:
            print(f"{quant:28s} {tensor_name} not found")


if __name__ == "__main__":
    main()
