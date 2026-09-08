#!/usr/bin/env python3
"""Regenerate the cold-prefill prompts used by bench-prefill.sh.

Three ~33k-token prompts, each built from a different part of the llama.cpp tree so
every request is a genuine cold prefill. The generated files are committed: the
sources drift with every llama.cpp pull, and a benchmark arm must be compared on
byte-identical prompts, so only regenerate deliberately and re-baseline afterwards.
"""
import glob
import os

ROOT = os.environ.get("LLAMA_REPO", os.path.expanduser("~/git/llama.cpp"))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = 125_000  # bytes; ~33k tokens of C++ on the Qwen tokeniser

SETS = {
    "server": sorted(glob.glob(f"{ROOT}/tools/server/*.cpp")),
    "core": sorted(glob.glob(f"{ROOT}/src/llama-*.cpp")),
    "metal": sorted(glob.glob(f"{ROOT}/ggml/src/ggml-metal/*.cpp"))
    + sorted(glob.glob(f"{ROOT}/ggml/src/ggml-metal/*.m")),
}

for name, files in SETS.items():
    out = [f"Review the following {name} source files for concurrency bugs. Reply with the single most severe finding.\n\n"]
    n = len(out[0])
    for f in files:
        if n >= TARGET:
            break
        s = open(f, errors="replace").read()
        chunk = f"// ==== {os.path.relpath(f, ROOT)} ====\n" + s[: TARGET - n]
        out.append(chunk)
        n += len(chunk)
    path = os.path.join(OUT_DIR, f"{name}.txt")
    with open(path, "w") as fh:
        fh.write("".join(out))
    print(path, n, "bytes")
