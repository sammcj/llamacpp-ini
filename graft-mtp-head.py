#!/usr/bin/env python3
"""Graft a standalone Qwen3.8-Flash-Next MTP head GGUF onto an existing split
quant as an extra split, so llama.cpp PR #27836 (embedded-MTP design) can load it.

Rewrites shard 1 (metadata-only) with block_count+nextn KVs and split.count=4,
rewrites the head as split 4, and hardlinks the untouched weight shards.

Usage: graft-mtp-head.py <shard1.gguf> <head.gguf> <out-dir> <out-basename>
"""

import os
import sys
from pathlib import Path
from typing import Optional

import gguf

SPLIT_NO = "split.no"
SPLIT_COUNT = "split.count"
SPLIT_TENSORS = "split.tensors.count"
SUPPRESS = (gguf.Keys.General.ARCHITECTURE, "GGUF.")


def copy_kvs(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, overrides: dict) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = overrides.pop(field.name, (None, None))[0]
        if value is None:
            value = field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    for key, (value, val_type) in overrides.items():
        writer.add_key_value(key, value, val_type)


def head_tensor_name(name: str, n_layer: int) -> "Optional[str]":
    """Map a standalone head tensor to its grafted name; None = shared with the trunk."""
    if name.startswith(f"blk.{n_layer}."):
        return name
    renames = {  # standalone-head names for the nextn hyper-connection head
        "output_hc_norm.weight": f"blk.{n_layer}.nextn.hc_head_norm.weight",
        "output_hc_down.weight": f"blk.{n_layer}.nextn.hc_head_down.weight",
        "output_hc_up.weight": f"blk.{n_layer}.nextn.hc_head_up.weight",
    }
    return renames.get(name)


def write_out(writer: gguf.GGUFWriter, reader: gguf.GGUFReader, names: "Optional[dict]" = None) -> None:
    tensors = [t for t in reader.tensors if names is None or t.name in names]
    for t in tensors:
        name = names[t.name] if names else t.name
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()


def main() -> None:
    shard1, head, out_dir, base = sys.argv[1:5]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    r1 = gguf.GGUFReader(shard1)
    arch = r1.fields[gguf.Keys.General.ARCHITECTURE].contents()
    n_layer = r1.fields[f"{arch}.block_count"].contents()

    # the head repo also ships trunk tensors (shared embeddings, final norms) for
    # standalone use; graft only the MTP block, renaming its hyper-connection head
    head_reader = gguf.GGUFReader(head)
    head_names = {}
    for t in head_reader.tensors:
        mapped = head_tensor_name(t.name, n_layer)
        if mapped is not None:
            head_names[t.name] = mapped
    n_head_tensors = len(head_names)
    ratios_field = r1.fields.get(f"{arch}.attention.compress_ratios")
    n_split = r1.fields[SPLIT_COUNT].contents()
    n_tensors_total = r1.fields[SPLIT_TENSORS].contents()

    u32 = gguf.GGUFValueType.UINT32
    overrides = {
        f"{arch}.block_count": (n_layer + 1, u32),
        f"{arch}.nextn_predict_layers": (1, u32),
        SPLIT_COUNT: (n_split + 1, gguf.GGUFValueType.UINT16),
        SPLIT_TENSORS: (n_tensors_total + n_head_tensors, gguf.GGUFValueType.INT32),
    }
    if ratios_field is not None:
        overrides[f"{arch}.attention.compress_ratios"] = (
            list(ratios_field.contents()) + [0],
            gguf.GGUFValueType.ARRAY,
        )

    def split_name(idx: int) -> Path:
        return out / f"{base}-{idx + 1:05d}-of-{n_split + 1:05d}.gguf"

    w1 = gguf.GGUFWriter(split_name(0), arch)
    copy_kvs(r1, w1, overrides)
    write_out(w1, r1)
    print(f"wrote {split_name(0)} (block_count {n_layer}->{n_layer + 1}, "
          f"splits {n_split}->{n_split + 1}, tensors {n_tensors_total}->{n_tensors_total + n_head_tensors})")

    src = Path(shard1)
    stem = src.name[: src.name.rfind(f"-00001-of-{n_split:05d}.gguf")]
    for idx in range(1, n_split):
        link = split_name(idx)
        target = src.with_name(f"{stem}-{idx + 1:05d}-of-{n_split:05d}.gguf")
        if not link.exists():
            os.link(target, link)
        print(f"linked {link} -> {target.name}")

    wh = gguf.GGUFWriter(split_name(n_split), arch)
    wh.add_key_value(SPLIT_NO, n_split, gguf.GGUFValueType.UINT16)
    wh.add_key_value(SPLIT_COUNT, n_split + 1, gguf.GGUFValueType.UINT16)
    wh.add_key_value(SPLIT_TENSORS, n_tensors_total + n_head_tensors, gguf.GGUFValueType.INT32)
    write_out(wh, head_reader, names=head_names)
    print(f"wrote {split_name(n_split)} ({n_head_tensors} MTP head tensors)")


if __name__ == "__main__":
    main()
