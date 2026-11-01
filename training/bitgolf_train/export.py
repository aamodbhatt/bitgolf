"""Read and write the versioned `.bgm` model format consumed by the Rust engine.

Layout (little-endian):

    magic      8 bytes   b"BGMODEL\\x00"
    version    u32       FORMAT_VERSION
    json_len   u32       length of the UTF-8 JSON header
    header     json_len  {"name", "config", "tensors": [{name, dtype, shape, offset, nbytes}],
                         "meta"}
    padding    to a 64-byte boundary; tensor offsets are relative to this point
    tensors    raw little-endian data, each tensor aligned to 64 bytes
    sha256     32 bytes over everything before it

Linear weights and the embedding table are stored as int8 with one float32 scale per row
(`<name>.scale`); norms and ALiBi slopes stay float32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .model import Block, ByteLM, QLinear
from .quant import quantize_rows

MAGIC = b"BGMODEL\x00"
FORMAT_VERSION = 1
ALIGN = 64


def _pad(n: int) -> int:
    return (-n) % ALIGN


def model_tensors(model: ByteLM) -> dict[str, np.ndarray]:
    """Quantise a trained model into the named int8/float32 arrays stored in a `.bgm`."""
    out: dict[str, np.ndarray] = {}

    def add_int8(name: str, weight: torch.Tensor) -> None:
        q, scale = quantize_rows(weight.detach().float().cpu())
        out[name] = q.numpy().astype(np.int8)
        out[f"{name}.scale"] = scale.squeeze(-1).numpy().astype(np.float32)

    def add_f32(name: str, value: torch.Tensor) -> None:
        out[name] = value.detach().float().cpu().numpy().astype(np.float32)

    add_int8("tok_emb", model.tok_emb)
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, Block)
        prefix = f"layers.{i}"
        add_f32(f"{prefix}.attn_norm", layer.attn_norm.weight)
        for name in ("wqkv", "wo", "w_in", "w_out"):
            linear = getattr(layer, name)
            assert isinstance(linear, QLinear)
            add_int8(f"{prefix}.{name}", linear.weight)
        add_f32(f"{prefix}.mlp_norm", layer.mlp_norm.weight)
    add_f32("final_norm", model.final_norm.weight)
    add_int8("lm_head", model.lm_head.weight)
    out["alibi_slopes"] = np.asarray(model.cfg.alibi_slopes(), dtype=np.float32)
    return out


def write_bgm(
    path: Path, cfg: ModelConfig, tensors: dict[str, np.ndarray], meta: dict[str, Any]
) -> str:
    """Write a `.bgm` file and return the sha256 of the whole file (the model id)."""
    entries = []
    blobs = []
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype == np.int8:
            dtype = "i8"
        elif arr.dtype == np.float32:
            dtype = "f32"
        else:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        if not np.all(np.isfinite(arr.astype(np.float64))):
            raise ValueError(f"{name}: contains non-finite values")
        raw = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        entries.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        blobs.append(raw + b"\x00" * _pad(len(raw)))
        offset += len(raw) + _pad(len(raw))
    header = json.dumps(
        {"name": cfg.name, "config": asdict(cfg), "tensors": entries, "meta": meta}, sort_keys=True
    ).encode("utf-8")
    prefix = MAGIC + struct.pack("<II", FORMAT_VERSION, len(header)) + header
    body = prefix + b"\x00" * _pad(len(prefix)) + b"".join(blobs)
    digest = hashlib.sha256(body).digest()
    data = body + digest
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_bgm(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Parse and verify a `.bgm` file; returns the JSON header and the tensors."""
    data = path.read_bytes()
    if len(data) < 48 or data[:8] != MAGIC:
        raise ValueError(f"{path}: not a bitgolf model (bad magic)")
    body, digest = data[:-32], data[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError(f"{path}: checksum mismatch, the file is corrupt")
    version, json_len = struct.unpack_from("<II", data, 8)
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format version {version}")
    header: dict[str, Any] = json.loads(data[16 : 16 + json_len])
    base = 16 + json_len
    base += _pad(base)
    tensors: dict[str, np.ndarray] = {}
    for entry in header["tensors"]:
        dtype = np.dtype("<i1") if entry["dtype"] == "i8" else np.dtype("<f4")
        start = base + entry["offset"]
        raw = body[start : start + entry["nbytes"]]
        tensors[entry["name"]] = np.frombuffer(raw, dtype=dtype).reshape(entry["shape"]).copy()
    return header, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a training checkpoint to .bgm")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    ns = parser.parse_args()
    ckpt = torch.load(ns.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = ByteLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model_id = write_bgm(ns.out, cfg, model_tensors(model), ckpt.get("meta", {}))
    print(f"wrote {ns.out} ({ns.out.stat().st_size} bytes) sha256={model_id}")


if __name__ == "__main__":
    main()
