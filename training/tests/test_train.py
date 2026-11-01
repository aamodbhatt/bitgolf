"""End-to-end CPU smoke test: train, fine-tune with fake quantisation, export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bitgolf_train.config import TrainConfig
from bitgolf_train.export import read_bgm
from bitgolf_train.train import train


def _write_sources(data_dir: Path) -> None:
    rng = np.random.default_rng(0)
    text = b"the cat sat on the mat. " * 400
    for name in ("prose", "code", "structured"):
        for split in ("train", "val"):
            noise = rng.integers(97, 123, size=2000, dtype=np.uint8).tobytes()
            (data_dir / f"{name}_{split}.bin").write_bytes(text + noise)


def test_tiny_run_trains_quantises_and_exports(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    cfg = TrainConfig(
        model="tiny",
        data_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        seq_len=128,
        batch_size=8,
        steps=30,
        qat_steps=5,
        lr=3e-3,
        qat_lr=3e-4,
        warmup_steps=5,
        weight_decay=0.1,
        eval_every=100,
        eval_batches=1,
        seed=1,
        device="cpu",
        compile=False,
        mix_prose=1.0,
        mix_code=1.0,
        mix_structured=1.0,
    )
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"the cat sat on the mat. " * 20)
    meta = train(cfg, [sample])
    val = meta["val_bpb"]
    assert val["float"]["prose"] < 8.0, "training must beat a uniform model"
    assert abs(val["int8_ptq"]["prose"] - val["float"]["prose"]) < 0.5
    assert 0.0 < meta["file_bpb"]["sample.txt"]["int8_bpb"] < 8.0
    header, tensors = read_bgm(tmp_path / "out" / "bitgolf-tiny.bgm")
    assert header["name"] == "tiny" and "lm_head" in tensors
    metrics = json.loads((tmp_path / "out" / "tiny.metrics.json").read_text())
    assert metrics["model_sha256"]
