"""`.bgm` export round trip and integrity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bitgolf_train.config import PRESETS
from bitgolf_train.export import model_tensors, read_bgm, write_bgm
from bitgolf_train.model import ByteLM


def test_round_trip_preserves_tensors(tmp_path: Path) -> None:
    cfg = PRESETS["tiny"]
    tensors = model_tensors(ByteLM(cfg))
    path = tmp_path / "m.bgm"
    model_id = write_bgm(path, cfg, tensors, {"note": "test"})
    assert len(model_id) == 64
    header, back = read_bgm(path)
    assert header["config"]["d_model"] == cfg.d_model
    assert header["meta"] == {"note": "test"}
    assert set(back) == set(tensors)
    for name, arr in tensors.items():
        assert np.array_equal(back[name], arr), name
    assert tensors["layers.0.wqkv"].dtype == np.int8
    assert tensors["layers.0.wqkv.scale"].shape == (tensors["layers.0.wqkv"].shape[0],)


def test_corruption_is_detected(tmp_path: Path) -> None:
    cfg = PRESETS["tiny"]
    path = tmp_path / "m.bgm"
    write_bgm(path, cfg, model_tensors(ByteLM(cfg)), {})
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        read_bgm(path)


def test_non_finite_weights_are_refused(tmp_path: Path) -> None:
    cfg = PRESETS["tiny"]
    tensors = model_tensors(ByteLM(cfg))
    tensors["final_norm"][0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        write_bgm(tmp_path / "m.bgm", cfg, tensors, {})
