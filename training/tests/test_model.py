"""Model, quantisation and config behaviour on the tiny CPU preset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bitgolf_train.config import PRESETS, ModelConfig, load_train_configs
from bitgolf_train.model import ByteLM, bits_per_byte
from bitgolf_train.quant import fake_quant, quantize_rows


def test_presets_are_valid_and_alibi_slopes_are_powers_of_two() -> None:
    for cfg in PRESETS.values():
        assert cfg.d_model % cfg.n_heads == 0
        for s in cfg.alibi_slopes():
            assert s == 2.0 ** round(torch.log2(torch.tensor(s)).item())


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig("bad", d_model=30, n_layers=1, n_heads=4, n_kv_heads=1, d_mlp=8, window=64)


def test_quantize_rows_bounds_error_and_handles_zeros() -> None:
    x = torch.randn(16, 64)
    q, scale = quantize_rows(x)
    assert q.dtype == torch.int8 and int(q.abs().max()) <= 127
    err = (q.float() * scale - x).abs().max(dim=-1).values
    assert torch.all(err <= scale.squeeze(-1) / 2 + 1e-6)
    zq, zs = quantize_rows(torch.zeros(3, 8))
    assert int(zq.abs().sum()) == 0 and float(zs.sum()) == 0.0


def test_fake_quant_passes_gradients_straight_through() -> None:
    x = torch.randn(4, 32, requires_grad=True)
    fake_quant(x).sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))


def test_forward_shapes_and_uniform_cost() -> None:
    model = ByteLM(PRESETS["tiny"])
    tokens = torch.randint(0, 256, (2, 40))
    logits = model(tokens)
    assert logits.shape == (2, 40, 256)
    uniform = torch.zeros(1, 5, 256)
    assert abs(float(bits_per_byte(uniform, torch.zeros(1, 5, dtype=torch.long))) - 8.0) < 1e-5


@pytest.mark.parametrize("quantized", [False, True])
def test_sliding_window_ignores_bytes_older_than_the_window(quantized: bool) -> None:
    cfg = PRESETS["tiny"]
    model = ByteLM(cfg).eval()
    model.set_quantized(quantized)
    seq = torch.randint(0, 256, (1, cfg.window * 3))
    changed = seq.clone()
    changed[0, 0] = (int(seq[0, 0]) + 1) % 256
    with torch.no_grad():
        a, b = model(seq), model(changed)
    # Position window and later cannot see position 0 at any depth (each layer looks back
    # window-1 bytes, so position t sees back to t - n_layers*(window-1)); check the last one.
    reach = cfg.n_layers * (cfg.window - 1)
    if seq.shape[1] - 1 > reach:
        assert torch.equal(a[0, -1], b[0, -1])
    assert not torch.equal(a[0, 1], b[0, 1])


def test_laptop_config_loads(tmp_path: Path) -> None:
    configs = load_train_configs(Path(__file__).parents[1] / "configs" / "laptop.toml")
    assert configs[0].model_config().name == "tiny"
    bad = tmp_path / "bad.toml"
    bad.write_text('[[run]]\nmodel = "tiny"\nsurprise = 1\n')
    with pytest.raises(ValueError):
        load_train_configs(bad)
