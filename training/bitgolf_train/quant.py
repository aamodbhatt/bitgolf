"""Symmetric int8 fake quantisation that mirrors the Rust engine.

Weights are quantised per output channel (one scale per row of a `[out, in]` matrix) and
activations per token (one scale per vector along the last dimension), both with absmax
scaling to [-127, 127]. The straight-through estimator passes gradients through the rounding.
"""

from __future__ import annotations

import torch

QMAX = 127.0


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise along the last dimension; returns int8 values and float32 scales (`[..., 1]`)."""
    amax = x.detach().abs().amax(dim=-1, keepdim=True).float()
    scale = amax / QMAX
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(x.float() / safe), -QMAX, QMAX)
    q = torch.where(scale > 0, q, torch.zeros_like(q))
    return q.to(torch.int8), scale


def fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Round-trip `x` through per-row int8 with a straight-through gradient."""
    q, scale = quantize_rows(x)
    deq = (q.float() * scale).to(x.dtype)
    return x + (deq - x).detach()
