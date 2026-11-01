"""Byte-level decoder-only transformer, numerically aligned with the Rust inference engine.

Architecture (per layer, pre-norm residual):

    h = RMSNorm(x);  q, k, v = W_qkv h         (multi-query: one shared K/V head)
    a = softmax(q k^T / sqrt(d_head) - slope * distance, sliding window W) v
    x = x + W_o a
    h = RMSNorm(x);  x = x + W_out relu(W_in h)^2

With `quantized=True` every linear layer sees int8-rounded inputs and per-channel int8 weights,
and q/k/v are rounded per head, exactly where the Rust engine quantises. Training uses
sliding-window attention on sequences longer than the window so the model never relies on
seeing the start of a sequence, which is what a ring-buffer KV cache gives it at inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .quant import fake_quant


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out: torch.Tensor = (normed * self.weight.float()).to(x.dtype)
        return out


class QLinear(nn.Module):
    """Bias-free linear layer with optional int8 fake quantisation of inputs and weights."""

    def __init__(self, d_in: int, d_out: int, std: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * std)
        self.quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return F.linear(fake_quant(x), fake_quant(self.weight))
        return F.linear(x, self.weight)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        std = 0.02
        out_std = std / math.sqrt(2 * cfg.n_layers)
        qkv_out = cfg.d_model + 2 * cfg.n_kv_heads * cfg.d_head
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.wqkv = QLinear(cfg.d_model, qkv_out, std)
        self.wo = QLinear(cfg.d_model, cfg.d_model, out_std)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.w_in = QLinear(cfg.d_model, cfg.d_mlp, std)
        self.w_out = QLinear(cfg.d_mlp, cfg.d_model, out_std)
        self.quantized = False

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch, seq, _ = x.shape
        qkv = self.wqkv(self.attn_norm(x))
        kv_dim = cfg.n_kv_heads * cfg.d_head
        q, k, v = qkv.split([cfg.d_model, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, seq, cfg.n_heads, cfg.d_head)
        k = k.view(batch, seq, cfg.n_kv_heads, cfg.d_head)
        v = v.view(batch, seq, cfg.n_kv_heads, cfg.d_head)
        if self.quantized:
            q, k, v = fake_quant(q), fake_quant(k), fake_quant(v)
        group = cfg.n_heads // cfg.n_kv_heads
        k = k.repeat_interleave(group, dim=2)
        v = v.repeat_interleave(group, dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias.to(q.dtype)
        )
        x = x + self.wo(attn.transpose(1, 2).reshape(batch, seq, cfg.d_model))
        hidden = F.relu(self.w_in(self.mlp_norm(x))).square()
        out: torch.Tensor = x + self.w_out(hidden)
        return out


def attention_bias(cfg: ModelConfig, seq: int, device: torch.device) -> torch.Tensor:
    """ALiBi distance penalty restricted to a causal sliding window of `cfg.window` keys."""
    pos = torch.arange(seq, device=device)
    dist = (pos[:, None] - pos[None, :]).float()
    visible = (dist >= 0) & (dist < cfg.window)
    slopes = torch.tensor(cfg.alibi_slopes(), device=device).view(-1, 1, 1)
    bias = -slopes * dist
    return torch.where(visible, bias, torch.full_like(bias, float("-inf"))).unsqueeze(0)


class ByteLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Parameter(torch.randn(cfg.vocab, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = QLinear(cfg.d_model, cfg.vocab, 0.02)
        self._quantized = False
        self._bias_cache: dict[tuple[int, str], torch.Tensor] = {}

    @property
    def quantized(self) -> bool:
        return self._quantized

    def set_quantized(self, enabled: bool) -> None:
        """Toggle int8 fake quantisation everywhere the Rust engine quantises."""
        self._quantized = enabled
        for module in self.modules():
            if isinstance(module, QLinear | Block):
                module.quantized = enabled

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _bias(self, seq: int, device: torch.device) -> torch.Tensor:
        key = (seq, str(device))
        if key not in self._bias_cache:
            self._bias_cache[key] = attention_bias(self.cfg, seq, device)
        return self._bias_cache[key]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map `[batch, seq]` byte ids to `[batch, seq, 256]` next-byte logits."""
        emb = fake_quant(self.tok_emb) if self._quantized else self.tok_emb
        x = F.embedding(tokens, emb)
        bias = self._bias(tokens.shape[1], tokens.device)
        for layer in self.layers:
            x = layer(x, bias)
        logits: torch.Tensor = self.lm_head(self.final_norm(x)).float()
        return logits


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy in bits per predicted byte."""
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss / math.log(2.0)
