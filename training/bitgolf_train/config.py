"""Typed configuration for models and training runs.

Model shapes are chosen for fast, deterministic CPU decoding rather than for GPU efficiency:
multi-query attention keeps the per-stream KV cache small (it dominates memory traffic when
decoding many independent blocks), ReLU^2 needs no transcendental functions, and ALiBi needs
no sin/cos tables, so the Rust engine never calls libm.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of a byte-level decoder-only transformer."""

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_mlp: int
    window: int
    vocab: int = 256
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads={self.n_heads} must be a multiple of n_kv_heads")
        if self.vocab != 256:
            raise ValueError("bitgolf models are byte-level: vocab must be 256")
        if self.window < 16:
            raise ValueError(f"window={self.window} is too small")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def alibi_slopes(self) -> list[float]:
        """Geometric ALiBi slopes 2^(-8(i+1)/H); exact powers of two when H is a power of two."""
        return [2.0 ** (-8.0 * (i + 1) / self.n_heads) for i in range(self.n_heads)]


PRESETS: dict[str, ModelConfig] = {
    "nano": ModelConfig(
        "nano", d_model=128, n_layers=4, n_heads=4, n_kv_heads=1, d_mlp=512, window=512
    ),
    "small": ModelConfig(
        "small", d_model=256, n_layers=6, n_heads=4, n_kv_heads=1, d_mlp=1024, window=512
    ),
    # Used by unit tests and the Rust fixtures: fast on CPU, same code paths as real models.
    "tiny": ModelConfig(
        "tiny", d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, d_mlp=64, window=64
    ),
}


@dataclass(frozen=True)
class TrainConfig:
    """One training run: a float phase followed by a quantisation-aware fine-tune."""

    model: str
    data_dir: str
    out_dir: str
    seq_len: int
    batch_size: int
    steps: int
    qat_steps: int
    lr: float
    qat_lr: float
    warmup_steps: int
    weight_decay: float
    eval_every: int
    eval_batches: int
    seed: int
    device: str
    compile: bool
    mix_prose: float
    mix_code: float
    mix_structured: float

    def model_config(self) -> ModelConfig:
        if self.model not in PRESETS:
            raise ValueError(f"unknown model preset {self.model!r}; known: {sorted(PRESETS)}")
        return PRESETS[self.model]

    def mixture(self) -> dict[str, float]:
        total = self.mix_prose + self.mix_code + self.mix_structured
        if total <= 0:
            raise ValueError("data mixture weights must sum to a positive number")
        return {
            "prose": self.mix_prose / total,
            "code": self.mix_code / total,
            "structured": self.mix_structured / total,
        }


def load_train_configs(path: Path) -> list[TrainConfig]:
    """Load one or more `[[run]]` tables from a TOML file; `[defaults]` fill missing keys."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults: dict[str, Any] = raw.get("defaults", {})
    runs: list[dict[str, Any]] = raw.get("run", [])
    if not runs:
        raise ValueError(f"{path}: expected at least one [[run]] table")
    known = {f.name for f in fields(TrainConfig)}
    configs = []
    for run in runs:
        merged = {**defaults, **run}
        unknown = set(merged) - known
        missing = known - set(merged)
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        configs.append(TrainConfig(**merged))
    return configs
