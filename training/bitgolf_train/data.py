"""Memory-mapped byte streams and a seeded mixture sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SOURCES = ("prose", "code", "structured")


class ByteMixture:
    """Samples `[batch, seq_len + 1]` windows from several memory-mapped sources."""

    def __init__(self, data_dir: Path, split: str, mixture: dict[str, float], seed: int) -> None:
        self.streams: dict[str, np.memmap] = {}
        for name in SOURCES:
            path = data_dir / f"{name}_{split}.bin"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run `python -m bitgolf_train.prepare`")
            self.streams[name] = np.memmap(path, dtype=np.uint8, mode="r")
        self.names = [n for n in SOURCES if mixture.get(n, 0.0) > 0]
        weights = np.array([mixture[n] for n in self.names], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int, seq_len: int, source: str | None = None) -> torch.Tensor:
        """Draw a batch; with `source` set, all rows come from that source."""
        if source is None:
            picks = self.rng.choice(len(self.names), size=batch_size, p=self.weights)
            names = [self.names[i] for i in picks]
        else:
            names = [source] * batch_size
        rows = np.empty((batch_size, seq_len + 1), dtype=np.uint8)
        for i, name in enumerate(names):
            stream = self.streams[name]
            if len(stream) <= seq_len + 1:
                raise ValueError(f"source {name!r} is shorter than one sequence")
            start = int(self.rng.integers(0, len(stream) - seq_len - 1))
            rows[i] = stream[start : start + seq_len + 1]
        return torch.from_numpy(rows.astype(np.int64))
