"""Write the tiny parity fixture used by the Rust tests.

Exports a seeded random `tiny` model to `.bgm` together with the next-byte logits that the
PyTorch reference (with int8 fake quantisation) produces for a fixed text longer than the
attention window. The Rust engine must reproduce these logits to within float tolerance,
which checks the file format, the architecture and the sliding-window cache end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import PRESETS
from .export import model_tensors, write_bgm
from .model import ByteLM

PARITY_TEXT = (
    "bitgolf parity fixture: the quick brown fox jumps over the lazy dog. "
    '{"id": 42, "tags": ["alpha", "beta"]} 2025-01-01T00:00:00Z INFO ok\n'
    'fn main() { println!("hello"); } -- longer than the 64-byte window.'
)


def build(out_dir: Path, seed: int = 7) -> dict[str, object]:
    torch.manual_seed(seed)
    cfg = PRESETS["tiny"]
    model = ByteLM(cfg)
    # Larger-than-default weights so predictions are far from uniform and parity is meaningful.
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                p.mul_(8.0)
    model.set_quantized(True)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = write_bgm(out_dir / "tiny.bgm", cfg, model_tensors(model), {"fixture": True})
    tokens = torch.tensor([list(PARITY_TEXT.encode("utf-8"))], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)[0]
    expected = {
        "model_sha256": model_id,
        "text": PARITY_TEXT,
        "logits": [[round(float(v), 6) for v in row] for row in logits],
    }
    (out_dir / "tiny_expected.json").write_text(json.dumps(expected) + "\n")
    return {"model_sha256": model_id, "positions": len(expected["logits"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("../crates/bitgolf-core/tests/fixtures"))
    print(json.dumps(build(parser.parse_args().out)))


if __name__ == "__main__":
    main()
