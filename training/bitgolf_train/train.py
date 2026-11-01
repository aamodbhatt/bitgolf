"""Train a byte-level model, then quantisation-aware fine-tune it to int8 and export it.

Phase 1 trains in bf16 autocast with AdamW (linear warmup, cosine decay to 10%).
Phase 2 enables int8 fake quantisation everywhere the Rust engine quantises and fine-tunes at a
lower learning rate. Metrics record the bits-per-byte of the float model, the same weights
quantised without fine-tuning (post-training quantisation), and the fine-tuned int8 model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainConfig, load_train_configs
from .data import SOURCES, ByteMixture
from .export import model_tensors, write_bgm
from .model import ByteLM, bits_per_byte


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def eval_sources(
    model: ByteLM, data: ByteMixture, cfg: TrainConfig, device: torch.device
) -> dict[str, float]:
    """Mean bits/byte per source over `eval_batches` fixed-seed validation batches."""
    model.eval()
    results: dict[str, float] = {}
    for source in SOURCES:
        data.rng = np.random.default_rng(cfg.seed + 99)
        total = 0.0
        for _ in range(cfg.eval_batches):
            batch = data.batch(cfg.batch_size, cfg.seq_len, source).to(device)
            with _autocast(device):
                logits = model(batch[:, :-1])
            total += float(bits_per_byte(logits, batch[:, 1:]))
        results[source] = total / cfg.eval_batches
    model.train()
    return results


@torch.no_grad()
def eval_file(model: ByteLM, data: bytes, device: torch.device, batch_size: int = 64) -> float:
    """Bits/byte over a file with sliding-window context, as the Rust engine decodes it.

    Each chunk of `window` bytes is scored with up to `window` bytes of preceding context.
    """
    model.eval()
    window = model.cfg.window
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    total_bits = 0.0
    seqs: list[tuple[np.ndarray, int]] = []

    def flush() -> float:
        width = max(len(s) for s, _ in seqs)
        tokens = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros((len(seqs), width - 1), dtype=torch.bool)
        for i, (s, ctx) in enumerate(seqs):
            tokens[i, : len(s)] = torch.from_numpy(s)
            mask[i, max(ctx - 1, 0) : len(s) - 1] = True
        tokens = tokens.to(device)
        with _autocast(device):
            logits = model(tokens[:, :-1]).float()
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)
        return float(nll[mask.to(device)].sum()) / math.log(2.0)

    # The first byte has no context, so the coder sends it with a uniform 8-bit code.
    for start in range(1, len(arr), window):
        ctx_start = max(0, start - window)
        seq = arr[ctx_start : min(len(arr), start + window)]
        seqs.append((seq, start - ctx_start))
        if len(seqs) == batch_size:
            total_bits += flush()
            seqs = []
    if seqs:
        total_bits += flush()
    model.train()
    return (total_bits + 8.0) / len(arr)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def _optimizer(model: ByteLM, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = [p for n, p in model.named_parameters() if p.ndim == 2 and "tok_emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or "tok_emb" in n]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=fused)


def _train_phase(
    model: ByteLM,
    step_fn: Any,
    data: ByteMixture,
    cfg: TrainConfig,
    steps: int,
    lr_fn: Any,
    device: torch.device,
    label: str,
    log: list[dict[str, Any]],
    opt: torch.optim.Optimizer,
) -> None:
    tokens_per_step = cfg.batch_size * cfg.seq_len
    started = time.perf_counter()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_fn(step)
        batch = data.batch(cfg.batch_size, cfg.seq_len).to(device, non_blocking=True)
        with _autocast(device):
            logits = step_fn(batch[:, :-1])
        loss = bits_per_byte(logits, batch[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            elapsed = time.perf_counter() - started
            rate = tokens_per_step * (step + 1) / max(elapsed, 1e-9)
            entry = {
                "phase": label,
                "step": step,
                "bpb": float(loss.detach()),
                "lr": lr_fn(step),
                "bytes_per_s": rate,
                "elapsed_s": elapsed,
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)


def train(cfg: TrainConfig, eval_files: list[Path]) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    mcfg: ModelConfig = cfg.model_config()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)
    train_data = ByteMixture(data_dir, "train", cfg.mixture(), cfg.seed)
    val_data = ByteMixture(data_dir, "val", cfg.mixture(), cfg.seed + 1)

    model = ByteLM(mcfg).to(device)
    step_fn: Any = torch.compile(model) if cfg.compile and device.type == "cuda" else model
    print(f"model {mcfg.name}: {model.num_params():,} parameters on {device}", flush=True)
    log: list[dict[str, Any]] = []
    started = time.perf_counter()

    opt = _optimizer(model, cfg.lr, cfg.weight_decay)
    _train_phase(
        model,
        step_fn,
        train_data,
        cfg,
        cfg.steps,
        lambda s: _lr_at(s, cfg),
        device,
        "float",
        log,
        opt,
    )
    float_eval = eval_sources(model, val_data, cfg, device)
    model.set_quantized(True)
    ptq_eval = eval_sources(model, val_data, cfg, device)

    qat_opt = _optimizer(model, cfg.qat_lr, cfg.weight_decay)
    qat_fn: Any = torch.compile(model) if cfg.compile and device.type == "cuda" else model
    _train_phase(
        model,
        qat_fn,
        train_data,
        cfg,
        cfg.qat_steps,
        lambda s: cfg.qat_lr * (1.0 - s / max(1, cfg.qat_steps)),
        device,
        "qat",
        log,
        qat_opt,
    )
    qat_eval = eval_sources(model, val_data, cfg, device)
    train_seconds = time.perf_counter() - started

    file_evals: dict[str, dict[str, float]] = {}
    for path in eval_files:
        data = path.read_bytes()
        model.set_quantized(True)
        int8_bpb = eval_file(model, data, device)
        model.set_quantized(False)
        float_bpb = eval_file(model, data, device)
        file_evals[path.name] = {"bytes": len(data), "float_bpb": float_bpb, "int8_bpb": int8_bpb}
        print(json.dumps({"file": path.name, **file_evals[path.name]}), flush=True)
    model.set_quantized(False)

    meta = {
        "train_config": asdict(cfg),
        "params": model.num_params(),
        "train_seconds": round(train_seconds, 1),
        "bytes_seen": cfg.batch_size * cfg.seq_len * (cfg.steps + cfg.qat_steps),
        "val_bpb": {"float": float_eval, "int8_ptq": ptq_eval, "int8_qat": qat_eval},
        "file_bpb": file_evals,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    torch.save(
        {"model_config": asdict(mcfg), "state_dict": model.state_dict(), "meta": meta},
        out / f"{mcfg.name}.pt",
    )
    model_id = write_bgm(out / f"bitgolf-{mcfg.name}.bgm", mcfg, model_tensors(model), meta)
    meta["model_sha256"] = model_id
    (out / f"{mcfg.name}.metrics.json").write_text(json.dumps(meta | {"log": log}, indent=2))
    print(
        json.dumps({"done": mcfg.name, "val_bpb": meta["val_bpb"], "sha256": model_id}), flush=True
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export bitgolf models")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", default=None, help="train only this preset")
    parser.add_argument("--eval-file", type=Path, action="append", default=[])
    ns = parser.parse_args()
    for cfg in load_train_configs(ns.config):
        if ns.only is None or ns.only == cfg.model:
            train(cfg, ns.eval_file)


if __name__ == "__main__":
    main()
