"""Benchmark bitgolf against standard compressors and write tables and plots.

Baselines run through their command-line tools (gzip, xz, zstd, brotli) with wall-clock
timing; bitgolf runs through `bitgolf bench`, which compresses, decompresses and verifies the
round trip. Stages append rows to a JSON Lines file; `report` turns them into
`<out>/results.json`, `<out>/results.md` and `<out>/pareto.png`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

BASELINES: dict[str, tuple[list[str], list[str]]] = {
    "gzip -9": (["gzip", "-9", "-c"], ["gzip", "-d", "-c"]),
    "xz -9e": (["xz", "-9e", "-c", "-T1"], ["xz", "-d", "-c"]),
    "zstd -19": (["zstd", "-19", "-c", "-q"], ["zstd", "-d", "-c", "-q"]),
    "zstd --ultra -22": (["zstd", "--ultra", "-22", "-c", "-q"], ["zstd", "-d", "-c", "-q"]),
    "brotli -q 11": (["brotli", "-q", "11", "-c"], ["brotli", "-d", "-c"]),
}

# The model decodes at single-digit KB/s per core. A bitgolf row that decoded faster than this
# had every block fall back to deflate, so its decode time measures deflate, not the model.
MODEL_DECODE_CEILING_KB_S = 1_000.0


@dataclass(frozen=True)
class Row:
    corpus: str
    method: str
    raw_bytes: int
    stored_bytes: int
    bits_per_byte: float
    encode_kb_s: float
    decode_kb_s: float
    threads: int
    model_bytes: int

    @property
    def total_with_model(self) -> int:
        return self.stored_bytes + self.model_bytes


def run_baseline(name: str, path: Path) -> Row:
    enc, dec = BASELINES[name]
    raw = path.read_bytes()
    started = time.perf_counter()
    packed = subprocess.run([*enc, str(path)], capture_output=True, check=True).stdout
    enc_s = time.perf_counter() - started
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(packed)
    started = time.perf_counter()
    restored = subprocess.run([*dec, tmp.name], capture_output=True, check=True).stdout
    dec_s = time.perf_counter() - started
    Path(tmp.name).unlink()
    if restored != raw:
        raise RuntimeError(f"{name} failed to round-trip {path}")
    return Row(
        path.name,
        name,
        len(raw),
        len(packed),
        len(packed) * 8 / len(raw),
        len(raw) / 1e3 / enc_s,
        len(raw) / 1e3 / dec_s,
        1,
        0,
    )


def run_bitgolf(
    binary: Path, model: Path, path: Path, threads: int, block_size: int, limit: int | None
) -> Row:
    cmd = [
        str(binary),
        "bench",
        str(path),
        "--model",
        str(model),
        "--thread-counts",
        str(threads),
        "--block-size",
        str(block_size),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
    r = json.loads(out.strip().splitlines()[-1])
    if not r["verified"]:
        raise RuntimeError(f"bitgolf round trip failed on {path}")
    return Row(
        r["file"] if limit is None else f"{r['file']}[:{limit}]",
        f"bitgolf-{r['model']}",
        r["raw_bytes"],
        r["stored_bytes"],
        r["bits_per_byte"],
        r["encode_kb_s"],
        r["decode_kb_s"],
        threads,
        r["model_bytes"],
    )


def markdown(rows: list[Row]) -> str:
    lines = [
        "| corpus | method | threads | bits/byte | ratio | stored bytes | + model bytes "
        "| encode KB/s | decode KB/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.corpus} | {r.method} | {r.threads} | {r.bits_per_byte:.3f} "
            f"| {r.raw_bytes / r.stored_bytes:.2f} | {r.stored_bytes:,} "
            f"| {r.total_with_model:,} | {r.encode_kb_s:,.1f} | {r.decode_kb_s:,.1f} |"
        )
    return "\n".join(lines) + "\n"


def pareto_plot(rows: list[Row], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Whole files coded by both bitgolf (multi-threaded) and every baseline; slices used only
    # for the thread-scaling rows are named "file[:n]" and left out.
    corpora = sorted(
        {
            r.corpus
            for r in rows
            if r.method.startswith("bitgolf") and r.threads > 1 and "[:" not in r.corpus
        }
    )
    # Decode speed is pooled only over files the model actually decoded, and over the same files
    # for every method, so a deflate-speed fallback file cannot inflate bitgolf's throughput.
    deflate_only = sorted(
        {
            r.corpus
            for r in rows
            if r.method.startswith("bitgolf")
            and r.corpus in corpora
            and r.threads > 1
            and r.decode_kb_s > MODEL_DECODE_CEILING_KB_S
        }
    )
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    # Baselines share one neutral colour and differ by shape; the two models carry the colour.
    # Each entry: marker, colour, label offset in points.
    styles: dict[str, tuple[str, str, tuple[int, int]]] = {
        "gzip -9": ("o", muted, (9, -3)),
        "xz -9e": ("s", muted, (-9, -10)),
        "zstd -19": ("^", muted, (-9, 4)),
        "zstd --ultra -22": ("v", muted, (9, -10)),
        "brotli -q 11": ("D", muted, (9, -3)),
        "bitgolf-nano": ("*", "#2a78d6", (10, -3)),
        "bitgolf-small": ("P", "#eb6834", (10, -3)),
    }
    for method, (marker, color, offset) in styles.items():
        picked = [
            r
            for r in rows
            if r.method == method
            and r.corpus in corpora
            and (r.threads > 1 or not method.startswith("bitgolf"))
        ]
        if not picked:
            continue
        # Pool the files by bytes (total stored bits over total raw bytes, total raw bytes over
        # total decode time) rather than averaging per-file ratios and speeds.
        raw = sum(r.raw_bytes for r in picked)
        bpb = sum(r.stored_bytes for r in picked) * 8 / raw
        timed = [r for r in picked if r.corpus not in deflate_only]
        speed = sum(r.raw_bytes for r in timed) / sum(r.raw_bytes / r.decode_kb_s for r in timed)
        size = 220 if marker == "*" else 110  # a star reads small at the same area
        ax.scatter(
            [speed], [bpb], marker=marker, s=size, color=color, edgecolors=surface, label=method
        )
        ax.annotate(
            method,
            (speed, bpb),
            textcoords="offset points",
            xytext=offset,
            ha="left" if offset[0] > 0 else "right",
            fontsize=8,
            color=ink if method.startswith("bitgolf") else muted,
        )
    ax.set_xscale("log")
    ax.set_xlim(1, 1e6)
    ax.set_xlabel("decode throughput, KB/s, log scale (bitgolf 8 threads, baselines 1)", color=ink)
    ax.set_ylabel("bits per byte (lower is better)", color=ink)
    title = f"Ratio vs decode speed, pooled over\n{', '.join(corpora)}"
    if deflate_only:
        title += (
            f"\n(speed leaves out {', '.join(deflate_only)}: "
            "every bitgolf block there fell back to deflate)"
        )
    ax.set_title(title, fontsize=10, color=ink)
    ax.grid(True, which="major", color=muted, alpha=0.15)
    ax.tick_params(colors=muted)
    for spine in ax.spines.values():
        spine.set_color(muted)
        spine.set_alpha(0.4)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, facecolor=surface)


def _record(rows_path: Path, row: Row) -> None:
    with rows_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row)) + "\n")
    print(json.dumps(asdict(row)), flush=True)


def report(rows_path: Path, out: Path, block_size: int) -> None:
    """Writes results.json, results.md and pareto.png from the recorded rows."""
    rows = [Row(**json.loads(line)) for line in rows_path.read_text().splitlines() if line]
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "cpu": "Intel Xeon Gold 6342, 8-vCPU quota on a shared pod",
        "bitgolf_build": "cargo build --release (baseline x86-64, no target-cpu flags)",
        "block_size": block_size,
    }
    payload = {
        "meta": meta,
        "rows": [asdict(r) | {"total_with_model": r.total_with_model} for r in rows],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "results.md").write_text(markdown(rows))
    pareto_plot(rows, out / "pareto.png")
    print(markdown(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["baselines", "bitgolf", "report"])
    parser.add_argument("--rows", type=Path, default=Path("../runs/bench/rows.jsonl"))
    parser.add_argument("--binary", type=Path, default=Path("../target/release/bitgolf"))
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--corpus", type=Path, nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", default=list(BASELINES))
    parser.add_argument("--threads", type=int, default=8)
    # 32 KiB blocks split a 256 KB slice into 8 blocks, one per thread of the 8-vCPU quota.
    parser.add_argument("--block-size", type=int, default=32768)
    parser.add_argument("--scaling-file", type=Path)
    parser.add_argument("--scaling-limit", type=int, default=262144)
    parser.add_argument(
        "--scaling-threads",
        type=int,
        nargs="*",
        help="thread counts for the scaling rows (default: 1 and --threads)",
    )
    parser.add_argument("--out", type=Path, default=Path("../docs/results"))
    ns = parser.parse_args()
    ns.rows.parent.mkdir(parents=True, exist_ok=True)
    if ns.stage == "baselines":
        for path in ns.corpus:
            for name in ns.methods:
                if shutil.which(BASELINES[name][0][0]) is None:
                    raise RuntimeError(f"{BASELINES[name][0][0]} is not installed")
                _record(ns.rows, run_baseline(name, path))
    elif ns.stage == "bitgolf":
        for model in ns.models:
            for path in ns.corpus:
                _record(
                    ns.rows, run_bitgolf(ns.binary, model, path, ns.threads, ns.block_size, None)
                )
            if ns.scaling_file is not None:
                for threads in ns.scaling_threads or (1, ns.threads):
                    row = run_bitgolf(
                        ns.binary, model, ns.scaling_file, threads, ns.block_size, ns.scaling_limit
                    )
                    _record(ns.rows, row)
    else:
        report(ns.rows, ns.out, ns.block_size)


if __name__ == "__main__":
    main()
