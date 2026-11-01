"""Build the training mixture: prose, permissively licensed code, and structured text.

Sources and licenses (see docs/DATA.md):
  * prose       FineWeb-Edu `sample/10BT` shard 0 (ODC-By 1.0)
  * code        Go standard library (BSD-3-Clause) and CPython standard library (PSF-2.0)
  * structured  JSON Lines and CSV rendered from FineWeb-Edu row metadata (ODC-By 1.0), plus
                synthetic logs from a seeded generator in this module

Evaluation files (enwik8, Canterbury, Silesia, the held-out code and JSON/log samples) are never
read here. The held-out code packages are excluded by path, and `HELD_OUT_CODE` is shared with
`corpora.py` so the split cannot drift.

Output: `<out>/{prose,code,structured}_{train,val}.bin` (raw bytes, documents separated by NUL)
and `<out>/manifest.json`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import sysconfig
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DOC_SEPARATOR = b"\x00"
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_FILE = "sample/10BT/000_00000.parquet"
# Packages reserved for the code evaluation sample; never part of training.
HELD_OUT_CODE = ("net/http", "asyncio")


@dataclass(frozen=True)
class PrepareArgs:
    out: Path
    prose_mb: int
    code_mb: int
    structured_mb: int
    val_fraction: float
    seed: int


def _write_split(name: str, docs: list[bytes], args: PrepareArgs) -> dict[str, object]:
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_fraction))
    splits = {"val": docs[:n_val], "train": docs[n_val:]}
    info: dict[str, object] = {"documents": len(docs)}
    for split, items in splits.items():
        payload = DOC_SEPARATOR.join(items) + DOC_SEPARATOR
        path = args.out / f"{name}_{split}.bin"
        path.write_bytes(payload)
        info[f"{split}_bytes"] = len(payload)
        info[f"{split}_sha256"] = hashlib.sha256(payload).hexdigest()
    return info


def _fineweb_rows(limit_bytes: int) -> Iterator[dict[str, object]]:
    path = hf_hub_download(FINEWEB_REPO, FINEWEB_FILE, repo_type="dataset")
    columns = ["text", "id", "url", "dump", "date", "language", "score", "int_score", "token_count"]
    total = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            yield row
            total += len(str(row["text"]).encode("utf-8"))
            if total >= limit_bytes:
                return


def build_prose_and_metadata(args: PrepareArgs) -> tuple[list[bytes], list[dict[str, object]]]:
    docs: list[bytes] = []
    meta: list[dict[str, object]] = []
    for row in _fineweb_rows(args.prose_mb << 20):
        text = str(row["text"]).replace("\x00", "")
        docs.append(text.encode("utf-8"))
        first_line = text.strip().split("\n", 1)[0][:120]
        meta.append({k: row[k] for k in row if k != "text"} | {"title": first_line})
    return docs, meta


def _go_root() -> Path:
    out = subprocess.run(["go", "env", "GOROOT"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _source_files(root: Path, suffix: str) -> list[Path]:
    files = []
    for path in sorted(root.rglob(f"*{suffix}")):
        rel = path.relative_to(root).as_posix()
        if "testdata" in rel or "/test/" in f"/{rel}" or "__pycache__" in rel:
            continue
        if any(rel.startswith(f"{held}/") for held in HELD_OUT_CODE):
            continue
        files.append(path)
    return files


def build_code(args: PrepareArgs) -> tuple[list[bytes], dict[str, object]]:
    roots = {
        "go-stdlib (BSD-3-Clause)": (_go_root() / "src", ".go"),
        "cpython-stdlib (PSF-2.0)": (Path(sysconfig.get_paths()["stdlib"]), ".py"),
    }
    docs: list[bytes] = []
    counts: dict[str, object] = {}
    budget = args.code_mb << 20
    per_root = budget // len(roots)
    for label, (root, suffix) in roots.items():
        used = 0
        files = _source_files(root, suffix)
        random.Random(args.seed).shuffle(files)
        n = 0
        for path in files:
            data = path.read_bytes().replace(b"\x00", b"")
            if used + len(data) > per_root:
                continue
            docs.append(data)
            used += len(data)
            n += 1
        counts[label] = {"files": n, "bytes": used}
    return docs, counts


_PATHS = [
    "/",
    "/index.html",
    "/api/v1/users",
    "/api/v1/orders",
    "/api/v1/items/{id}",
    "/static/app.{h}.js",
    "/static/style.{h}.css",
    "/login",
    "/search?q={q}",
    "/healthz",
    "/metrics",
    "/api/v2/sessions/{id}/events",
]
_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "curl/8.5.0",
    "python-requests/2.31.0",
    "Go-http-client/1.1",
]
_SERVICES = ["api-gateway", "order-service", "billing", "auth", "search-indexer", "scheduler"]
_MESSAGES = [
    "request completed",
    "cache miss",
    "retrying upstream call",
    "user authenticated",
    "job scheduled",
    "connection reset by peer",
    "slow query detected",
    "payment authorized",
    "rate limit exceeded",
    "shutting down worker",
]
_WORDS = ["alpha", "beta", "gamma", "delta", "report", "invoice", "kernel", "matrix", "river"]


def _synthetic_log_doc(rng: random.Random) -> bytes:
    """One synthetic log file in a randomly chosen common format."""
    lines: list[str] = []
    style = rng.choice(["apache", "app", "json"])
    t = rng.randrange(1_700_000_000, 1_760_000_000)
    for _ in range(rng.randrange(50, 400)):
        t += rng.randrange(0, 5)
        day, sec = divmod(t, 86400)
        hh, rem = divmod(sec, 3600)
        mm, ss = divmod(rem, 60)
        stamp = f"2025-{1 + day % 12:02d}-{1 + day % 28:02d}T{hh:02d}:{mm:02d}:{ss:02d}"
        ms = rng.randrange(1000)
        path = (
            rng.choice(_PATHS)
            .replace("{id}", str(rng.randrange(1, 99999)))
            .replace("{h}", f"{rng.getrandbits(32):08x}")
            .replace("{q}", rng.choice(_WORDS))
        )
        status = rng.choice([200] * 12 + [201, 204, 301, 304, 400, 401, 403, 404, 500, 503])
        octets = (
            rng.randrange(1, 255),
            rng.randrange(256),
            rng.randrange(256),
            rng.randrange(1, 255),
        )
        ip = ".".join(map(str, octets))
        method = rng.choice(["GET"] * 6 + ["POST", "PUT", "DELETE"])
        latency = round(rng.expovariate(1 / 40), 1)
        if style == "apache":
            lines.append(
                f'{ip} - - [{stamp}+0000] "{method} {path} HTTP/1.1" {status} '
                f'{rng.randrange(0, 90000)} "-" "{rng.choice(_AGENTS)}"'
            )
        elif style == "app":
            level = rng.choice(["INFO"] * 8 + ["WARN", "ERROR", "DEBUG"])
            lines.append(
                f"{stamp}.{ms:03d}Z {level:<5} [worker-{rng.randrange(16)}] "
                f"{rng.choice(_SERVICES)}: {rng.choice(_MESSAGES)} "
                f"id={rng.randrange(1, 10**6)} latency_ms={latency}"
            )
        else:
            lines.append(
                json.dumps(
                    {
                        "ts": f"{stamp}.{ms:03d}Z",
                        "level": rng.choice(["info", "warn", "error"]),
                        "service": rng.choice(_SERVICES),
                        "msg": rng.choice(_MESSAGES),
                        "method": method,
                        "path": path,
                        "status": status,
                        "duration_ms": latency,
                    },
                    separators=(",", ":"),
                )
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_structured(meta: list[dict[str, object]], args: PrepareArgs) -> list[bytes]:
    rng = random.Random(args.seed + 1)
    docs: list[bytes] = []
    budget = args.structured_mb << 20
    used = 0
    third = budget // 3
    # JSON Lines and CSV documents of 50-300 metadata rows each.
    for fmt in ("jsonl", "csv"):
        produced = 0
        cursor = 0
        while produced < third and meta:
            n = rng.randrange(50, 300)
            rows = [meta[(cursor + i) % len(meta)] for i in range(n)]
            cursor += n
            if fmt == "jsonl":
                text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            else:
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                text = buf.getvalue()
            data = text.encode("utf-8")
            docs.append(data)
            produced += len(data)
        used += produced
    while used < budget:
        data = _synthetic_log_doc(rng)
        docs.append(data)
        used += len(data)
    return docs


def prepare(args: PrepareArgs) -> dict[str, object]:
    args.out.mkdir(parents=True, exist_ok=True)
    prose, meta = build_prose_and_metadata(args)
    code, code_counts = build_code(args)
    structured = build_structured(meta, args)
    manifest: dict[str, object] = {
        "seed": args.seed,
        "held_out_code_packages": list(HELD_OUT_CODE),
        "prose": {"source": f"{FINEWEB_REPO}/{FINEWEB_FILE}", "license": "ODC-By-1.0"}
        | _write_split("prose", prose, args),
        "code": {"sources": code_counts} | _write_split("code", code, args),
        "structured": {
            "sources": [
                "fineweb-edu metadata as JSONL/CSV (ODC-By-1.0)",
                "synthetic logs (seeded generator)",
            ]
        }
        | _write_split("structured", structured, args),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/train"))
    parser.add_argument("--prose-mb", type=int, default=600)
    parser.add_argument("--code-mb", type=int, default=120)
    parser.add_argument("--structured-mb", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    ns = parser.parse_args()
    manifest = prepare(
        PrepareArgs(ns.out, ns.prose_mb, ns.code_mb, ns.structured_mb, ns.val_fraction, ns.seed)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
