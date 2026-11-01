"""Fetch and assemble the evaluation corpora. These files are never used for training.

enwik8     first 10^8 bytes of a 2006 English Wikipedia XML dump (Hutter Prize)
alice29, lcet10, plrabn12   Canterbury corpus text files
dickens    Silesia corpus
code       held-out Go `net/http` and CPython `asyncio` sources (see prepare.HELD_OUT_CODE)
jsonlog    npm's bundled package.json manifests (JSON) followed by the pod's dpkg and apt logs
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

from .prepare import HELD_OUT_CODE, _go_root

URLS = {
    "enwik8": "http://mattmahoney.net/dc/enwik8.zip",
    "cantrbry": "https://corpus.canterbury.ac.nz/resources/cantrbry.zip",
    "dickens": "https://sun.aei.polsl.pl/~sdeor/corpus/dickens.bz2",
}
EXPECTED_SIZES = {
    "enwik8": 100_000_000,
    "alice29": 152_089,
    "lcet10": 426_754,
    "plrabn12": 481_861,
    "dickens": 10_192_446,
}


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(request, timeout=300) as resp:
        data: bytes = resp.read()
    return data


def _code_sample() -> bytes:
    go_dir = _go_root() / "src" / HELD_OUT_CODE[0]
    py_dir = Path(sysconfig.get_paths()["stdlib"]) / HELD_OUT_CODE[1]
    parts = [p.read_bytes() for p in sorted(go_dir.glob("*.go")) if not p.name.endswith("_test.go")]
    parts += [p.read_bytes() for p in sorted(py_dir.glob("*.py"))]
    return b"\n".join(parts)


def _jsonlog_sample(node_root: Path) -> bytes:
    manifests = sorted(node_root.rglob("package.json"))
    lines = []
    for path in manifests:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        lines.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    logs = [Path("/var/log/dpkg.log"), Path("/var/log/apt/history.log")]
    blob = "\n".join(lines).encode("utf-8") + b"\n"
    return blob + b"".join(p.read_bytes() for p in logs if p.exists())


def fetch(out: Path, node_root: Path) -> dict[str, dict[str, object]]:
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(_download(URLS["enwik8"]))) as zf:
        files["enwik8"] = zf.read("enwik8")
    with zipfile.ZipFile(io.BytesIO(_download(URLS["cantrbry"]))) as zf:
        for name in ("alice29", "lcet10", "plrabn12"):
            files[name] = zf.read(f"{name}.txt")
    files["dickens"] = bz2.decompress(_download(URLS["dickens"]))
    files["code"] = _code_sample()
    files["jsonlog"] = _jsonlog_sample(node_root)
    report: dict[str, dict[str, object]] = {}
    for name, data in files.items():
        expected = EXPECTED_SIZES.get(name)
        if expected is not None and len(data) != expected:
            raise RuntimeError(f"{name}: expected {expected} bytes, downloaded {len(data)}")
        (out / name).write_bytes(data)
        report[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (out / "corpora.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch evaluation corpora")
    parser.add_argument("--out", type=Path, default=Path("data/corpora"))
    parser.add_argument(
        "--node-root",
        type=Path,
        required=True,
        help="directory containing npm's bundled node_modules",
    )
    ns = parser.parse_args()
    print(json.dumps(fetch(ns.out, ns.node_root), indent=2))


if __name__ == "__main__":
    main()
