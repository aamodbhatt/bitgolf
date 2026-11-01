#!/usr/bin/env bash
# Full local verification: formatting, lints, Rust tests, wasm determinism, Python checks.
# Needs: Rust stable with the wasm32-unknown-unknown target, wasm-bindgen-cli 0.2.130,
# Node 18+, and uv.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== rust: fmt, clippy, tests"
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

echo "== wasm32: determinism against native goldens (scalar, then SIMD128)"
OUT=target/pkg bash scripts/build-wasm.sh
node crates/bitgolf-wasm/tests/determinism.mjs target/pkg
SIMD=1 OUT=target/pkg-simd bash scripts/build-wasm.sh
node crates/bitgolf-wasm/tests/determinism.mjs target/pkg-simd

echo "== python: ruff, mypy, pytest (CPU)"
cd training
uv run ruff check .
uv run ruff format --check .
uv run mypy
CUDA_VISIBLE_DEVICES= uv run pytest -q
