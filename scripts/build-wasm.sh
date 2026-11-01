#!/usr/bin/env bash
# Build the wasm32 package into web/pkg. Needs the wasm32-unknown-unknown target and
# wasm-bindgen-cli 0.2.130 (`cargo install wasm-bindgen-cli --version 0.2.130 --locked`).
# Set SIMD=1 to enable wasm SIMD128 (results are bit-identical either way).
set -euo pipefail
cd "$(dirname "$0")/.."
flags=""
if [ "${SIMD:-0}" = "1" ]; then flags="-C target-feature=+simd128"; fi
RUSTFLAGS="$flags" cargo build --release -p bitgolf-wasm --target wasm32-unknown-unknown --target-dir target/wasm
wasm-bindgen --target web --no-typescript --out-dir "${OUT:-web/pkg}" \
  target/wasm/wasm32-unknown-unknown/release/bitgolf_wasm.wasm
echo "wrote ${OUT:-web/pkg}"
