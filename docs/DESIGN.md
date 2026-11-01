# bitgolf design

bitgolf stores text by predicting it. A byte-level transformer outputs a probability for each
of the 256 possible next bytes; a range coder then spends `-log2 p(actual byte)` bits on it.
Everything interesting in the system follows from one hard requirement: **the decoder must
reproduce the encoder's probabilities exactly, on every machine**. A single differing bit in
one logit changes a frequency table, desynchronises the range coder, and turns the rest of the
block into garbage.

## Goals

- Compress text well below classical compressors on the kinds of text the model knows.
- Bit-exact inference across x86-64 (any SIMD level) and wasm32 (scalar and SIMD128), so a
  `.bg` file written in a browser decodes natively and vice versa.
- Random access: any block decodes on its own; a store `get` decodes one block prefix.
- Never worse than deflate on data the model does not understand (binary, noise).
- Corruption is always reported, never turned into wrong output.

## Non-goals

- Beating cmix/nncp on enwik8. Those adapt to the file while coding and run for hours or days;
  bitgolf models are small, frozen, and general.
- GPU inference. Determinism across vendors' GPU kernels is a different (much harder) project.
- Mutable stores. The store is append-only; deletes and compaction are on the roadmap.

## Determinism strategy

Floating point is deterministic when (a) only correctly rounded IEEE-754 operations are used,
(b) the order of operations is fixed, and (c) nothing calls a platform math library.

| Concern | Choice |
|---|---|
| Transcendentals | `exp` is a Cody-Waite range reduction plus a degree-6 polynomial in plain `f32` (`math::exp_nonpos`); `rsqrt` is a bit-level guess plus three Newton steps; `log2` (reporting only) is an atanh series. No libm anywhere. |
| FMA contraction | Rust never fuses `a * b + c` without explicit `mul_add`, so builds with or without FMA hardware compute the same values. Verified by running the goldens on an `x86-64-v3` (AVX2/FMA) build. |
| Matrix products | Weights and activations are int8 values. They are stored as `f32` integers and multiplied in `f32`: every product is at most 127^2 and every partial sum stays below 2^24 for vectors up to 1040 long (`EXACT_DOT_MAX_LEN`), so the sum is exact in any order. This gives wide float SIMD with 4 independent accumulators and no ordering constraint at all. |
| Other reductions | RMSNorm sums of squares and the softmax denominator use a fixed 8-lane interleave plus a scalar tail, so SIMD and scalar builds add the same numbers in the same order. |
| Rounding | `round_ties_even` is the `1.5 * 2^23` add/subtract trick, matching `torch.round`, instead of `roundevenf` (a libcall on baseline x86-64). |
| Attention order | Keys are visited in distance order (newest first) regardless of KV ring capacity, so a rewindable browser session and the codec compute identical sums. |
| Batching | Encoders and decoders batch blocks differently (lockstep groups, any thread count). Every output element is produced by the same per-row recipe, so batch shape never matters. |
| Probabilities to integers | `FreqTable::from_logits` uses the deterministic `exp`, an `f64` scale with truncation, a floor of 1 per symbol, and gives the remainder to the most likely symbol (lowest index on ties). |

The proof is empirical and runs in CI: `tests/golden.rs` hashes every logit's bit pattern and
the compressed bytes for three models; `crates/bitgolf-wasm/tests/determinism.mjs` checks the
same hashes from the wasm build under Node, both scalar and with SIMD128. The browser demo
repeats the check in Chromium during screenshot capture.

## Model

| | nano | small |
|---|---|---|
| parameters | 0.75 M | 4.26 M |
| layers x width | 4 x 128 | 6 x 256 |
| heads (query / key-value) | 4 / 1 | 4 / 1 |
| MLP | ReLU^2, 4x | ReLU^2, 4x |
| attention window | 512 bytes | 512 bytes |
| `.bgm` size | 0.78 MB | 4.33 MB |

- **RMSNorm** instead of LayerNorm: one reduction instead of two and no mean subtraction.
- **ALiBi** instead of rotary embeddings: a linear distance penalty needs no sin/cos tables
  and only depends on relative distance, which is all a ring-buffer KV cache knows.
- **ReLU^2 MLP**: as strong as gated units at this scale (Primer) with no transcendental.
- **Multi-query attention**: one shared key/value head makes the per-stream cache 4x smaller,
  which matters because decoding many blocks in lockstep is bound by cache traffic.
- **Sliding-window training**: training sequences (1024 bytes) are twice the window, with a
  banded causal mask. Positions past the first window never see the sequence start, which is
  exactly the situation of a ring-buffer cache at inference. A model trained only on
  start-anchored sequences learns to lean on the first tokens ("attention sinks") and degrades
  once they are evicted.
- **int8 quantisation**: weights per output row; activations per token at every linear layer
  input; queries, keys and values per head. The training fake-quant (`quant.py`) and the Rust
  kernels use the same formula, `scale = max|x| / 127`, `q = round_ties_even(x / scale)`.
  Measured cost on validation data: +0.0006 bits/byte (small) and +0.0019 (nano) before
  fine-tuning; the quantisation-aware fine-tune recovers it.

## Formats

### `.bgm` model

`BGMODEL\0`, `u32` version, `u32` JSON length, a JSON header (config, tensor directory,
training metadata), 64-byte aligned little-endian tensors (int8 with a float32 scale per row,
or float32), and a trailing sha256 over everything before it. The sha256 of the whole file is
the **model id**; containers and stores record it and refuse to decode with any other model.
The loader validates shapes, finiteness, scale signs and dimension limits (including the
exact-dot bound above).

### `.bg` container

```
header   "BITGOLF\0" | version u16 | flags u16 | model sha256 | window u32 | block_size u32
         | preamble_len u32 | preamble | crc32c(header)
blocks   independent payloads, back to back
index    count u32 | count x (codec u8, raw_len u32, offset u64, len u32,
         crc32c(raw), crc32c(payload)) | crc32c(index)
trailer  index_offset u64 | total_raw_len u64 | "BGINDEX\0"
```

- **Independent blocks** (64 KiB by default) give random access (`bitgolf cat --block N`) and
  parallelism. Each block starts from a cold model, optionally primed with the preamble; the
  primed KV state is computed once and cloned, so priming is free per block.
- **Per-block codec**: `model`, `deflate` (pure-Rust miniz_oxide) or `stored`, whichever is
  smallest. Model coding of a block is abandoned early if it averages more than 7 bits/byte
  after any 4 KiB, so binary data costs little time.
- **Two checksums per block**: the payload CRC fails fast before spending model time; the raw
  CRC catches anything else, including a model or engine mismatch.
- The index is at the end so blocks can be written as they finish; the fixed-size trailer
  locates it. Every field is bounds-checked; a property test flips random bits and truncates
  files and requires an error (never different output).

### Store

`data.bgs` is a sequence of frames, each one compressed block plus its key table (key, value
offset, length, CRC32C per value); `MANIFEST` records the model id, block size, preamble and
the committed length of `data.bgs`.

A flush appends the frame, `fsync`s the data file, writes `MANIFEST.tmp`, `fsync`s it, renames
it over `MANIFEST` and `fsync`s the directory. Bytes past the committed length can only come
from an interrupted flush, so `open` truncates them. Failure modes:

| Failure | Result |
|---|---|
| crash before or during the frame append | torn tail truncated on open; store at last flush |
| crash after append, before rename | frame ignored (not committed); leftover `.tmp` deleted |
| crash after rename | new frame visible |
| bit flip in a payload | `get` of an affected key returns `Corrupt` (payload/value CRC) |
| bit flip in a key table or manifest | `open` returns `Corrupt` |
| data file shorter than committed | `open` returns `Corrupt` (data loss, not a torn write) |
| different model | `ModelMismatch` naming both ids |

`tests/store.rs` simulates the first three with real files at random cut points (property
tests) and checks that the recovered store accepts new writes. A `get` decodes only the block
prefix that ends with the value and verifies that value's CRC.

## Alternatives considered

- **Integer-only inference** (fixed-point softmax and norms). Fully deterministic by
  construction but costs accuracy and a lot of engineering; the exact-`f32` integer dot product
  gives the same guarantee where it matters (99% of the arithmetic) at full SIMD speed.
- **int8 SIMD intrinsics** (`pmaddubsw`, wasm `i32x4.dot_i16x8`). Faster, but platform
  intrinsics are exactly what portability rules out, and autovectorised `i8 -> i32` code was
  measured slower than the exact-`f32` formulation.
- **Rotary embeddings**: need sin/cos with bit-exact range reduction for large positions.
- **Arithmetic coding with 32-bit range**: with 2^20 frequency totals a 32-bit range loses
  precision; the 64-bit range coder keeps the loss below 2^-36 per symbol.
- **One model call per file** (a single stream): simpler, but no random access and no
  parallel decode. Blocks cost a cold start each (mitigated by priming).

## Known limitations

- Throughput is that of a neural compressor, KB/s rather than MB/s. On 8 shared cores the small
  model encodes at 1.6-3.0 KB/s and decodes at 1.7-3.3 KB/s; nano at 4.3-6.8 and 4.1-7.5 KB/s.
- The 512-byte window cannot see long-range repeats that LZ77 finds. On very repetitive data
  (the JSON/log sample) xz and brotli win clearly; a hybrid (LZ matches as model context) is on
  the roadmap.
- Small models trained for minutes generalise unevenly: excellent on web prose and source code
  like their training mix and ahead of every baseline on Wikipedia markup, but behind brotli
  (small) or every baseline except gzip (nano) on 19th-century literature.
