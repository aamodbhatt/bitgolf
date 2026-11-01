//! Shared helpers for integration tests.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use bitgolf_core::Model;

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name)
}

pub fn tiny_model() -> Arc<Model> {
    let bytes = std::fs::read(fixture("tiny.bgm")).expect("fixture model");
    Arc::new(Model::from_bytes(&bytes).expect("valid fixture model"))
}

/// Mixed text with some repetition, long enough for several blocks.
pub fn sample_text(len: usize) -> Vec<u8> {
    let parts = [
        "The range coder spends -log2 p bits on every byte. ",
        "{\"user\":\"ada\",\"id\":1815,\"tags\":[\"math\",\"engines\"]}\n",
        "2025-03-01T12:00:00Z INFO request completed status=200 ms=12\n",
        "fn decode(&mut self) -> u8 { self.model.step() }\n",
    ];
    let mut out = Vec::with_capacity(len);
    let mut i = 0usize;
    while out.len() < len {
        out.extend_from_slice(parts[i % parts.len()].as_bytes());
        i = i.wrapping_mul(31).wrapping_add(7);
    }
    out.truncate(len);
    out
}

/// Deterministic pseudo-random bytes (xorshift), standing in for binary data.
pub fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}
