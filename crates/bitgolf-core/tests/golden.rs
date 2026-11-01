//! Determinism goldens shared with the wasm32 test (`crates/bitgolf-wasm/tests/determinism.mjs`).
//!
//! For each model, the sha256 of every logit's bit pattern and of the compressed container
//! must match `tests/fixtures/goldens.json` exactly. The same file is checked by the wasm
//! build under Node, so passing both proves bit-identical inference across targets.
//! Run with `BITGOLF_BLESS=1` to rewrite the goldens after an intentional change.

use std::path::PathBuf;

use bitgolf_core::codec::{self, CompressOptions};
use bitgolf_core::error::hex;
use bitgolf_core::{Model, compress};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Serialize, Deserialize)]
struct Case {
    model: String,
    block_size: usize,
    logits_sha256: String,
    container_sha256: String,
}

#[derive(Serialize, Deserialize)]
struct Goldens {
    text: String,
    cases: Vec<Case>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn logits_and_containers_match_goldens() {
    let path = root().join("crates/bitgolf-core/tests/fixtures/goldens.json");
    let mut goldens: Goldens = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let text = std::fs::read(root().join(&goldens.text)).unwrap();
    let bless = std::env::var_os("BITGOLF_BLESS").is_some();
    for case in &mut goldens.cases {
        let model = Model::from_bytes(&std::fs::read(root().join(&case.model)).unwrap()).unwrap();
        let logits = hex(&codec::logits_digest(&model, &text));
        let packed =
            compress(&model, &text, &CompressOptions { block_size: case.block_size, preamble: Vec::new() })
                .unwrap();
        let container = hex(&Sha256::digest(&packed));
        if bless {
            case.logits_sha256 = logits;
            case.container_sha256 = container;
        } else {
            assert_eq!(logits, case.logits_sha256, "logits digest for {}", case.model);
            assert_eq!(container, case.container_sha256, "container digest for {}", case.model);
        }
    }
    if bless {
        std::fs::write(&path, serde_json::to_string_pretty(&goldens).unwrap() + "\n").unwrap();
    }
}
