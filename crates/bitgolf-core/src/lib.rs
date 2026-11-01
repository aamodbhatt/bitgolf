//! bitgolf-core: neural text storage with bit-exact inference.
//!
//! A small byte-level transformer predicts each next byte; a range coder spends
//! `-log2 p(byte)` bits on it. Inference is integer matrix products plus a handful of float
//! operations in a fixed order with no libm, so the same model produces bit-identical
//! predictions on x86-64 and wasm32, and a file compressed on one decodes on the other.
//!
//! * [`model`]: `.bgm` loading and deterministic incremental inference
//! * [`freq`] and [`rangecoder`]: probabilities to bits and back
//! * [`container`]: the `.bg` block container
//! * [`codec`]: compress, decompress, random access and per-byte costs
//! * [`store`]: an append-only, crash-safe key/value store built on the codec

#![forbid(unsafe_code)]

pub mod codec;
pub mod container;
pub mod error;
pub mod freq;
pub mod math;
pub mod model;
pub mod rangecoder;
pub mod store;

pub use codec::{CompressOptions, compress, decompress};
pub use error::{Error, Result};
pub use model::Model;
pub use store::Store;
