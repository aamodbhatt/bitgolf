//! Parser for the versioned `.bgm` model file.
//!
//! Layout (little-endian): 8-byte magic `BGMODEL\0`, `u32` format version, `u32` JSON header
//! length, the UTF-8 JSON header, zero padding to 64 bytes, the tensor data (each tensor
//! 64-byte aligned, offsets relative to the start of the data section), and a trailing sha256
//! over everything before it.

use std::collections::HashMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const MAGIC: &[u8; 8] = b"BGMODEL\0";
const FORMAT_VERSION: u32 = 1;
const ALIGN: usize = 64;
const MAX_HEADER: usize = 1 << 20;

/// Architecture hyper-parameters stored in the model header.
#[derive(Debug, Clone, PartialEq, Deserialize, serde::Serialize)]
pub struct ModelConfig {
    /// Human-readable preset name, e.g. `small`.
    pub name: String,
    /// Width of the residual stream.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of shared key/value heads (multi-query when 1).
    pub n_kv_heads: usize,
    /// Hidden width of the ReLU^2 MLP.
    pub d_mlp: usize,
    /// Sliding attention window in bytes, including the current byte.
    pub window: usize,
    /// Vocabulary size; always 256 for byte-level models.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
}

impl ModelConfig {
    /// Width of one attention head.
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Output width of the fused query/key/value projection.
    pub fn qkv_dim(&self) -> usize {
        self.d_model + 2 * self.n_kv_heads * self.d_head()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(Error::InvalidModel(msg));
        if self.vocab != 256 {
            return bad(format!("vocab must be 256, got {}", self.vocab));
        }
        if self.d_model == 0
            || self.d_model > crate::math::EXACT_DOT_MAX_LEN
            || self.n_heads == 0
            || self.n_kv_heads == 0
        {
            return bad("model dimensions out of range".into());
        }
        if self.d_model % self.n_heads != 0 || self.n_heads % self.n_kv_heads != 0 {
            return bad("d_model must divide into n_heads, and n_heads into n_kv_heads".into());
        }
        if self.n_layers == 0
            || self.n_layers > 64
            || self.d_mlp == 0
            || self.d_mlp > crate::math::EXACT_DOT_MAX_LEN
        {
            return bad("layer count or MLP width out of range".into());
        }
        if self.window < 2 || self.window > 1 << 16 {
            return bad(format!("window {} out of range", self.window));
        }
        if !(self.norm_eps > 0.0 && self.norm_eps < 1.0) {
            return bad(format!("norm_eps {} out of range", self.norm_eps));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    nbytes: usize,
}

#[derive(Deserialize)]
struct Header {
    config: ModelConfig,
    tensors: Vec<TensorEntry>,
    #[serde(default)]
    meta: serde_json::Value,
}

/// A tensor as stored in the file.
pub(crate) enum Tensor {
    I8 { shape: Vec<usize>, data: Vec<i8> },
    F32 { shape: Vec<usize>, data: Vec<f32> },
}

/// A parsed, checksum-verified model file.
pub(crate) struct ModelFile {
    pub config: ModelConfig,
    pub meta: serde_json::Value,
    pub tensors: HashMap<String, Tensor>,
    /// sha256 of the complete file: the model id recorded in containers.
    pub id: [u8; 32],
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

pub(crate) fn parse(bytes: &[u8]) -> Result<ModelFile> {
    let bad = |msg: &str| Error::InvalidModel(msg.to_string());
    if bytes.len() < 16 + 32 || &bytes[..8] != MAGIC {
        return Err(bad("not a bitgolf model (bad magic)"));
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if Sha256::digest(body).as_slice() != digest {
        return Err(bad("checksum mismatch: the model file is corrupt or truncated"));
    }
    let version = read_u32(bytes, 8);
    if version != FORMAT_VERSION {
        return Err(Error::InvalidModel(format!(
            "unsupported model format version {version} (this build reads {FORMAT_VERSION})"
        )));
    }
    let json_len = read_u32(bytes, 12) as usize;
    if json_len > MAX_HEADER || 16 + json_len > body.len() {
        return Err(bad("header length out of range"));
    }
    let header: Header = serde_json::from_slice(&body[16..16 + json_len])
        .map_err(|e| Error::InvalidModel(format!("bad JSON header: {e}")))?;
    header.config.validate()?;
    let data_start = (16 + json_len).div_ceil(ALIGN) * ALIGN;
    let mut tensors = HashMap::new();
    for entry in header.tensors {
        let start = data_start
            .checked_add(entry.offset)
            .filter(|s| s.checked_add(entry.nbytes).is_some_and(|end| end <= body.len()))
            .ok_or_else(|| Error::InvalidModel(format!("tensor {} out of bounds", entry.name)))?;
        let raw = &body[start..start + entry.nbytes];
        let count: usize = entry.shape.iter().product();
        let tensor = match entry.dtype.as_str() {
            "i8" if raw.len() == count => {
                Tensor::I8 { shape: entry.shape, data: raw.iter().map(|&b| b as i8).collect() }
            }
            "f32" if raw.len() == count * 4 => {
                let data: Vec<f32> =
                    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                if data.iter().any(|v| !v.is_finite()) {
                    return Err(Error::InvalidModel(format!("tensor {} is not finite", entry.name)));
                }
                Tensor::F32 { shape: entry.shape, data }
            }
            other => {
                return Err(Error::InvalidModel(format!(
                    "tensor {}: dtype {other} does not match its size",
                    entry.name
                )));
            }
        };
        tensors.insert(entry.name, tensor);
    }
    let id: [u8; 32] = Sha256::digest(bytes).into();
    Ok(ModelFile { config: header.config, meta: header.meta, tensors, id })
}
