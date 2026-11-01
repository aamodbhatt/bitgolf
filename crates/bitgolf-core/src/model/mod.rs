//! Int8 byte-level transformer: weights, loading and validation.
//!
//! Inference lives in [`engine`]; the file format in [`format`].

mod engine;
mod format;
mod kernels;

use std::collections::HashMap;

pub use engine::{Stream, Workspace};
pub use format::ModelConfig;

use crate::error::{Error, Result, hex};
use format::Tensor;

/// Int8 matrix `[rows, cols]` (row-major) with one float scale per row. The int8 values are
/// held as `f32` so products run on float SIMD while staying exact.
pub(crate) struct QMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub scales: Vec<f32>,
}

pub(crate) struct Layer {
    pub attn_norm: Vec<f32>,
    pub wqkv: QMatrix,
    pub wo: QMatrix,
    pub mlp_norm: Vec<f32>,
    pub w_in: QMatrix,
    pub w_out: QMatrix,
}

/// A loaded, validated model. Immutable and shareable across threads.
pub struct Model {
    config: ModelConfig,
    id: [u8; 32],
    file_len: usize,
    meta: serde_json::Value,
    /// Dequantised embedding table `[256, d_model]`.
    pub(crate) tok_emb: Vec<f32>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) lm_head: QMatrix,
    pub(crate) slopes: Vec<f32>,
    pub(crate) inv_sqrt_d_head: f32,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.config.name)
            .field("id", &self.id_hex())
            .finish_non_exhaustive()
    }
}

struct TensorSet(HashMap<String, Tensor>);

impl TensorSet {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        self.0.remove(name).ok_or_else(|| Error::InvalidModel(format!("missing tensor {name}")))
    }

    fn f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        match self.take(name)? {
            Tensor::F32 { shape: s, data } if s == shape => Ok(data),
            _ => Err(Error::InvalidModel(format!("tensor {name}: expected f32 {shape:?}"))),
        }
    }

    fn qmatrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<QMatrix> {
        let data = match self.take(name)? {
            Tensor::I8 { shape, data } if shape == [rows, cols] => data,
            _ => {
                return Err(Error::InvalidModel(format!("tensor {name}: expected i8 [{rows}, {cols}]")));
            }
        };
        let scales = self.f32(&format!("{name}.scale"), &[rows])?;
        if scales.iter().any(|s| *s < 0.0) {
            return Err(Error::InvalidModel(format!("tensor {name}: negative scale")));
        }
        let data = data.into_iter().map(f32::from).collect();
        Ok(QMatrix { rows, cols, data, scales })
    }
}

impl Model {
    /// Parses and validates a `.bgm` file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let file = format::parse(bytes)?;
        let cfg = file.config;
        let d = cfg.d_model;
        let mut t = TensorSet(file.tensors);
        let emb = t.qmatrix("tok_emb", cfg.vocab, d)?;
        let tok_emb = emb
            .data
            .chunks_exact(d)
            .zip(&emb.scales)
            .flat_map(|(row, &s)| row.iter().map(move |&q| q * s))
            .collect();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            layers.push(Layer {
                attn_norm: t.f32(&format!("{p}.attn_norm"), &[d])?,
                wqkv: t.qmatrix(&format!("{p}.wqkv"), cfg.qkv_dim(), d)?,
                wo: t.qmatrix(&format!("{p}.wo"), d, d)?,
                mlp_norm: t.f32(&format!("{p}.mlp_norm"), &[d])?,
                w_in: t.qmatrix(&format!("{p}.w_in"), cfg.d_mlp, d)?,
                w_out: t.qmatrix(&format!("{p}.w_out"), d, cfg.d_mlp)?,
            });
        }
        let final_norm = t.f32("final_norm", &[d])?;
        let lm_head = t.qmatrix("lm_head", cfg.vocab, d)?;
        let slopes = t.f32("alibi_slopes", &[cfg.n_heads])?;
        if let Some(extra) = t.0.keys().next() {
            return Err(Error::InvalidModel(format!("unexpected tensor {extra}")));
        }
        let inv_sqrt_d_head = crate::math::rsqrt(cfg.d_head() as f32);
        Ok(Self {
            config: cfg,
            id: file.id,
            file_len: bytes.len(),
            meta: file.meta,
            tok_emb,
            layers,
            final_norm,
            lm_head,
            slopes,
            inv_sqrt_d_head,
        })
    }

    /// Architecture hyper-parameters.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// sha256 of the model file; containers record it so data is never decoded with the
    /// wrong model.
    pub fn id(&self) -> [u8; 32] {
        self.id
    }

    /// Hex form of [`Model::id`].
    pub fn id_hex(&self) -> String {
        hex(&self.id)
    }

    /// Size of the `.bgm` file in bytes (what a user must ship next to the data).
    pub fn file_len(&self) -> usize {
        self.file_len
    }

    /// Training metadata embedded in the model file (free-form JSON).
    pub fn meta(&self) -> &serde_json::Value {
        &self.meta
    }

    /// Number of stored weights (int8 and float).
    pub fn num_params(&self) -> usize {
        let q = |m: &QMatrix| m.rows * m.cols;
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| q(&l.wqkv) + q(&l.wo) + q(&l.w_in) + q(&l.w_out) + 2 * self.config.d_model)
            .sum();
        self.tok_emb.len() + per_layer + self.final_norm.len() + q(&self.lm_head)
    }
}
