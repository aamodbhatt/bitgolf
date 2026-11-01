//! Incremental inference with a ring-buffer KV cache.
//!
//! A [`Stream`] is the decoding state of one independent byte sequence. [`Model::step_batch`]
//! advances many streams by one byte in lockstep, which turns every weight product into a
//! small matrix product with good cache reuse. Each row is still computed with the same
//! scalar recipe as a batch of one, so batching never changes a single bit of the output.

use super::kernels::{matmul, quantize_row, rms_norm};
use super::{Model, QMatrix};
use crate::error::{Error, Result};
use crate::math::{dot_exact, exp_nonpos, sum_f32};

/// Decoding state of one byte sequence: position and the int8 key/value cache.
///
/// Keys and values are quantised per position and per KV head to int8 values plus one scale,
/// exactly as in training, which keeps attention scores exact integer dot products.
#[derive(Clone)]
pub struct Stream {
    pos: u64,
    capacity: usize,
    k: Vec<f32>,
    k_scale: Vec<f32>,
    v: Vec<f32>,
    v_scale: Vec<f32>,
}

impl Stream {
    /// An empty stream whose cache holds exactly one attention window.
    pub fn new(model: &Model) -> Self {
        Self::alloc(model, model.config().window)
    }

    /// An empty stream that retains `capacity >= window` positions so it can be rewound.
    pub fn with_capacity(model: &Model, capacity: usize) -> Result<Self> {
        if capacity < model.config().window {
            return Err(Error::InvalidArgument(format!(
                "stream capacity {capacity} is smaller than the model window {}",
                model.config().window
            )));
        }
        Ok(Self::alloc(model, capacity))
    }

    fn alloc(model: &Model, capacity: usize) -> Self {
        let cfg = model.config();
        let kv_dim = cfg.n_kv_heads * cfg.d_head();
        let slots = cfg.n_layers * capacity;
        Self {
            pos: 0,
            capacity,
            k: vec![0.0; slots * kv_dim],
            k_scale: vec![0.0; slots * cfg.n_kv_heads],
            v: vec![0.0; slots * kv_dim],
            v_scale: vec![0.0; slots * cfg.n_kv_heads],
        }
    }

    /// Number of bytes consumed so far.
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// Moves back to `pos`, forgetting later bytes. Fails if the cache no longer holds the
    /// window of history that position `pos` needs.
    pub fn rewind(&mut self, pos: u64, window: usize) -> Result<()> {
        let oldest_kept = self.pos.saturating_sub(self.capacity as u64);
        let oldest_needed = pos.saturating_sub(window as u64 - 1);
        if pos > self.pos || oldest_needed < oldest_kept {
            return Err(Error::InvalidArgument(format!(
                "cannot rewind stream from {} to {pos} with capacity {}",
                self.pos, self.capacity
            )));
        }
        self.pos = pos;
        Ok(())
    }
}

/// Reusable scratch buffers for [`Model::step_batch`], sized on demand.
#[derive(Default)]
pub struct Workspace {
    x: Vec<f32>,
    h: Vec<f32>,
    hq: Vec<f32>,
    hs: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    mlp: Vec<f32>,
    proj: Vec<f32>,
    attn_scratch: AttnScratch,
}

/// Per-step attention buffers.
#[derive(Default)]
struct AttnScratch {
    slots: Vec<usize>,
    scores: Vec<f32>,
    q8: Vec<f32>,
    q_scale: Vec<f32>,
    inv_sum: Vec<f32>,
}

impl Workspace {
    /// Empty workspace; buffers grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, model: &Model, batch: usize) {
        let cfg = model.config();
        let wide = cfg.d_model.max(cfg.d_mlp);
        self.x.resize(batch * cfg.d_model, 0.0);
        self.h.resize(batch * cfg.d_model, 0.0);
        self.hq.resize(batch * wide, 0.0);
        self.hs.resize(batch, 0.0);
        self.qkv.resize(batch * cfg.qkv_dim(), 0.0);
        self.attn.resize(batch * cfg.d_model, 0.0);
        self.mlp.resize(batch * cfg.d_mlp, 0.0);
        self.proj.resize(batch * cfg.d_model.max(cfg.vocab), 0.0);
        let a = &mut self.attn_scratch;
        a.slots.reserve(cfg.window);
        a.scores.resize(cfg.n_heads * cfg.window, 0.0);
        a.q8.resize(cfg.d_model, 0.0);
        a.q_scale.resize(cfg.n_heads, 0.0);
        a.inv_sum.resize(cfg.n_heads, 0.0);
    }
}

/// Quantises each `width`-wide row of `src` into `q`/`scales`.
fn quantize_rows(src: &[f32], width: usize, q: &mut [f32], scales: &mut [f32]) {
    for ((row, qrow), s) in src.chunks_exact(width).zip(q.chunks_exact_mut(width)).zip(scales) {
        *s = quantize_row(row, qrow);
    }
}

/// `residual[b] += matmul(quantize(input[b]), w)`.
fn add_projection(
    input: &[f32],
    w: &QMatrix,
    hq: &mut [f32],
    hs: &mut [f32],
    proj: &mut [f32],
    residual: &mut [f32],
) {
    let batch = hs.len();
    let (q, out) = (&mut hq[..batch * w.cols], &mut proj[..batch * w.rows]);
    quantize_rows(input, w.cols, q, hs);
    matmul(q, hs, w, out);
    for (r, p) in residual.iter_mut().zip(out.iter()) {
        *r += *p;
    }
}

impl Model {
    /// Advances one stream by `token` and writes next-byte logits.
    pub fn step(&self, stream: &mut Stream, token: u8, ws: &mut Workspace, logits: &mut [f32]) {
        self.step_batch(&mut [stream], &[token], ws, logits);
    }

    /// Advances every stream by its token; `logits` receives `streams.len() * 256` values.
    pub fn step_batch(
        &self,
        streams: &mut [&mut Stream],
        tokens: &[u8],
        ws: &mut Workspace,
        logits: &mut [f32],
    ) {
        let cfg = self.config();
        let batch = streams.len();
        let d = cfg.d_model;
        assert_eq!(tokens.len(), batch, "one token per stream");
        assert_eq!(logits.len(), batch * cfg.vocab, "logits buffer size");
        ws.prepare(self, batch);
        for (row, &t) in ws.x.chunks_exact_mut(d).zip(tokens) {
            row.copy_from_slice(&self.tok_emb[t as usize * d..(t as usize + 1) * d]);
        }
        for (li, layer) in self.layers.iter().enumerate() {
            for (x, h) in ws.x.chunks_exact(d).zip(ws.h.chunks_exact_mut(d)) {
                rms_norm(x, &layer.attn_norm, cfg.norm_eps, h);
            }
            let qkv_dim = cfg.qkv_dim();
            quantize_rows(&ws.h, d, &mut ws.hq[..batch * d], &mut ws.hs);
            matmul(&ws.hq[..batch * d], &ws.hs, &layer.wqkv, &mut ws.qkv);
            for (b, stream) in streams.iter_mut().enumerate() {
                self.attend(
                    li,
                    stream,
                    &ws.qkv[b * qkv_dim..(b + 1) * qkv_dim],
                    &mut ws.attn[b * d..(b + 1) * d],
                    &mut ws.attn_scratch,
                );
            }
            add_projection(&ws.attn, &layer.wo, &mut ws.hq, &mut ws.hs, &mut ws.proj, &mut ws.x);

            for (x, h) in ws.x.chunks_exact(d).zip(ws.h.chunks_exact_mut(d)) {
                rms_norm(x, &layer.mlp_norm, cfg.norm_eps, h);
            }
            quantize_rows(&ws.h, d, &mut ws.hq[..batch * d], &mut ws.hs);
            matmul(&ws.hq[..batch * d], &ws.hs, &layer.w_in, &mut ws.mlp);
            for u in ws.mlp.iter_mut() {
                let r = if *u > 0.0 { *u } else { 0.0 };
                *u = r * r;
            }
            add_projection(&ws.mlp, &layer.w_out, &mut ws.hq, &mut ws.hs, &mut ws.proj, &mut ws.x);
        }
        for (x, h) in ws.x.chunks_exact(d).zip(ws.h.chunks_exact_mut(d)) {
            rms_norm(x, &self.final_norm, cfg.norm_eps, h);
        }
        quantize_rows(&ws.h, d, &mut ws.hq[..batch * d], &mut ws.hs);
        matmul(&ws.hq[..batch * d], &ws.hs, &self.lm_head, logits);
        for stream in streams.iter_mut() {
            stream.pos += 1;
        }
    }

    /// Stores this position's key/value in the cache and computes sliding-window attention.
    ///
    /// Keys are visited once for all query heads that share them (multi-query attention), so
    /// each cached key and value is loaded once per step rather than once per head.
    fn attend(
        &self,
        layer: usize,
        stream: &mut Stream,
        qkv: &[f32],
        out: &mut [f32],
        scratch: &mut AttnScratch,
    ) {
        let cfg = self.config();
        let (dh, n_kv, n_heads) = (cfg.d_head(), cfg.n_kv_heads, cfg.n_heads);
        let kv_dim = n_kv * dh;
        let cap = stream.capacity;
        let base = layer * cap;
        let newest = (stream.pos % cap as u64) as usize;
        let here = base + newest;
        let (q, rest) = qkv.split_at(cfg.d_model);
        let (k_new, v_new) = rest.split_at(kv_dim);
        for g in 0..n_kv {
            let dst = here * kv_dim + g * dh;
            stream.k_scale[here * n_kv + g] =
                quantize_row(&k_new[g * dh..(g + 1) * dh], &mut stream.k[dst..dst + dh]);
            stream.v_scale[here * n_kv + g] =
                quantize_row(&v_new[g * dh..(g + 1) * dh], &mut stream.v[dst..dst + dh]);
        }
        let n_keys = (stream.pos + 1).min(cfg.window as u64) as usize;
        // Cache slot of the key at each distance, newest first, without a division per key.
        scratch.slots.clear();
        scratch
            .slots
            .extend((0..n_keys).map(|d| base + if d <= newest { newest - d } else { cap + newest - d }));
        for head in 0..n_heads {
            let s = quantize_row(&q[head * dh..(head + 1) * dh], &mut scratch.q8[head * dh..(head + 1) * dh]);
            scratch.q_scale[head] = s * self.inv_sqrt_d_head;
        }
        let group = n_heads / n_kv;
        let scores = &mut scratch.scores[..n_heads * n_keys];
        for (dist, &at) in scratch.slots.iter().enumerate() {
            for g in 0..n_kv {
                let key = &stream.k[at * kv_dim + g * dh..at * kv_dim + (g + 1) * dh];
                let k_scale = stream.k_scale[at * n_kv + g];
                for head in g * group..(g + 1) * group {
                    let raw = dot_exact(&scratch.q8[head * dh..(head + 1) * dh], key);
                    scores[head * n_keys + dist] =
                        raw * scratch.q_scale[head] * k_scale - self.slopes[head] * dist as f32;
                }
            }
        }
        for head in 0..n_heads {
            let row = &mut scores[head * n_keys..(head + 1) * n_keys];
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
            for s in row.iter_mut() {
                *s = exp_nonpos(*s - max);
            }
            scratch.inv_sum[head] = 1.0 / sum_f32(row);
        }
        out.fill(0.0);
        for (dist, &at) in scratch.slots.iter().enumerate() {
            for g in 0..n_kv {
                let value = &stream.v[at * kv_dim + g * dh..at * kv_dim + (g + 1) * dh];
                let v_scale = stream.v_scale[at * n_kv + g];
                for head in g * group..(g + 1) * group {
                    let w = scores[head * n_keys + dist] * v_scale;
                    for (a, &v) in out[head * dh..(head + 1) * dh].iter_mut().zip(value) {
                        *a += w * v;
                    }
                }
            }
        }
        for head in 0..n_heads {
            let inv = scratch.inv_sum[head];
            for a in out[head * dh..(head + 1) * dh].iter_mut() {
                *a *= inv;
            }
        }
    }
}
