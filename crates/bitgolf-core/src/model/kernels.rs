//! Batched kernels. Each output element is computed by one fixed scalar recipe, so results do
//! not depend on batch size, row order or thread count: that is what lets the encoder and the
//! decoder batch blocks differently and still agree bit for bit.
//!
//! Quantised values are kept as small integers stored in `f32` so that dot products run on
//! wide float SIMD while remaining exact (see [`crate::math::dot_exact`]).

use super::QMatrix;
use crate::math::{dot_exact, round_ties_even, rsqrt, sum_squares};

/// Symmetric per-row int8 quantisation, identical to the training fake-quant:
/// `scale = max|x| / 127`, `q = round_ties_even(x / scale)`. Writes the integers (as `f32`)
/// to `out` and returns the scale.
#[inline]
pub fn quantize_row(x: &[f32], out: &mut [f32]) -> f32 {
    let amax = x.iter().fold(0.0f32, |m, &v| {
        let a = v.abs();
        if a > m { a } else { m }
    });
    let scale = amax / 127.0;
    if scale > 0.0 {
        for (q, &v) in out.iter_mut().zip(x) {
            let r = round_ties_even(v / scale);
            *q = r.clamp(-127.0, 127.0);
        }
    } else {
        out.fill(0.0);
    }
    scale
}

/// `out = x * rsqrt(mean(x^2) + eps) * gain`.
#[inline]
pub fn rms_norm(x: &[f32], gain: &[f32], eps: f32, out: &mut [f32]) {
    let inv = rsqrt(sum_squares(x) / x.len() as f32 + eps);
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(gain) {
        *o = v * inv * g;
    }
}

/// Batched quantised matrix product: `out[b, n] = (x[b] . w[n]) * x_scale[b] * w.scale[n]`.
///
/// `x` holds `batch` rows of `w.cols` quantised values. Rows of `w` are the outer loop so
/// each weight row is loaded once and reused across the whole batch while it sits in L1.
pub fn matmul(x: &[f32], x_scales: &[f32], w: &QMatrix, out: &mut [f32]) {
    let batch = x_scales.len();
    let (k, n_out) = (w.cols, w.rows);
    debug_assert_eq!(x.len(), batch * k);
    debug_assert_eq!(out.len(), batch * n_out);
    for (n, (w_row, &w_scale)) in w.data.chunks_exact(k).zip(&w.scales).enumerate() {
        for b in 0..batch {
            let acc = dot_exact(&x[b * k..(b + 1) * k], w_row);
            out[b * n_out + n] = acc * x_scales[b] * w_scale;
        }
    }
}
