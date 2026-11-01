//! Deterministic scalar math.
//!
//! Every function here uses only IEEE-754 `+ - * /`, comparisons and bit casts, which are
//! correctly rounded and therefore bit-identical on every conforming target (x86-64 SSE2/AVX2,
//! wasm32 with or without SIMD, aarch64). Rust never contracts `a * b + c` into a fused
//! multiply-add, and nothing here calls into libm, whose `exp`/`log` differ between platforms.

/// `1.5 * 2^23`: adding and subtracting it rounds an `f32` to the nearest integer with ties to
/// even (the IEEE default rounding), valid for `|x| < 2^22`.
const ROUND_MAGIC: f32 = 12_582_912.0;

/// Rounds to the nearest integer, ties to even, matching `torch.round`. Valid for `|x| < 2^22`.
#[inline(always)]
pub fn round_ties_even(x: f32) -> f32 {
    (x + ROUND_MAGIC) - ROUND_MAGIC
}

const LOG2_E: f32 = std::f32::consts::LOG2_E;
// ln(2) split so that `n * LN2_HI` is exact for |n| < 2^9 (Cody-Waite range reduction).
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;

/// `e^x` for `x <= 0`, accurate to about 2 ulp; inputs below -87 flush to `e^-87`.
///
/// Used for softmax, where arguments are `logit - max <= 0`.
#[inline(always)]
pub fn exp_nonpos(x: f32) -> f32 {
    let x = if x < -87.0 { -87.0 } else { x };
    let shifted = x * LOG2_E + ROUND_MAGIC;
    let n = shifted - ROUND_MAGIC;
    let exponent = (shifted.to_bits() as i32).wrapping_sub(0x4B40_0000);
    let f = (x - n * LN2_HI) - n * LN2_LO;
    // Degree-6 Taylor polynomial of e^f on |f| <= ln(2)/2, in Horner form.
    let p = 1.0
        + f * (1.0 + f * (0.5 + f * (1.0 / 6.0 + f * (1.0 / 24.0 + f * (1.0 / 120.0 + f * (1.0 / 720.0))))));
    p * f32::from_bits(((exponent + 127) as u32) << 23)
}

/// `1 / sqrt(x)` for `x > 0` via a bit-level initial guess and three Newton steps.
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    let mut y = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(x.to_bits() >> 1));
    let half = 0.5 * x;
    y = y * (1.5 - half * y * y);
    y = y * (1.5 - half * y * y);
    y * (1.5 - half * y * y)
}

/// Base-2 logarithm of a positive finite `f64`, accurate to ~1e-12. Used for reporting.
pub fn log2(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let mut exponent = ((bits >> 52) & 0x7FF) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        exponent += 1;
    }
    // ln(m) = 2 atanh(s) with s = (m - 1) / (m + 1), |s| <= 0.172.
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    let mut k = 1.0;
    while k < 30.0 {
        sum += term / k;
        term *= s2;
        k += 2.0;
    }
    exponent as f64 + 2.0 * sum * std::f64::consts::LOG2_E
}

/// Sum of `a[i] * b[i]` in a fixed order: eight interleaved lanes, then the tail.
///
/// Eight independent partial sums let the compiler emit SIMD code without reassociating
/// floating-point additions, so the result is identical with and without vectorisation.
/// (The tail is a separate scalar: indexing the lane array dynamically would force it out of
/// registers.)
#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut lanes = [0.0f32; 8];
    let full = a.len() / 8 * 8;
    for (ca, cb) in a[..full].chunks_exact(8).zip(b[..full].chunks_exact(8)) {
        for l in 0..8 {
            lanes[l] += ca[l] * cb[l];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in a[full..].iter().zip(&b[full..]) {
        tail += x * y;
    }
    reduce_lanes(lanes) + tail
}

/// Sum of squares with the same fixed order as [`dot_f32`].
#[inline(always)]
pub fn sum_squares(a: &[f32]) -> f32 {
    dot_f32(a, a)
}

/// Sum in a fixed order: eight interleaved lanes, then the tail.
#[inline(always)]
pub fn sum_f32(a: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let full = a.len() / 8 * 8;
    for c in a[..full].chunks_exact(8) {
        for l in 0..8 {
            lanes[l] += c[l];
        }
    }
    let mut tail = 0.0f32;
    for x in &a[full..] {
        tail += x;
    }
    reduce_lanes(lanes) + tail
}

#[inline(always)]
fn reduce_lanes(l: [f32; 8]) -> f32 {
    ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]))
}

/// Largest vector length for which [`dot_exact`] is exact: `127 * 127 * 1040 < 2^24`.
pub const EXACT_DOT_MAX_LEN: usize = 1040;

/// Dot product of two vectors of small integers stored as `f32` (int8 values, |v| <= 127).
///
/// Every product and partial sum is an integer below `2^24`, so it is represented exactly in
/// `f32` and the result does not depend on summation order at all. That makes it both
/// deterministic and fast: four independent 8-lane accumulators give the compiler wide SIMD
/// with enough parallel dependency chains to hide add latency.
#[inline(always)]
pub fn dot_exact(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= EXACT_DOT_MAX_LEN);
    let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
    let full = a.len() / 32 * 32;
    for (ca, cb) in a[..full].chunks_exact(32).zip(b[..full].chunks_exact(32)) {
        for l in 0..8 {
            s0[l] += ca[l] * cb[l];
            s1[l] += ca[l + 8] * cb[l + 8];
            s2[l] += ca[l + 16] * cb[l + 16];
            s3[l] += ca[l + 24] * cb[l + 24];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in a[full..].iter().zip(&b[full..]) {
        tail += x * y;
    }
    for l in 0..8 {
        s0[l] += (s1[l] + s2[l]) + s3[l];
    }
    reduce_lanes(s0) + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_matches_ties_to_even() {
        for (x, want) in [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (-0.5, -0.0), (-1.5, -2.0), (126.6, 127.0)] {
            assert_eq!(round_ties_even(x), want, "round({x})");
        }
    }

    #[test]
    fn exp_is_accurate() {
        let mut x = -87.0f32;
        while x <= 0.0 {
            let got = exp_nonpos(x) as f64;
            let want = (x as f64).exp();
            assert!(((got - want) / want).abs() < 4e-7, "exp({x}) = {got}, want {want}");
            x += 0.0137;
        }
        assert_eq!(exp_nonpos(0.0), 1.0);
    }

    #[test]
    fn rsqrt_is_accurate() {
        for &x in &[1e-6f32, 0.01, 0.5, 1.0, 2.0, 3.7, 1e3, 1e6] {
            let want = 1.0 / (x as f64).sqrt();
            let got = rsqrt(x) as f64;
            assert!(((got - want) / want).abs() < 1e-6, "rsqrt({x}) = {got}, want {want}");
        }
    }

    #[test]
    fn log2_is_accurate() {
        for &x in &[1e-9, 0.3, 1.0, 1.5, 2.0, 1000.0, 1048576.0, 3.0e9] {
            assert!((log2(x) - x.log2()).abs() < 1e-10, "log2({x})");
        }
    }

    #[test]
    fn dot_products_match_reference() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.25 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.125).collect();
        let reference: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        assert!((dot_f32(&a, &b) as f64 - reference).abs() < 1e-3);
        let ai: Vec<i32> = (0..1040).map(|i| i % 255 - 127).collect();
        let bi: Vec<i32> = (0..1040).map(|i| if i % 3 == 0 { 127 } else { -127 }).collect();
        let want: i32 = ai.iter().zip(&bi).map(|(x, y)| x * y).sum();
        let af: Vec<f32> = ai.iter().map(|&v| v as f32).collect();
        let bf: Vec<f32> = bi.iter().map(|&v| v as f32).collect();
        assert_eq!(dot_exact(&af, &bf), want as f32);
        let worst = vec![127.0f32; EXACT_DOT_MAX_LEN];
        assert_eq!(dot_exact(&worst, &worst), (127 * 127 * EXACT_DOT_MAX_LEN) as f32);
    }
}
