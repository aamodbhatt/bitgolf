//! Deterministic mapping from model logits to integer symbol frequencies.
//!
//! The encoder and decoder must build the exact same table from the exact same logits, so
//! this uses only correctly rounded IEEE operations ([`crate::math::exp_nonpos`], `f64`
//! multiply and truncation) in a fixed order. Every symbol gets a frequency of at least one,
//! so any byte can be coded even when the model is certain it will not appear.

use crate::math::{exp_nonpos, log2};

/// Frequencies sum to `2^PROB_BITS`.
pub const PROB_BITS: u32 = 20;
/// Sum of all frequencies.
pub const TOTAL: u32 = 1 << PROB_BITS;

/// Cumulative frequency table over the 256 byte values.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FreqTable {
    cum: [u32; 257],
}

impl FreqTable {
    /// Equal frequencies, used for the first byte of an unprimed block.
    pub fn uniform() -> Self {
        let mut cum = [0u32; 257];
        for (i, c) in cum.iter_mut().enumerate() {
            *c = i as u32 * (TOTAL / 256);
        }
        Self { cum }
    }

    /// Quantises `softmax(logits)` to integer frequencies summing to [`TOTAL`].
    ///
    /// Non-finite logits are treated as impossible (frequency 1). Rounding leftovers go to
    /// the most likely symbol (lowest index on ties).
    pub fn from_logits(logits: &[f32]) -> Self {
        assert_eq!(logits.len(), 256, "byte-level logits");
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| if v.is_finite() && v > m { v } else { m });
        let mut weights = [0.0f32; 256];
        let mut sum = 0.0f64;
        for (w, &l) in weights.iter_mut().zip(logits) {
            *w = if l.is_finite() { exp_nonpos(l - max) } else { 0.0 };
            sum += f64::from(*w);
        }
        let mut freq = [1u32; 256];
        if sum > 0.0 {
            let scale = f64::from(TOTAL - 256) / sum;
            for (f, &w) in freq.iter_mut().zip(&weights) {
                *f += (f64::from(w) * scale) as u32;
            }
        }
        let assigned: u32 = freq.iter().sum();
        let best = (0..256).fold(0, |b, i| if freq[i] > freq[b] { i } else { b });
        freq[best] += TOTAL - assigned;
        let mut cum = [0u32; 257];
        for i in 0..256 {
            cum[i + 1] = cum[i] + freq[i];
        }
        Self { cum }
    }

    /// Cumulative frequency of all symbols below `symbol`.
    #[inline]
    pub fn cum(&self, symbol: u8) -> u32 {
        self.cum[symbol as usize]
    }

    /// Frequency of `symbol` (at least 1).
    #[inline]
    pub fn freq(&self, symbol: u8) -> u32 {
        self.cum[symbol as usize + 1] - self.cum[symbol as usize]
    }

    /// The symbol whose cumulative range contains `target < TOTAL`.
    #[inline]
    pub fn symbol_for(&self, target: u32) -> u8 {
        (self.cum.partition_point(|&c| c <= target) - 1) as u8
    }

    /// Ideal code length of `symbol` under this table, in bits.
    pub fn cost_bits(&self, symbol: u8) -> f64 {
        f64::from(PROB_BITS) - log2(f64::from(self.freq(symbol)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn tables_are_complete(logits in proptest::collection::vec(-1e4f32..1e4, 256)) {
            let t = FreqTable::from_logits(&logits);
            prop_assert_eq!(t.cum[256], TOTAL);
            for s in 0..=255u8 {
                prop_assert!(t.freq(s) >= 1);
                prop_assert_eq!(t.symbol_for(t.cum(s)), s);
                prop_assert_eq!(t.symbol_for(t.cum(s) + t.freq(s) - 1), s);
            }
        }
    }

    #[test]
    fn non_finite_logits_are_handled() {
        let mut logits = [0.0f32; 256];
        logits[3] = f32::NAN;
        logits[4] = f32::INFINITY;
        logits[5] = f32::NEG_INFINITY;
        let t = FreqTable::from_logits(&logits);
        assert_eq!(t.cum[256], TOTAL);
        assert_eq!(t.freq(3), 1);
        let all_nan = FreqTable::from_logits(&[f32::NAN; 256]);
        assert_eq!(all_nan.cum[256], TOTAL);
    }

    #[test]
    fn confident_prediction_is_cheap() {
        let mut logits = [-30.0f32; 256];
        logits[b'e' as usize] = 30.0;
        let t = FreqTable::from_logits(&logits);
        assert!(t.cost_bits(b'e') < 0.001);
        assert!((t.cost_bits(b'x') - 20.0).abs() < 1e-9);
    }
}
