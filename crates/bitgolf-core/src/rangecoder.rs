//! A 64-bit range coder with byte-wise renormalisation and carry propagation.
//!
//! The range is kept in `[2^56, 2^64)`, so with frequency totals of `2^20` the integer
//! division `range / total` loses less than `2^-36` of the interval per symbol: coding
//! overhead is dominated by the model, not the coder. Carries ripple into bytes already
//! written, which is simple because blocks are coded into an in-memory buffer.

use crate::error::{Error, Result};

const TOP: u64 = 1 << 56;

/// Encodes symbols given their cumulative frequency, frequency and total (a power of two).
pub struct RangeEncoder {
    low: u64,
    range: u64,
    out: Vec<u8>,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    /// A fresh encoder covering the full interval.
    pub fn new() -> Self {
        Self { low: 0, range: u64::MAX, out: Vec::new() }
    }

    /// Narrows the interval to `[cum, cum + freq)` out of `2^total_bits`.
    #[inline]
    pub fn encode(&mut self, cum: u32, freq: u32, total_bits: u32) {
        debug_assert!(freq > 0 && u64::from(cum) + u64::from(freq) <= 1 << total_bits);
        let r = self.range >> total_bits;
        let (low, carry) = self.low.overflowing_add(r * u64::from(cum));
        if carry {
            self.propagate_carry();
        }
        self.low = low;
        self.range = r * u64::from(freq);
        while self.range < TOP {
            self.out.push((self.low >> 56) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    fn propagate_carry(&mut self) {
        for byte in self.out.iter_mut().rev() {
            if *byte == 0xFF {
                *byte = 0;
            } else {
                *byte += 1;
                return;
            }
        }
        unreachable!("range coder carry past the start of the stream");
    }

    /// Bytes written so far (excluding the final flush).
    pub fn len(&self) -> usize {
        self.out.len()
    }

    /// True when nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    /// Flushes the shortest byte string that identifies the final interval.
    pub fn finish(mut self) -> Vec<u8> {
        let low = u128::from(self.low);
        let high = low + u128::from(self.range);
        for n in 1..=8u32 {
            let unit = 1u128 << (64 - 8 * n);
            let value = low.div_ceil(unit) * unit;
            if value < high {
                if value >> 64 != 0 {
                    self.propagate_carry();
                }
                let value = value as u64;
                for i in 0..n {
                    self.out.push((value >> (56 - 8 * i)) as u8);
                }
                return self.out;
            }
        }
        unreachable!("an 8-byte flush always identifies the interval");
    }
}

/// Decodes a stream produced by [`RangeEncoder`]. Reads past the end as zero bytes.
pub struct RangeDecoder<'a> {
    code: u64,
    range: u64,
    r: u64,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    /// Starts decoding `input`.
    pub fn new(input: &'a [u8]) -> Self {
        let mut d = Self { code: 0, range: u64::MAX, r: 0, input, pos: 0 };
        for _ in 0..8 {
            d.code = (d.code << 8) | u64::from(d.next_byte());
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Returns the cumulative-frequency target of the next symbol, in `[0, 2^total_bits)`.
    #[inline]
    pub fn target(&mut self, total_bits: u32) -> u32 {
        self.r = self.range >> total_bits;
        let t = self.code / self.r;
        t.min((1u64 << total_bits) - 1) as u32
    }

    /// Consumes the symbol `[cum, cum + freq)` found via [`RangeDecoder::target`].
    #[inline]
    pub fn consume(&mut self, cum: u32, freq: u32) -> Result<()> {
        self.code -= self.r * u64::from(cum);
        self.range = self.r * u64::from(freq);
        if self.code >= self.range {
            return Err(Error::Corrupt("range coder stream is inconsistent".into()));
        }
        while self.range < TOP {
            self.code = (self.code << 8) | u64::from(self.next_byte());
            self.range <<= 8;
        }
        Ok(())
    }

    /// True if the decoder read beyond the input plus the encoder's implicit zero padding.
    pub fn overran(&self) -> bool {
        self.pos > self.input.len() + 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freq::{FreqTable, PROB_BITS};
    use proptest::prelude::*;

    fn table(seed: &[f32]) -> FreqTable {
        let logits: Vec<f32> = (0..256).map(|i| seed[i % seed.len()] * (i % 7) as f32).collect();
        FreqTable::from_logits(&logits)
    }

    fn round_trip(symbols: &[u8], tables: &[FreqTable]) {
        let mut enc = RangeEncoder::new();
        for (i, &s) in symbols.iter().enumerate() {
            let t = &tables[i % tables.len()];
            enc.encode(t.cum(s), t.freq(s), PROB_BITS);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &want) in symbols.iter().enumerate() {
            let t = &tables[i % tables.len()];
            let s = t.symbol_for(dec.target(PROB_BITS));
            assert_eq!(s, want, "symbol {i}");
            dec.consume(t.cum(s), t.freq(s)).unwrap();
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn round_trips_arbitrary_tables(
            symbols in proptest::collection::vec(any::<u8>(), 0..2000),
            seeds in proptest::collection::vec(proptest::collection::vec(-20f32..20.0, 1..8), 1..6),
        ) {
            let tables: Vec<FreqTable> = seeds.iter().map(|s| table(s)).collect();
            round_trip(&symbols, &tables);
        }

        #[test]
        fn round_trips_extreme_skew(symbols in proptest::collection::vec(any::<u8>(), 0..3000)) {
            // Every symbol coded under a table that makes it as unlikely as possible, and under
            // one that makes it near certain: exercises carries and long renormalisations.
            let mut tables = Vec::new();
            for hot in [0u8, 255, 97] {
                let mut logits = [-80.0f32; 256];
                logits[hot as usize] = 80.0;
                tables.push(FreqTable::from_logits(&logits));
            }
            round_trip(&symbols, &tables);
        }
    }

    #[test]
    fn empty_stream_is_tiny() {
        assert!(RangeEncoder::new().finish().len() <= 1);
    }

    #[test]
    fn near_certain_symbols_cost_almost_nothing() {
        let mut logits = [-40.0f32; 256];
        logits[b'a' as usize] = 40.0;
        let t = FreqTable::from_logits(&logits);
        let mut enc = RangeEncoder::new();
        for _ in 0..100_000 {
            enc.encode(t.cum(b'a'), t.freq(b'a'), PROB_BITS);
        }
        assert!(enc.finish().len() < 64);
    }
}
