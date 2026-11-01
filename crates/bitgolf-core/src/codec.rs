//! Model-driven block coding: compress/decompress whole containers, single blocks, block
//! prefixes, and per-byte cost analysis.
//!
//! Blocks are independent. A group of blocks is coded in lockstep (one model step advances
//! every block in the group by one byte), and groups run in parallel when the `parallel`
//! feature is on. Because every model output is a pure function of its own block's bytes,
//! grouping and thread count never change the encoded bits.

use sha2::{Digest, Sha256};

use crate::container::{self, BlockCodec, Container, EncodedBlock, Header};
use crate::error::{Error, Result};
use crate::freq::{FreqTable, PROB_BITS};
use crate::model::{Model, Stream, Workspace};
use crate::rangecoder::{RangeDecoder, RangeEncoder};

/// Default block size: large enough that cold starts cost little, small enough for random
/// access and parallelism.
pub const DEFAULT_BLOCK_SIZE: usize = 64 << 10;
/// Most blocks coded in lockstep by one thread.
const MAX_GROUP: usize = 16;
/// Model coding of a block is abandoned once it averages more than this many bits per byte
/// (checked every [`ABORT_CHECK`] bytes): such data is binary and deflate will do better.
const ABORT_BITS_PER_BYTE: usize = 7;
const ABORT_CHECK: usize = 4096;

/// Options for [`compress`].
#[derive(Clone, Debug)]
pub struct CompressOptions {
    /// Raw bytes per block.
    pub block_size: usize,
    /// Context fed to the model before every block.
    pub preamble: Vec<u8>,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self { block_size: DEFAULT_BLOCK_SIZE, preamble: Vec::new() }
    }
}

/// Model state after the preamble: cloned to start every block.
#[derive(Clone)]
pub struct Primer {
    stream: Stream,
    table: FreqTable,
}

impl Primer {
    /// Runs the model over `preamble`. An empty preamble starts from a uniform first byte.
    pub fn new(model: &Model, preamble: &[u8]) -> Self {
        let mut stream = Stream::new(model);
        let mut table = FreqTable::uniform();
        let mut ws = Workspace::new();
        let mut logits = [0.0f32; 256];
        for &b in preamble {
            model.step(&mut stream, b, &mut ws, &mut logits);
            table = FreqTable::from_logits(&logits);
        }
        Self { stream, table }
    }
}

fn map_groups<T: Sync, R: Send>(items: &[T], f: impl Fn(&[T]) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let threads = rayon::current_num_threads().max(1);
        let group = items.len().div_ceil(threads).clamp(1, MAX_GROUP);
        items.par_chunks(group).map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items.chunks(MAX_GROUP).map(f).collect()
    }
}

/// Advances the streams flagged in `active` by their tokens and refreshes their tables.
fn step_active(
    model: &Model,
    streams: &mut [Stream],
    tables: &mut [FreqTable],
    active: &[bool],
    tokens: &[u8],
    ws: &mut Workspace,
    logits: &mut Vec<f32>,
) {
    let mut refs: Vec<&mut Stream> =
        streams.iter_mut().zip(active).filter(|(_, a)| **a).map(|(s, _)| s).collect();
    if refs.is_empty() {
        return;
    }
    logits.resize(refs.len() * 256, 0.0);
    model.step_batch(&mut refs, tokens, ws, logits);
    let targets = tables.iter_mut().zip(active).filter(|(_, a)| **a).map(|(t, _)| t);
    for (table, row) in targets.zip(logits.chunks_exact(256)) {
        *table = FreqTable::from_logits(row);
    }
}

/// Range-codes each block; `None` where coding was abandoned as hopeless.
fn encode_group(model: &Model, primer: &Primer, blocks: &[&[u8]]) -> Vec<Option<Vec<u8>>> {
    let n = blocks.len();
    let mut streams = vec![primer.stream.clone(); n];
    let mut tables = vec![primer.table.clone(); n];
    let mut encoders: Vec<Option<RangeEncoder>> = (0..n).map(|_| Some(RangeEncoder::new())).collect();
    let (mut ws, mut logits) = (Workspace::new(), Vec::new());
    let max_len = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let mut active = vec![false; n];
    let mut tokens = Vec::with_capacity(n);
    for t in 0..max_len {
        tokens.clear();
        for i in 0..n {
            active[i] = false;
            let Some(enc) = encoders[i].as_mut() else { continue };
            let Some(&byte) = blocks[i].get(t) else { continue };
            enc.encode(tables[i].cum(byte), tables[i].freq(byte), PROB_BITS);
            let done = t + 1;
            if done % ABORT_CHECK == 0 && enc.len() * 8 > done * ABORT_BITS_PER_BYTE {
                encoders[i] = None;
                continue;
            }
            if done < blocks[i].len() {
                active[i] = true;
                tokens.push(byte);
            }
        }
        step_active(model, &mut streams, &mut tables, &active, &tokens, &mut ws, &mut logits);
    }
    encoders.into_iter().map(|e| e.map(RangeEncoder::finish)).collect()
}

/// Decodes the first `want[i]` bytes of each model-coded payload.
fn decode_group(model: &Model, primer: &Primer, jobs: &[(&[u8], usize)]) -> Result<Vec<Vec<u8>>> {
    let n = jobs.len();
    let mut streams = vec![primer.stream.clone(); n];
    let mut tables = vec![primer.table.clone(); n];
    let mut decoders: Vec<RangeDecoder<'_>> = jobs.iter().map(|(p, _)| RangeDecoder::new(p)).collect();
    let mut outputs: Vec<Vec<u8>> = jobs.iter().map(|(_, w)| Vec::with_capacity(*w)).collect();
    let (mut ws, mut logits) = (Workspace::new(), Vec::new());
    let max_len = jobs.iter().map(|(_, w)| *w).max().unwrap_or(0);
    let mut active = vec![false; n];
    let mut tokens = Vec::with_capacity(n);
    for t in 0..max_len {
        tokens.clear();
        for i in 0..n {
            active[i] = false;
            if t >= jobs[i].1 {
                continue;
            }
            let table = &tables[i];
            let byte = table.symbol_for(decoders[i].target(PROB_BITS));
            decoders[i].consume(table.cum(byte), table.freq(byte))?;
            outputs[i].push(byte);
            if t + 1 < jobs[i].1 {
                active[i] = true;
                tokens.push(byte);
            }
        }
        step_active(model, &mut streams, &mut tables, &active, &tokens, &mut ws, &mut logits);
    }
    if decoders.iter().any(RangeDecoder::overran) {
        return Err(Error::Corrupt("model payload ended early".into()));
    }
    Ok(outputs)
}

fn deflate(data: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec(data, 9)
}

/// Compresses `data` into a `.bg` container.
pub fn compress(model: &Model, data: &[u8], opts: &CompressOptions) -> Result<Vec<u8>> {
    let blocks = encode_blocks(model, data, opts)?;
    let header = Header {
        model_id: model.id(),
        window: model.config().window as u32,
        block_size: opts.block_size as u32,
        preamble: opts.preamble.clone(),
    };
    Ok(container::write(&header, &blocks))
}

/// Splits `data` into blocks and codes each with the cheapest of model, deflate and stored.
pub fn encode_blocks(model: &Model, data: &[u8], opts: &CompressOptions) -> Result<Vec<EncodedBlock>> {
    if opts.block_size == 0 || opts.block_size > container::MAX_BLOCK {
        return Err(Error::InvalidArgument(format!(
            "block size must be between 1 and {} bytes",
            container::MAX_BLOCK
        )));
    }
    if opts.preamble.len() > container::MAX_PREAMBLE {
        return Err(Error::InvalidArgument("preamble is larger than 64 KiB".into()));
    }
    let primer = Primer::new(model, &opts.preamble);
    let chunks: Vec<&[u8]> = data.chunks(opts.block_size).collect();
    let coded: Vec<Option<Vec<u8>>> =
        map_groups(&chunks, |group| encode_group(model, &primer, group)).into_iter().flatten().collect();
    Ok(chunks.iter().zip(coded).map(|(raw, model_payload)| choose_codec(raw, model_payload)).collect())
}

/// Range-codes one block under the model, or `None` if it was abandoned as incompressible.
pub fn model_encode(model: &Model, primer: &Primer, raw: &[u8]) -> Option<Vec<u8>> {
    encode_group(model, primer, &[raw]).pop().flatten()
}

/// Codes one block after `primer` with the cheapest codec.
pub fn encode_single(model: &Model, primer: &Primer, raw: &[u8]) -> EncodedBlock {
    choose_codec(raw, model_encode(model, primer, raw))
}

/// Picks the smallest encoding of a block.
pub fn choose_codec(raw: &[u8], model_payload: Option<Vec<u8>>) -> EncodedBlock {
    let deflated = deflate(raw);
    let (codec, payload) = match model_payload {
        Some(m) if m.len() <= deflated.len() && m.len() <= raw.len() => (BlockCodec::Model, m),
        _ if deflated.len() < raw.len() => (BlockCodec::Deflate, deflated),
        _ => (BlockCodec::Stored, raw.to_vec()),
    };
    EncodedBlock { codec, raw_len: raw.len() as u32, raw_crc: container::crc32c(raw), payload }
}

fn check_model(model: &Model, c: &Container<'_>) -> Result<()> {
    c.require_model(&model.id())?;
    if c.header().window as usize != model.config().window {
        return Err(Error::Corrupt("container window does not match the model".into()));
    }
    Ok(())
}

fn decode_simple(codec: BlockCodec, payload: &[u8], raw_len: usize) -> Result<Vec<u8>> {
    match codec {
        BlockCodec::Stored => Ok(payload.to_vec()),
        BlockCodec::Deflate => miniz_oxide::inflate::decompress_to_vec_with_limit(payload, raw_len)
            .map_err(|e| Error::Corrupt(format!("deflate payload: {e:?}"))),
        BlockCodec::Model => unreachable!("model blocks are decoded in groups"),
    }
}

/// Decodes the blocks listed in `indices` (full length, checksums verified).
pub fn decode_blocks(model: &Model, c: &Container<'_>, indices: &[usize]) -> Result<Vec<Vec<u8>>> {
    check_model(model, c)?;
    let primer = Primer::new(model, &c.header().preamble);
    let mut payloads = Vec::with_capacity(indices.len());
    for &i in indices {
        payloads.push((i, c.payload(i)?));
    }
    let model_jobs: Vec<(usize, &[u8], usize)> = payloads
        .iter()
        .filter(|(i, _)| c.blocks()[*i].codec == BlockCodec::Model)
        .map(|&(i, p)| (i, p, c.blocks()[i].raw_len as usize))
        .collect();
    let decoded: Vec<Result<Vec<Vec<u8>>>> = map_groups(&model_jobs, |group| {
        let jobs: Vec<(&[u8], usize)> = group.iter().map(|&(_, p, n)| (p, n)).collect();
        decode_group(model, &primer, &jobs)
    });
    let mut model_out = Vec::with_capacity(model_jobs.len());
    for group in decoded {
        model_out.extend(group?);
    }
    let mut model_out = model_out.into_iter();
    let mut result = Vec::with_capacity(indices.len());
    for (i, payload) in payloads {
        let e = c.blocks()[i];
        let raw = match e.codec {
            BlockCodec::Model => model_out.next().expect("one output per model job"),
            codec => decode_simple(codec, payload, e.raw_len as usize)?,
        };
        if raw.len() != e.raw_len as usize || container::crc32c(&raw) != e.raw_crc {
            return Err(Error::Corrupt(format!(
                "block {i}: decoded data fails its checksum (wrong model or engine mismatch)"
            )));
        }
        result.push(raw);
    }
    Ok(result)
}

/// Decompresses a whole container.
pub fn decompress(model: &Model, bytes: &[u8]) -> Result<Vec<u8>> {
    let c = Container::parse(bytes)?;
    let all: Vec<usize> = (0..c.blocks().len()).collect();
    let blocks = decode_blocks(model, &c, &all)?;
    let mut out = Vec::with_capacity(c.total_len() as usize);
    for b in blocks {
        out.extend_from_slice(&b);
    }
    Ok(out)
}

/// Decodes only the first `len` bytes of one block (no whole-block checksum: callers verify
/// the bytes they asked for with their own checksum).
pub fn decode_prefix(
    model: &Model,
    primer: &Primer,
    codec: BlockCodec,
    payload: &[u8],
    raw_len: usize,
    len: usize,
) -> Result<Vec<u8>> {
    let len = len.min(raw_len);
    match codec {
        BlockCodec::Model => {
            let mut out = decode_group(model, primer, &[(payload, len)])?;
            Ok(out.pop().expect("one job"))
        }
        codec => {
            let mut raw = decode_simple(codec, payload, raw_len)?;
            raw.truncate(len);
            Ok(raw)
        }
    }
}

/// Bits the coder spends on each byte of `data` when coded as one block after `preamble`.
pub fn byte_costs(model: &Model, data: &[u8], preamble: &[u8]) -> Vec<f32> {
    let primer = Primer::new(model, preamble);
    let mut stream = primer.stream;
    let mut table = primer.table;
    let mut ws = Workspace::new();
    let mut logits = [0.0f32; 256];
    let mut costs = Vec::with_capacity(data.len());
    for (i, &b) in data.iter().enumerate() {
        costs.push(table.cost_bits(b) as f32);
        if i + 1 < data.len() {
            model.step(&mut stream, b, &mut ws, &mut logits);
            table = FreqTable::from_logits(&logits);
        }
    }
    costs
}

/// sha256 over the bit patterns of every logit produced while reading `data`. Equal digests
/// on two platforms mean bit-identical inference.
pub fn logits_digest(model: &Model, data: &[u8]) -> [u8; 32] {
    let mut stream = Stream::new(model);
    let mut ws = Workspace::new();
    let mut logits = [0.0f32; 256];
    let mut hasher = Sha256::new();
    for &b in data {
        model.step(&mut stream, b, &mut ws, &mut logits);
        for l in logits {
            hasher.update(l.to_bits().to_le_bytes());
        }
    }
    hasher.finalize().into()
}
