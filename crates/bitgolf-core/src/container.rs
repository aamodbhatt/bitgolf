//! The `.bg` container: a header, independently decodable blocks, a block index and a trailer.
//!
//! ```text
//! header   magic "BITGOLF\0" | version u16 | flags u16 | model sha256 [32] | window u32
//!          | block_size u32 | preamble_len u32 | preamble | header crc32c u32
//! blocks   payload 0 | payload 1 | ...
//! index    count u32 | count x (codec u8, raw_len u32, offset u64, len u32,
//!          raw_crc32c u32, payload_crc32c u32) | index crc32c u32
//! trailer  index_offset u64 | total_raw_len u64 | magic "BGINDEX\0"
//! ```
//!
//! All integers are little-endian. The index sits at the end so blocks can be streamed out as
//! they finish; the fixed-size trailer locates it. Every structure carries a CRC32C so damage
//! is reported precisely instead of producing wrong output.

use crate::error::{Error, Result, hex};

const MAGIC: &[u8; 8] = b"BITGOLF\0";
const TRAILER_MAGIC: &[u8; 8] = b"BGINDEX\0";
const VERSION: u16 = 1;
const ENTRY_SIZE: usize = 25;
const TRAILER_SIZE: usize = 24;
/// Largest block the format accepts (raw bytes).
pub const MAX_BLOCK: usize = 64 << 20;
/// Largest priming preamble the format accepts.
pub const MAX_PREAMBLE: usize = 64 << 10;

const CASTAGNOLI: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_ISCSI);

/// CRC32C (Castagnoli) of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    CASTAGNOLI.checksum(data)
}

/// How a block's payload is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockCodec {
    /// Range-coded under the model's predictions.
    Model,
    /// Raw deflate, used when the model does worse (e.g. binary data).
    Deflate,
    /// Stored verbatim, used when nothing compresses.
    Stored,
}

impl BlockCodec {
    fn tag(self) -> u8 {
        match self {
            Self::Model => 0,
            Self::Deflate => 1,
            Self::Stored => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::Model),
            1 => Ok(Self::Deflate),
            2 => Ok(Self::Stored),
            t => Err(Error::Corrupt(format!("unknown block codec {t}"))),
        }
    }

    /// Short lowercase name for display.
    pub fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Deflate => "deflate",
            Self::Stored => "stored",
        }
    }
}

/// Container-wide settings needed to decode any block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    /// sha256 of the `.bgm` model file used for coding.
    pub model_id: [u8; 32],
    /// Attention window of the model (checked on decode).
    pub window: u32,
    /// Nominal block size used when splitting the input.
    pub block_size: u32,
    /// Bytes fed to the model before every block, so blocks start with warm context.
    pub preamble: Vec<u8>,
}

/// One block as listed in the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockEntry {
    /// Payload encoding.
    pub codec: BlockCodec,
    /// Decoded length.
    pub raw_len: u32,
    /// Byte offset of the payload within the container.
    pub offset: u64,
    /// Payload length.
    pub len: u32,
    /// CRC32C of the decoded bytes.
    pub raw_crc: u32,
    /// CRC32C of the payload bytes.
    pub payload_crc: u32,
}

/// A block ready to be written.
pub struct EncodedBlock {
    /// Payload encoding.
    pub codec: BlockCodec,
    /// Decoded length.
    pub raw_len: u32,
    /// CRC32C of the decoded bytes.
    pub raw_crc: u32,
    /// Encoded bytes.
    pub payload: Vec<u8>,
}

/// Serialises a complete container.
pub fn write(header: &Header, blocks: &[EncodedBlock]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&header.model_id);
    out.extend_from_slice(&header.window.to_le_bytes());
    out.extend_from_slice(&header.block_size.to_le_bytes());
    out.extend_from_slice(&(header.preamble.len() as u32).to_le_bytes());
    out.extend_from_slice(&header.preamble);
    let header_crc = crc32c(&out);
    out.extend_from_slice(&header_crc.to_le_bytes());

    let mut entries = Vec::with_capacity(blocks.len());
    let mut total: u64 = 0;
    for b in blocks {
        entries.push(BlockEntry {
            codec: b.codec,
            raw_len: b.raw_len,
            offset: out.len() as u64,
            len: b.payload.len() as u32,
            raw_crc: b.raw_crc,
            payload_crc: crc32c(&b.payload),
        });
        out.extend_from_slice(&b.payload);
        total += u64::from(b.raw_len);
    }
    let index_offset = out.len() as u64;
    let index_start = out.len();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in &entries {
        out.push(e.codec.tag());
        out.extend_from_slice(&e.raw_len.to_le_bytes());
        out.extend_from_slice(&e.offset.to_le_bytes());
        out.extend_from_slice(&e.len.to_le_bytes());
        out.extend_from_slice(&e.raw_crc.to_le_bytes());
        out.extend_from_slice(&e.payload_crc.to_le_bytes());
    }
    let index_crc = crc32c(&out[index_start..]);
    out.extend_from_slice(&index_crc.to_le_bytes());
    out.extend_from_slice(&index_offset.to_le_bytes());
    out.extend_from_slice(&total.to_le_bytes());
    out.extend_from_slice(TRAILER_MAGIC);
    out
}

/// Little-endian cursor over untrusted bytes with bounds-checked reads.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or_else(|| Error::Corrupt(format!("{} is truncated", self.what)))?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes")))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
}

/// A parsed and validated container borrowing the underlying bytes.
pub struct Container<'a> {
    header: Header,
    blocks: Vec<BlockEntry>,
    total_len: u64,
    bytes: &'a [u8],
}

impl<'a> Container<'a> {
    /// Parses and validates the header, index and trailer (not the payloads).
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < MAGIC.len() || &bytes[..8] != MAGIC {
            return Err(Error::Corrupt("not a bitgolf container (bad magic)".into()));
        }
        let mut r = Reader { bytes, pos: 8, what: "header" };
        let version = r.u16()?;
        if version != VERSION {
            return Err(Error::Corrupt(format!("unsupported container version {version}")));
        }
        let _flags = r.u16()?;
        let model_id: [u8; 32] = r.take(32)?.try_into().expect("32 bytes");
        let window = r.u32()?;
        let block_size = r.u32()?;
        let preamble_len = r.u32()? as usize;
        if preamble_len > MAX_PREAMBLE {
            return Err(Error::Corrupt(format!("preamble length {preamble_len} too large")));
        }
        let preamble = r.take(preamble_len)?.to_vec();
        let header_end = r.pos;
        if r.u32()? != crc32c(&bytes[..header_end]) {
            return Err(Error::Corrupt("header checksum mismatch".into()));
        }
        let payload_start = r.pos as u64;

        if bytes.len() < payload_start as usize + TRAILER_SIZE + 8 {
            return Err(Error::Corrupt("container is truncated".into()));
        }
        let trailer = &bytes[bytes.len() - TRAILER_SIZE..];
        if &trailer[16..] != TRAILER_MAGIC {
            return Err(Error::Corrupt("missing trailer: the file is truncated".into()));
        }
        let mut t = Reader { bytes: trailer, pos: 0, what: "trailer" };
        let index_offset = t.u64()?;
        let total_len = t.u64()?;
        let index_end = (bytes.len() - TRAILER_SIZE) as u64;
        if index_offset < payload_start || index_offset + 8 > index_end {
            return Err(Error::Corrupt("index offset out of range".into()));
        }
        let index = &bytes[index_offset as usize..index_end as usize];
        let (body, crc) = index.split_at(index.len() - 4);
        if crc32c(body) != u32::from_le_bytes(crc.try_into().expect("4 bytes")) {
            return Err(Error::Corrupt("block index checksum mismatch".into()));
        }
        let mut ir = Reader { bytes: body, pos: 0, what: "block index" };
        let count = ir.u32()? as usize;
        if body.len() != 4 + count * ENTRY_SIZE {
            return Err(Error::Corrupt("block index size does not match its count".into()));
        }
        let mut blocks = Vec::with_capacity(count);
        let mut expected_offset = payload_start;
        let mut sum: u64 = 0;
        for i in 0..count {
            let e = BlockEntry {
                codec: BlockCodec::from_tag(ir.u8()?)?,
                raw_len: ir.u32()?,
                offset: ir.u64()?,
                len: ir.u32()?,
                raw_crc: ir.u32()?,
                payload_crc: ir.u32()?,
            };
            if e.offset != expected_offset || e.raw_len as usize > MAX_BLOCK {
                return Err(Error::Corrupt(format!("block {i} has an invalid offset or length")));
            }
            expected_offset += u64::from(e.len);
            sum += u64::from(e.raw_len);
            blocks.push(e);
        }
        if expected_offset != index_offset || sum != total_len {
            return Err(Error::Corrupt("block index does not cover the payload area".into()));
        }
        let header = Header { model_id, window, block_size, preamble };
        Ok(Self { header, blocks, total_len, bytes })
    }

    /// Container-wide settings.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The block index.
    pub fn blocks(&self) -> &[BlockEntry] {
        &self.blocks
    }

    /// Total decoded length.
    pub fn total_len(&self) -> u64 {
        self.total_len
    }

    /// Size of the whole container in bytes.
    pub fn stored_len(&self) -> usize {
        self.bytes.len()
    }

    /// The payload of block `i`, after verifying its checksum.
    pub fn payload(&self, i: usize) -> Result<&'a [u8]> {
        let e =
            self.blocks.get(i).ok_or_else(|| Error::InvalidArgument(format!("block {i} does not exist")))?;
        let p = &self.bytes[e.offset as usize..e.offset as usize + e.len as usize];
        if crc32c(p) != e.payload_crc {
            return Err(Error::Corrupt(format!("block {i}: payload checksum mismatch")));
        }
        Ok(p)
    }

    /// Fails unless the container was written with `model_id`.
    pub fn require_model(&self, model_id: &[u8; 32]) -> Result<()> {
        if &self.header.model_id != model_id {
            return Err(Error::ModelMismatch { expected: hex(&self.header.model_id), actual: hex(model_id) });
        }
        Ok(())
    }
}
