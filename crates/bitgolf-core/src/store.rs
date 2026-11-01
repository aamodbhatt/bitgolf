//! An append-only key/value store whose values are packed into model-compressed blocks.
//!
//! On disk a store is a directory with two files:
//!
//! * `data.bgs`: a sequence of frames. Each frame holds one compressed block plus the keys
//!   it contains (the persisted key index), each with its offset, length and CRC32C:
//!   `"BGFR" | meta_len u32 | meta_crc32c u32 | meta | payload`, where meta is
//!   `codec u8 | raw_len u32 | raw_crc u32 | payload_len u32 | payload_crc u32 | n u32 |
//!   n x (key_len u16 | key | value_offset u32 | value_len u32 | value_crc u32)`.
//! * `MANIFEST`: model id, block size, preamble, and how many bytes of `data.bgs` are
//!   committed, protected by a CRC32C.
//!
//! A flush appends a frame, fsyncs the data file, then atomically replaces the manifest
//! (write `MANIFEST.tmp`, fsync, rename, fsync the directory). Bytes past the committed length
//! are a torn or unacknowledged write and are discarded on open, so a crash at any point
//! leaves the store at its last completed flush. A `get` decodes only the prefix of the one
//! block holding the value.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::codec::{self, Primer};
use crate::container::{BlockCodec, MAX_BLOCK, MAX_PREAMBLE, crc32c};
use crate::error::{Error, Result, hex};
use crate::model::Model;

const MANIFEST: &str = "MANIFEST";
const MANIFEST_TMP: &str = "MANIFEST.tmp";
const DATA: &str = "data.bgs";
const MANIFEST_MAGIC: &[u8; 8] = b"BGSTORE\0";
const FRAME_MAGIC: &[u8; 4] = b"BGFR";
const VERSION: u16 = 1;
const MAX_META: usize = 64 << 20;
/// Default raw bytes per block.
pub const DEFAULT_STORE_BLOCK: usize = 16 << 10;

/// Settings used when a store is created; an existing store keeps its own.
#[derive(Clone, Debug)]
pub struct StoreOptions {
    /// Raw bytes buffered before a block is sealed. Smaller blocks make `get` faster and
    /// compression slightly worse.
    pub block_size: usize,
    /// Context fed to the model before every block.
    pub preamble: Vec<u8>,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self { block_size: DEFAULT_STORE_BLOCK, preamble: Vec::new() }
    }
}

/// Size and compression statistics.
#[derive(Clone, Debug, serde::Serialize)]
pub struct StoreStats {
    /// Distinct live keys.
    pub keys: usize,
    /// Committed frames (blocks).
    pub blocks: u64,
    /// Raw bytes in committed blocks.
    pub raw_bytes: u64,
    /// Bytes used by `data.bgs` (payloads plus key tables).
    pub stored_bytes: u64,
    /// `stored_bytes * 8 / raw_bytes`.
    pub bits_per_byte: f64,
    /// Blocks coded by the model, by deflate and stored raw.
    pub codec_counts: [u64; 3],
    /// Raw bytes waiting for the next flush.
    pub pending_bytes: usize,
    /// Torn bytes discarded when the store was opened.
    pub recovered_bytes: u64,
}

#[derive(Clone, Copy)]
struct ValueRef {
    frame: u64,
    offset: u32,
    len: u32,
    crc: u32,
}

#[derive(Clone, Copy)]
struct FrameInfo {
    codec: BlockCodec,
    raw_len: u32,
    payload_offset: u64,
    payload_len: u32,
    payload_crc: u32,
}

struct Manifest {
    model_id: [u8; 32],
    block_size: u32,
    preamble: Vec<u8>,
    committed: u64,
    frames: u64,
}

impl Manifest {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MANIFEST_MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.model_id);
        b.extend_from_slice(&self.block_size.to_le_bytes());
        b.extend_from_slice(&(self.preamble.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.preamble);
        b.extend_from_slice(&self.committed.to_le_bytes());
        b.extend_from_slice(&self.frames.to_le_bytes());
        let crc = crc32c(&b);
        b.extend_from_slice(&crc.to_le_bytes());
        b
    }

    fn decode(b: &[u8]) -> Result<Self> {
        let bad = |m: &str| Error::Corrupt(format!("manifest: {m}"));
        if b.len() < 8 + 2 + 32 + 4 + 4 + 16 + 4 || &b[..8] != MANIFEST_MAGIC {
            return Err(bad("bad magic or truncated"));
        }
        let (body, crc) = b.split_at(b.len() - 4);
        if crc32c(body) != u32::from_le_bytes(crc.try_into().expect("4 bytes")) {
            return Err(bad("checksum mismatch"));
        }
        if u16::from_le_bytes([b[8], b[9]]) != VERSION {
            return Err(bad("unsupported version"));
        }
        let model_id: [u8; 32] = b[10..42].try_into().expect("32 bytes");
        let block_size = u32::from_le_bytes(b[42..46].try_into().expect("4 bytes"));
        let plen = u32::from_le_bytes(b[46..50].try_into().expect("4 bytes")) as usize;
        if body.len() != 50 + plen + 16 {
            return Err(bad("length mismatch"));
        }
        let preamble = b[50..50 + plen].to_vec();
        let rest = &b[50 + plen..];
        Ok(Self {
            model_id,
            block_size,
            preamble,
            committed: u64::from_le_bytes(rest[..8].try_into().expect("8 bytes")),
            frames: u64::from_le_bytes(rest[8..16].try_into().expect("8 bytes")),
        })
    }
}

/// The append-only, model-compressed key/value store.
pub struct Store {
    dir: PathBuf,
    model: Arc<Model>,
    primer: Primer,
    manifest: Manifest,
    data: File,
    index: BTreeMap<Vec<u8>, ValueRef>,
    frames: HashMap<u64, FrameInfo>,
    raw_bytes: u64,
    codec_counts: [u64; 3],
    pending: Vec<u8>,
    pending_index: BTreeMap<Vec<u8>, (u32, u32)>,
    pending_keys: Vec<(Vec<u8>, u32, u32)>,
    recovered_bytes: u64,
}

fn sync_dir(dir: &Path) -> Result<()> {
    // Directory fsync makes the rename durable on POSIX; Windows has no equivalent handle.
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

fn write_manifest(dir: &Path, m: &Manifest) -> Result<()> {
    let tmp = dir.join(MANIFEST_TMP);
    let mut f = File::create(&tmp)?;
    f.write_all(&m.encode())?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, dir.join(MANIFEST))?;
    sync_dir(dir)
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn codec_index(c: BlockCodec) -> usize {
    match c {
        BlockCodec::Model => 0,
        BlockCodec::Deflate => 1,
        BlockCodec::Stored => 2,
    }
}

impl Store {
    /// Opens the store in `dir`, creating it with `opts` if it does not exist. Discards any
    /// torn write left by a crash and verifies every frame's key table.
    pub fn open(dir: impl AsRef<Path>, model: Arc<Model>, opts: &StoreOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let manifest_path = dir.join(MANIFEST);
        if !manifest_path.exists() {
            if opts.block_size == 0 || opts.block_size > MAX_BLOCK {
                return Err(Error::InvalidArgument("store block size out of range".into()));
            }
            if opts.preamble.len() > MAX_PREAMBLE {
                return Err(Error::InvalidArgument("store preamble is larger than 64 KiB".into()));
            }
            let data_path = dir.join(DATA);
            if data_path.exists() && fs::metadata(&data_path)?.len() > 0 {
                return Err(Error::Corrupt(format!(
                    "{} exists without a manifest; refusing to overwrite it",
                    data_path.display()
                )));
            }
            File::create(&data_path)?.sync_all()?;
            write_manifest(
                &dir,
                &Manifest {
                    model_id: model.id(),
                    block_size: opts.block_size as u32,
                    preamble: opts.preamble.clone(),
                    committed: 0,
                    frames: 0,
                },
            )?;
        }
        let manifest = Manifest::decode(&fs::read(&manifest_path)?)?;
        if manifest.model_id != model.id() {
            return Err(Error::ModelMismatch { expected: hex(&manifest.model_id), actual: model.id_hex() });
        }
        let _ = fs::remove_file(dir.join(MANIFEST_TMP));
        let data = OpenOptions::new().read(true).write(true).open(dir.join(DATA))?;
        let len = data.metadata()?.len();
        if len < manifest.committed {
            return Err(Error::Corrupt(format!(
                "data file has {len} bytes but the manifest commits {}",
                manifest.committed
            )));
        }
        let recovered_bytes = len - manifest.committed;
        if recovered_bytes > 0 {
            data.set_len(manifest.committed)?;
            data.sync_all()?;
        }
        let primer = Primer::new(&model, &manifest.preamble);
        let mut store = Self {
            dir,
            model,
            primer,
            manifest,
            data,
            index: BTreeMap::new(),
            frames: HashMap::new(),
            raw_bytes: 0,
            codec_counts: [0; 3],
            pending: Vec::new(),
            pending_index: BTreeMap::new(),
            pending_keys: Vec::new(),
            recovered_bytes,
        };
        store.load_frames()?;
        Ok(store)
    }

    fn load_frames(&mut self) -> Result<()> {
        let mut offset = 0u64;
        let mut count = 0u64;
        let mut f = &self.data;
        while offset < self.manifest.committed {
            let corrupt = |m: &str| Error::Corrupt(format!("frame at offset {offset}: {m}"));
            let mut head = [0u8; 12];
            f.seek(SeekFrom::Start(offset))?;
            f.read_exact(&mut head)?;
            let meta_len = le_u32(&head, 4) as usize;
            if &head[..4] != FRAME_MAGIC || !(21..=MAX_META).contains(&meta_len) {
                return Err(corrupt("bad frame header"));
            }
            let mut meta = vec![0u8; meta_len];
            f.read_exact(&mut meta)?;
            if crc32c(&meta) != le_u32(&head, 8) {
                return Err(corrupt("key table checksum mismatch"));
            }
            let info = FrameInfo {
                codec: match meta[0] {
                    0 => BlockCodec::Model,
                    1 => BlockCodec::Deflate,
                    2 => BlockCodec::Stored,
                    _ => return Err(corrupt("unknown codec")),
                },
                raw_len: le_u32(&meta, 1),
                payload_offset: offset + 12 + meta_len as u64,
                payload_len: le_u32(&meta, 9),
                payload_crc: le_u32(&meta, 13),
            };
            let n = le_u32(&meta, 17) as usize;
            let mut at = 21;
            for _ in 0..n {
                if at + 2 > meta.len() {
                    return Err(corrupt("key table truncated"));
                }
                let klen = u16::from_le_bytes([meta[at], meta[at + 1]]) as usize;
                if at + 2 + klen + 12 > meta.len() {
                    return Err(corrupt("key table truncated"));
                }
                let key = meta[at + 2..at + 2 + klen].to_vec();
                at += 2 + klen;
                let r = ValueRef {
                    frame: offset,
                    offset: le_u32(&meta, at),
                    len: le_u32(&meta, at + 4),
                    crc: le_u32(&meta, at + 8),
                };
                at += 12;
                if u64::from(r.offset) + u64::from(r.len) > u64::from(info.raw_len) {
                    return Err(corrupt("value lies outside its block"));
                }
                self.index.insert(key, r);
            }
            self.raw_bytes += u64::from(info.raw_len);
            self.codec_counts[codec_index(info.codec)] += 1;
            self.frames.insert(offset, info);
            offset = info.payload_offset + u64::from(info.payload_len);
            count += 1;
        }
        if offset != self.manifest.committed || count != self.manifest.frames {
            return Err(Error::Corrupt("frames do not match the manifest".into()));
        }
        Ok(())
    }

    /// Stores `value` under `key`, replacing any previous value. Durable after
    /// [`Store::flush`] returns (a full block flushes automatically).
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if key.is_empty() || key.len() > u16::MAX as usize {
            return Err(Error::InvalidArgument("keys must be 1..=65535 bytes".into()));
        }
        if value.len() > MAX_BLOCK / 2 {
            return Err(Error::InvalidArgument("value larger than 32 MiB".into()));
        }
        if !self.pending.is_empty() && self.pending.len() + value.len() > MAX_BLOCK {
            self.flush()?;
        }
        let offset = self.pending.len() as u32;
        self.pending.extend_from_slice(value);
        self.pending_index.insert(key.to_vec(), (offset, value.len() as u32));
        self.pending_keys.push((key.to_vec(), offset, value.len() as u32));
        if self.pending.len() >= self.manifest.block_size as usize {
            self.flush()?;
        }
        Ok(())
    }

    /// Compresses buffered values into a block and commits it durably.
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let block = codec::encode_single(&self.model, &self.primer, &self.pending);
        let mut meta = Vec::new();
        meta.push(match block.codec {
            BlockCodec::Model => 0,
            BlockCodec::Deflate => 1,
            BlockCodec::Stored => 2,
        });
        meta.extend_from_slice(&block.raw_len.to_le_bytes());
        meta.extend_from_slice(&block.raw_crc.to_le_bytes());
        meta.extend_from_slice(&(block.payload.len() as u32).to_le_bytes());
        meta.extend_from_slice(&crc32c(&block.payload).to_le_bytes());
        meta.extend_from_slice(&(self.pending_keys.len() as u32).to_le_bytes());
        for (key, off, len) in &self.pending_keys {
            let value = &self.pending[*off as usize..(*off + *len) as usize];
            meta.extend_from_slice(&(key.len() as u16).to_le_bytes());
            meta.extend_from_slice(key);
            meta.extend_from_slice(&off.to_le_bytes());
            meta.extend_from_slice(&len.to_le_bytes());
            meta.extend_from_slice(&crc32c(value).to_le_bytes());
        }
        let mut frame = Vec::with_capacity(12 + meta.len() + block.payload.len());
        frame.extend_from_slice(FRAME_MAGIC);
        frame.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc32c(&meta).to_le_bytes());
        frame.extend_from_slice(&meta);
        frame.extend_from_slice(&block.payload);

        let frame_offset = self.manifest.committed;
        // A failed earlier flush may have left bytes past the committed end.
        self.data.set_len(frame_offset)?;
        self.data.seek(SeekFrom::Start(frame_offset))?;
        self.data.write_all(&frame)?;
        self.data.sync_data()?;
        let next = Manifest {
            model_id: self.manifest.model_id,
            block_size: self.manifest.block_size,
            preamble: self.manifest.preamble.clone(),
            committed: frame_offset + frame.len() as u64,
            frames: self.manifest.frames + 1,
        };
        write_manifest(&self.dir, &next)?;
        self.manifest = next;

        let info = FrameInfo {
            codec: block.codec,
            raw_len: block.raw_len,
            payload_offset: frame_offset + 12 + meta.len() as u64,
            payload_len: block.payload.len() as u32,
            payload_crc: crc32c(&block.payload),
        };
        for (key, off, len) in self.pending_keys.drain(..) {
            let value = &self.pending[off as usize..(off + len) as usize];
            let r = ValueRef { frame: frame_offset, offset: off, len, crc: crc32c(value) };
            self.index.insert(key, r);
        }
        self.raw_bytes += u64::from(info.raw_len);
        self.codec_counts[codec_index(info.codec)] += 1;
        self.frames.insert(frame_offset, info);
        self.pending.clear();
        self.pending_index.clear();
        Ok(())
    }

    fn read_payload(&self, info: &FrameInfo) -> Result<Vec<u8>> {
        let mut f = &self.data;
        let mut payload = vec![0u8; info.payload_len as usize];
        f.seek(SeekFrom::Start(info.payload_offset))?;
        f.read_exact(&mut payload)?;
        if crc32c(&payload) != info.payload_crc {
            return Err(Error::Corrupt(format!(
                "block at offset {}: payload checksum mismatch",
                info.payload_offset
            )));
        }
        Ok(payload)
    }

    /// Decodes the first `len` bytes of the block in frame `frame`.
    fn block_prefix(&self, frame: u64, len: usize) -> Result<Vec<u8>> {
        let info = self.frames.get(&frame).expect("index refers to a known frame");
        let payload = self.read_payload(info)?;
        codec::decode_prefix(&self.model, &self.primer, info.codec, &payload, info.raw_len as usize, len)
    }

    fn extract(block: &[u8], r: &ValueRef) -> Result<Vec<u8>> {
        let value = &block[r.offset as usize..(r.offset + r.len) as usize];
        if crc32c(value) != r.crc {
            return Err(Error::Corrupt("value checksum mismatch".into()));
        }
        Ok(value.to_vec())
    }

    /// Looks up `key`, decoding only the block prefix that ends with its value.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(&(off, len)) = self.pending_index.get(key) {
            return Ok(Some(self.pending[off as usize..(off + len) as usize].to_vec()));
        }
        let Some(r) = self.index.get(key) else { return Ok(None) };
        let block = self.block_prefix(r.frame, (r.offset + r.len) as usize)?;
        Self::extract(&block, r).map(Some)
    }

    /// All live entries whose key starts with `prefix`, in key order. Each block is decoded
    /// at most once.
    pub fn scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut keys: Vec<&Vec<u8>> = self
            .index
            .range(prefix.to_vec()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k)
            .filter(|k| !self.pending_index.contains_key(*k))
            .collect();
        keys.extend(
            self.pending_index
                .range(prefix.to_vec()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .map(|(k, _)| k),
        );
        keys.sort();
        let mut need: HashMap<u64, usize> = HashMap::new();
        for k in &keys {
            if let Some(r) = self.index.get(*k).filter(|_| !self.pending_index.contains_key(*k)) {
                let end = (r.offset + r.len) as usize;
                let e = need.entry(r.frame).or_default();
                *e = (*e).max(end);
            }
        }
        let mut blocks = HashMap::new();
        for (frame, len) in need {
            blocks.insert(frame, self.block_prefix(frame, len)?);
        }
        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            let value = match self.pending_index.get(k) {
                Some(&(off, len)) => self.pending[off as usize..(off + len) as usize].to_vec(),
                None => {
                    let r = &self.index[k];
                    Self::extract(&blocks[&r.frame], r)?
                }
            };
            out.push((k.clone(), value));
        }
        Ok(out)
    }

    /// Size and compression statistics.
    pub fn stats(&self) -> StoreStats {
        let keys =
            self.index.len() + self.pending_index.keys().filter(|k| !self.index.contains_key(*k)).count();
        let stored = self.manifest.committed;
        StoreStats {
            keys,
            blocks: self.manifest.frames,
            raw_bytes: self.raw_bytes,
            stored_bytes: stored,
            bits_per_byte: if self.raw_bytes > 0 { stored as f64 * 8.0 / self.raw_bytes as f64 } else { 0.0 },
            codec_counts: self.codec_counts,
            pending_bytes: self.pending.len(),
            recovered_bytes: self.recovered_bytes,
        }
    }
}
