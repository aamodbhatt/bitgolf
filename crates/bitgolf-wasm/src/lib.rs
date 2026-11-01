//! WebAssembly bindings: the same deterministic engine as the native CLI, so a `.bg` file
//! written in the browser decodes natively and vice versa.

use std::rc::Rc;

use bitgolf_core::codec::{self, CompressOptions};
use bitgolf_core::container::Container;
use bitgolf_core::error::hex;
use bitgolf_core::freq::FreqTable;
use bitgolf_core::model::{Stream, Workspace};
use bitgolf_core::{Model, compress, decompress};
use wasm_bindgen::prelude::*;

fn js_err(e: bitgolf_core::Error) -> JsError {
    JsError::new(&e.to_string())
}

/// A loaded model plus the codec operations that use it.
#[wasm_bindgen]
pub struct Engine {
    model: Rc<Model>,
}

#[wasm_bindgen]
impl Engine {
    /// Loads a `.bgm` model from its bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Engine, JsError> {
        Ok(Self { model: Rc::new(Model::from_bytes(bytes).map_err(js_err)?) })
    }

    /// JSON description: name, parameter count, window, sha256.
    pub fn info(&self) -> String {
        let c = self.model.config();
        serde_json::json!({
            "name": c.name,
            "params": self.model.num_params(),
            "window": c.window,
            "layers": c.n_layers,
            "d_model": c.d_model,
            "sha256": self.model.id_hex(),
        })
        .to_string()
    }

    /// Compresses `data` into a `.bg` container.
    pub fn compress(&self, data: &[u8], block_size: usize) -> Result<Vec<u8>, JsError> {
        compress(&self.model, data, &CompressOptions { block_size, preamble: Vec::new() }).map_err(js_err)
    }

    /// Decompresses a `.bg` container written by any bitgolf build.
    pub fn decompress(&self, packed: &[u8]) -> Result<Vec<u8>, JsError> {
        decompress(&self.model, packed).map_err(js_err)
    }

    /// Bits spent on each byte when `data` is coded as one block.
    pub fn byte_costs(&self, data: &[u8]) -> Vec<f32> {
        codec::byte_costs(&self.model, data, &[])
    }

    /// Hex sha256 over every logit's bit pattern while reading `data` (determinism check).
    pub fn logits_digest(&self, data: &[u8]) -> String {
        hex(&codec::logits_digest(&self.model, data))
    }

    /// Starts an incremental scoring session for live typing (up to `max_len` bytes).
    pub fn session(&self, max_len: usize) -> Result<Session, JsError> {
        let stream =
            Stream::with_capacity(&self.model, max_len.max(self.model.config().window)).map_err(js_err)?;
        Ok(Session {
            model: Rc::clone(&self.model),
            fresh: stream.clone(),
            stream,
            table: FreqTable::uniform(),
            ws: Workspace::new(),
            history: Vec::new(),
            costs: Vec::new(),
            max_len,
        })
    }
}

/// Incremental per-byte costs for text that changes a little at a time: only bytes after
/// the first edited position are re-scored, by rewinding the KV cache.
#[wasm_bindgen]
pub struct Session {
    model: Rc<Model>,
    fresh: Stream,
    stream: Stream,
    table: FreqTable,
    ws: Workspace,
    history: Vec<u8>,
    costs: Vec<f32>,
    max_len: usize,
}

#[wasm_bindgen]
impl Session {
    /// Scores `text`, reusing work for the prefix shared with the previous call. Returns the
    /// bits for every byte.
    pub fn update(&mut self, text: &[u8]) -> Result<Vec<f32>, JsError> {
        if text.len() > self.max_len {
            return Err(JsError::new(&format!("text longer than {} bytes", self.max_len)));
        }
        let shared = self.history.iter().zip(text).take_while(|(a, b)| a == b).count();
        let window = self.model.config().window;
        let mut logits = [0.0f32; 256];
        if shared == 0 {
            self.stream = self.fresh.clone();
            self.table = FreqTable::uniform();
        } else if shared < self.history.len() || shared < text.len() {
            self.stream.rewind(shared as u64 - 1, window).map_err(js_err)?;
            self.model.step(&mut self.stream, text[shared - 1], &mut self.ws, &mut logits);
            self.table = FreqTable::from_logits(&logits);
        }
        self.costs.truncate(shared);
        for i in shared..text.len() {
            self.costs.push(self.table.cost_bits(text[i]) as f32);
            if i + 1 < text.len() {
                self.model.step(&mut self.stream, text[i], &mut self.ws, &mut logits);
                self.table = FreqTable::from_logits(&logits);
            }
        }
        self.history = text.to_vec();
        Ok(self.costs.clone())
    }
}

/// JSON description of a `.bg` container (no model needed).
#[wasm_bindgen]
pub fn container_info(packed: &[u8]) -> Result<String, JsError> {
    let c = Container::parse(packed).map_err(js_err)?;
    let blocks: Vec<_> = c
        .blocks()
        .iter()
        .map(|b| serde_json::json!({"codec": b.codec.name(), "raw": b.raw_len, "stored": b.len}))
        .collect();
    Ok(serde_json::json!({
        "model": hex(&c.header().model_id),
        "raw_bytes": c.total_len(),
        "stored_bytes": c.stored_len(),
        "block_size": c.header().block_size,
        "blocks": blocks,
    })
    .to_string())
}
