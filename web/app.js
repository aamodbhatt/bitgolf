// bitgolf browser demo: live per-character surprisal and in-browser .bg compression.
const $ = (sel) => document.querySelector(sel);
const encoder = new TextEncoder();

// Same ten-step palette as `bitgolf heatmap`, from "free" to "8+ bits".
const BUCKETS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8];
const bucket = (bits) => {
  const i = BUCKETS.findIndex((b) => bits < b);
  return i === -1 ? 9 : i;
};

const SAMPLE = `Compression is prediction. A model that guesses the next byte well can store it in fewer bits.
{"id": 1815, "name": "Ada Lovelace", "tags": ["analytical engine", "notes"]}
2025-10-31T17:03:12Z INFO request completed status=200 latency_ms=12
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.`;

class WorkerClient {
  constructor() {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      this.pending.delete(data.id);
      if (data.ok) p.resolve(data);
      else p.reject(new Error(data.error));
    };
  }
  call(type, payload = {}, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, ...payload }, transfer);
    });
  }
}

const client = new WorkerClient();
let maxLiveBytes = 8192;
let scoring = false;
let queued = null;

async function gzipBytes(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return (await new Response(stream).arrayBuffer()).byteLength;
}

function renderHeatmap(bytes, costs) {
  // Group bytes into characters so multi-byte UTF-8 shows the sum of its bytes' bits.
  const frag = document.createDocumentFragment();
  const text = new TextDecoder().decode(bytes);
  let at = 0;
  for (const ch of text) {
    const width = encoder.encode(ch).length;
    let bits = 0;
    for (let i = at; i < at + width && i < costs.length; i++) bits += costs[i];
    at += width;
    const span = document.createElement('span');
    span.className = `b${bucket(bits)}`;
    span.dataset.b = bits.toFixed(2);
    span.textContent = ch === '\n' ? '↵\n' : ch;
    frag.append(span);
  }
  $('#heatmap').replaceChildren(frag);
}

async function score(text) {
  if (scoring) {
    queued = text;
    return;
  }
  scoring = true;
  try {
    let bytes = encoder.encode(text);
    if (bytes.length > maxLiveBytes) bytes = bytes.slice(0, maxLiveBytes);
    const started = performance.now();
    const { costs } = await client.call('score', { text: bytes });
    const ms = performance.now() - started;
    renderHeatmap(bytes, costs);
    const bits = costs.reduce((a, b) => a + b, 0);
    const n = Math.max(bytes.length, 1);
    const gz = bytes.length ? await gzipBytes(bytes) : 0;
    $('#stat-bytes').textContent = bytes.length.toLocaleString();
    $('#stat-model').textContent = bytes.length ? (bits / n).toFixed(3) : '-';
    $('#stat-gzip').textContent = bytes.length ? ((gz * 8) / n).toFixed(3) : '-';
    $('#stat-ratio').textContent = bytes.length && bits > 0 ? `${((gz * 8) / bits).toFixed(1)}x` : '-';
    $('#stat-ms').textContent = `${ms.toFixed(0)} ms`;
  } finally {
    scoring = false;
    if (queued !== null) {
      const next = queued;
      queued = null;
      score(next);
    }
  }
}

async function loadModel(file) {
  $('#model-info').textContent = 'loading model...';
  const { info, maxLiveBytes: limit } = await client.call('load', { url: `models/${file}` });
  maxLiveBytes = limit;
  $('#model-info').textContent =
    `${info.name}: ${(info.params / 1e6).toFixed(2)}M params, ${info.layers} layers, ` +
    `${info.window}-byte window, sha256 ${info.sha256.slice(0, 12)}`;
  await score($('#input').value);
}

async function listModels() {
  try {
    const res = await fetch('models/index.json');
    if (res.ok) return await res.json();
  } catch {
    // Static hosting without `bitgolf serve`: fall back to the bundled names.
  }
  return [{ file: 'bitgolf-nano.bgm', name: 'nano' }, { file: 'bitgolf-small.bgm', name: 'small' }];
}

function download(bytes, name) {
  const url = URL.createObjectURL(new Blob([bytes]));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function handleFile(file) {
  const data = new Uint8Array(await file.arrayBuffer());
  const isContainer = data.length >= 8 && new TextDecoder().decode(data.slice(0, 7)) === 'BITGOLF';
  const out = $('#file-result');
  out.textContent = isContainer ? 'decompressing...' : 'compressing...';
  try {
    if (isContainer) {
      const { raw, ms, info } = await client.call('decompress', { data }, [data.buffer]);
      out.textContent =
        `decoded ${raw.length.toLocaleString()} bytes from ${info.blocks.length} blocks ` +
        `in ${(ms / 1000).toFixed(1)} s; checksums verified`;
      download(raw, file.name.replace(/\.bg$/, '') || 'decoded');
    } else {
      const blockSize = Number($('#block-size').value);
      const { packed, ms, info } = await client.call('compress', { data, blockSize }, [data.buffer]);
      const bpb = ((packed.length * 8) / Math.max(info.raw_bytes, 1)).toFixed(3);
      const model = info.blocks.filter((b) => b.codec === 'model').length;
      out.textContent =
        `${info.raw_bytes.toLocaleString()} -> ${packed.length.toLocaleString()} bytes ` +
        `(${bpb} bits/byte) in ${(ms / 1000).toFixed(1)} s; ${model}/${info.blocks.length} blocks model-coded`;
      download(packed, `${file.name}.bg`);
    }
  } catch (err) {
    out.textContent = `error: ${err.message}`;
  }
}

async function main() {
  const input = $('#input');
  input.value = SAMPLE;
  let timer = 0;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => score(input.value), 60);
  });
  const select = $('#model');
  const models = await listModels();
  for (const m of models) select.append(new Option(m.name, m.file));
  const preferred = models.find((m) => m.name === 'small') ?? models[0];
  select.value = preferred.file;
  select.addEventListener('change', () => loadModel(select.value));
  $('#file').addEventListener('change', (e) => e.target.files[0] && handleFile(e.target.files[0]));
  const drop = $('#drop');
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  const tip = $('#tip');
  $('#heatmap').addEventListener('mousemove', (e) => {
    const b = e.target.dataset?.b;
    if (b === undefined) {
      tip.hidden = true;
      return;
    }
    tip.textContent = `${b} bits`;
    tip.style.left = `${e.clientX + 12}px`;
    tip.style.top = `${e.clientY + 12}px`;
    tip.hidden = false;
  });
  await loadModel(select.value);
}

main().catch((err) => {
  $('#model-info').textContent = `failed to start: ${err.message}`;
});

// Exposed for automated checks (determinism in the browser and screenshot capture).
window.bitgolf = {
  digest: async (text) => (await client.call('digest', { data: encoder.encode(text) })).digest,
  score,
};
