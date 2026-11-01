// Runs the wasm engine off the main thread so typing stays smooth while the model works.
import init, { Engine, container_info } from './pkg/bitgolf_wasm.js';

let engine = null;
let session = null;
const MAX_LIVE_BYTES = 8192;

const handlers = {
  async load({ url }) {
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    engine?.free();
    session?.free();
    engine = new Engine(bytes);
    session = engine.session(MAX_LIVE_BYTES);
    return { info: JSON.parse(engine.info()), maxLiveBytes: MAX_LIVE_BYTES };
  },
  score({ text }) {
    const costs = session.update(text);
    return { costs };
  },
  compress({ data, blockSize }) {
    const started = performance.now();
    const packed = engine.compress(data, blockSize);
    return { packed, ms: performance.now() - started, info: JSON.parse(container_info(packed)) };
  },
  decompress({ data }) {
    const started = performance.now();
    const info = JSON.parse(container_info(data));
    const raw = engine.decompress(data);
    return { raw, ms: performance.now() - started, info };
  },
  digest({ data }) {
    return { digest: engine.logits_digest(data) };
  },
};

const ready = init();

self.onmessage = async ({ data: msg }) => {
  try {
    await ready;
    const result = await handlers[msg.type](msg);
    self.postMessage({ id: msg.id, ok: true, ...result });
  } catch (err) {
    self.postMessage({ id: msg.id, ok: false, error: String(err?.message ?? err) });
  }
};
