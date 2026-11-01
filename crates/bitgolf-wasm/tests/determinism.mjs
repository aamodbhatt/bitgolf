// Checks the wasm32 build against the native goldens: identical logits and identical
// compressed bytes, plus a cross-target round trip. Run from the repository root:
//   node crates/bitgolf-wasm/tests/determinism.mjs [path/to/pkg]
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const pkg = resolve(process.argv[2] ?? 'web/pkg');
const bindings = await import(pathToFileURL(resolve(pkg, 'bitgolf_wasm.js')).href);
bindings.initSync({ module: readFileSync(resolve(pkg, 'bitgolf_wasm_bg.wasm')) });

const goldens = JSON.parse(readFileSync('crates/bitgolf-core/tests/fixtures/goldens.json', 'utf8'));
const text = readFileSync(goldens.text);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

let failures = 0;
for (const c of goldens.cases) {
  const engine = new bindings.Engine(readFileSync(c.model));
  const logits = engine.logits_digest(text);
  const packed = engine.compress(text, c.block_size);
  const container = sha256(packed);
  const restored = Buffer.from(engine.decompress(packed));
  const ok = logits === c.logits_sha256 && container === c.container_sha256 && restored.equals(text);
  console.log(`${ok ? 'PASS' : 'FAIL'} ${c.model}: logits ${logits.slice(0, 16)} container ${container.slice(0, 16)}`);
  if (!ok) failures += 1;
  engine.free();
}
if (failures > 0) {
  console.error(`${failures} golden case(s) differ between native and wasm32`);
  process.exit(1);
}
console.log(`wasm32 matches native on ${goldens.cases.length} model(s)`);
