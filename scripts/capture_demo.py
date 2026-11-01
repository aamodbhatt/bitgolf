"""Capture the README hero GIF from the running browser demo.

Usage (with `bitgolf serve` listening on 127.0.0.1:8091):
    uv run --with playwright python scripts/capture_demo.py --out docs/assets

Types text into the live heatmap one character at a time, grabbing a frame every few
characters, and also checks in-browser determinism: the logits digest of the golden text
computed by wasm in Chromium must equal the native golden.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

TYPED = (
    "Hello! bitgolf only pays for the bytes it fails to predict.\n"
    '{"user": "ada", "id": 1815, "tags": ["math", "engines"]}\n'
    '{"user": "alan", "id": 1912, "tags": ["math", "enigma"]}\n'
    "Predictable bytes are nearly free; surprises cost bits."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8091/")
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--goldens",
        type=Path,
        default=Path("crates/bitgolf-core/tests/fixtures/goldens.json"),
    )
    parser.add_argument(
        "--heatmap-html", type=Path, help="also screenshot a `bitgolf heatmap` page"
    )
    ns = parser.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    goldens = json.loads(ns.goldens.read_text())
    golden_text = Path(goldens["text"]).read_text(encoding="utf-8")
    small = next(c for c in goldens["cases"] if c["model"].endswith("bitgolf-small.bgm"))

    frames: list[Image.Image] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1180, "height": 760}, device_scale_factor=1)
        page.goto(ns.url)
        page.wait_for_function("document.querySelector('#stat-model').textContent !== '-'")
        page.wait_for_timeout(500)

        digest = page.evaluate("t => window.bitgolf.digest(t)", golden_text)
        if digest != small["logits_sha256"]:
            raise SystemExit(f"browser digest {digest} != native {small['logits_sha256']}")
        print(f"Chromium wasm digest matches native golden: {digest}")

        box = page.locator("#input")
        box.fill("")
        page.wait_for_timeout(300)
        clip = {"x": 0, "y": 0, "width": 1180, "height": 760}
        for i, ch in enumerate(TYPED):
            box.type(ch)
            if i % 3 == 2 or ch == "\n" or i == len(TYPED) - 1:
                page.wait_for_timeout(140)
                shot = page.screenshot(clip=clip)
                frames.append(Image.open(io.BytesIO(shot)).convert("RGB"))
        page.wait_for_timeout(600)
        for _ in range(12):
            frames.append(Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB"))
        if ns.heatmap_html is not None:
            page.goto(ns.heatmap_html.resolve().as_uri())
            page.wait_for_timeout(300)
            page.screenshot(path=str(ns.out / "heatmap.png"))
        browser.close()

    scaled = [f.resize((944, 608), Image.Resampling.LANCZOS) for f in frames]
    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in scaled]
    gif = ns.out / "demo.gif"
    palette[0].save(
        gif,
        save_all=True,
        append_images=palette[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
