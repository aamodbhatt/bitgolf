//! Self-contained HTML page colouring every character by the bits it cost to store.

use std::fmt::Write;

/// Inputs for one heatmap page.
pub struct Heatmap<'a> {
    /// Displayed file name.
    pub title: &'a str,
    /// Model preset name and short id.
    pub model: String,
    /// The text that was scored.
    pub data: &'a [u8],
    /// Bits spent on each byte of `data`.
    pub costs: &'a [f32],
    /// Size of the same bytes under deflate level 9 (for comparison).
    pub deflate_bytes: usize,
}

fn escape(out: &mut String, c: char) {
    match c {
        '<' => out.push_str("&lt;"),
        '>' => out.push_str("&gt;"),
        '&' => out.push_str("&amp;"),
        '"' => out.push_str("&quot;"),
        c => out.push(c),
    }
}

/// Bucket 0..=9 for a character's cost; the stylesheet maps buckets to colours.
fn bucket(bits: f32) -> usize {
    match bits {
        b if b < 0.25 => 0,
        b if b < 0.5 => 1,
        b if b < 1.0 => 2,
        b if b < 1.5 => 3,
        b if b < 2.0 => 4,
        b if b < 3.0 => 5,
        b if b < 4.0 => 6,
        b if b < 6.0 => 7,
        b if b < 8.0 => 8,
        _ => 9,
    }
}

const COLORS: [&str; 10] = [
    "transparent",
    "#1d2b53",
    "#2a3f7a",
    "#3b5aa8",
    "#6a4fb0",
    "#a3459c",
    "#d1495b",
    "#ec6d3c",
    "#f59e2b",
    "#ffd23f",
];

/// Renders the page.
pub fn render(h: &Heatmap<'_>) -> String {
    let total_bits: f64 = h.costs.iter().map(|&c| f64::from(c)).sum();
    let bpb = total_bits / h.data.len().max(1) as f64;
    let deflate_bpb = h.deflate_bytes as f64 * 8.0 / h.data.len().max(1) as f64;
    let mut body = String::with_capacity(h.data.len() * 24);
    // Walk characters with their byte widths so a multi-byte UTF-8 character shows the sum
    // of its bytes' costs; invalid bytes render as U+FFFD one byte at a time.
    let mut chars: Vec<(char, usize)> = Vec::with_capacity(h.data.len());
    for chunk in h.data.utf8_chunks() {
        chars.extend(chunk.valid().chars().map(|c| (c, c.len_utf8())));
        chars.extend(chunk.invalid().iter().map(|_| ('\u{fffd}', 1)));
    }
    let mut byte = 0usize;
    for (c, width) in chars {
        let bits: f32 = h.costs[byte..byte + width].iter().sum();
        byte += width;
        if c == '\n' {
            let _ = writeln!(body, "<span class=\"b{}\" data-b=\"{bits:.2}\">\u{21b5}</span>", bucket(bits));
            continue;
        }
        let _ = write!(body, "<span class=\"b{}\" data-b=\"{bits:.2}\">", bucket(bits));
        escape(&mut body, c);
        body.push_str("</span>");
    }
    let mut css = String::new();
    for (i, color) in COLORS.iter().enumerate() {
        let fg = if i >= 7 { "#14131a" } else { "#e9e6f2" };
        let _ = writeln!(css, ".b{i}{{background:{color};color:{fg}}}");
    }
    let legend: String = COLORS.iter().map(|c| format!("<i style=\"background:{c}\"></i>")).collect();
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>bitgolf heatmap: {title}</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0f0e14;color:#e9e6f2;font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
header{{position:sticky;top:0;background:#0f0e14ee;backdrop-filter:blur(6px);padding:16px 24px;border-bottom:1px solid #2a2836}}
h1{{margin:0 0 4px;font-size:18px}} .stats{{color:#a9a4bb;font-size:13px}} .stats b{{color:#ffd23f}}
.legend{{display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:#a9a4bb}}
.legend i{{display:inline-block;width:22px;height:12px;border-radius:2px;border:1px solid #2a2836}}
pre{{margin:0;padding:20px 24px;white-space:pre-wrap;word-break:break-word;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
pre span{{border-radius:2px}} pre span:hover{{outline:1px solid #fff}}
#tip{{position:fixed;pointer-events:none;background:#000;color:#fff;padding:2px 6px;border-radius:4px;font:12px ui-monospace,monospace;display:none}}
{css}</style></head><body>
<header><h1>{title}</h1>
<div class="stats">model <b>{model}</b> &middot; {n} bytes &middot; <b>{bpb:.3} bits/byte</b> &middot; deflate -9: {deflate_bpb:.3} bits/byte</div>
<div class="legend">0 bits {legend} 8+ bits &nbsp; (hover a character for its cost)</div></header>
<pre id="t">{body}</pre><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.getElementById('t').addEventListener('mousemove',e=>{{const b=e.target.dataset.b;if(b===undefined){{tip.style.display='none';return}}tip.textContent=b+' bits';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.style.display='block'}});
</script></body></html>
"#,
        title = h.title.replace(['<', '>', '&', '"'], "_"),
        model = h.model,
        n = h.data.len(),
    )
}
