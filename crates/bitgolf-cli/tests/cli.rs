//! Drives the real `bitgolf` binary end to end with the tiny fixture model.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures() -> PathBuf {
    root().join("crates/bitgolf-core/tests/fixtures")
}

fn bitgolf(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_bitgolf"))
        .args(args)
        .arg("--models-dir")
        .arg(fixtures())
        .output()
        .expect("run bitgolf");
    assert!(out.status.success(), "bitgolf {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
    out
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bitgolf-cli-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn compress_info_cat_decompress_round_trip() {
    let dir = temp("roundtrip");
    let input = fixtures().join("golden_text.txt");
    let packed = dir.join("text.bg");
    let restored = dir.join("text.out");
    let model = fixtures().join("tiny.bgm");
    bitgolf(&["compress", s(&input), "-o", s(&packed), "--model", s(&model), "--block-size", "300"]);
    let info = String::from_utf8(bitgolf(&["info", s(&packed), "--blocks"]).stdout).unwrap();
    assert!(info.contains("block size  300 bytes"), "{info}");
    // The model is found by the sha256 recorded in the container.
    bitgolf(&["decompress", s(&packed), "-o", s(&restored)]);
    let original = std::fs::read(&input).unwrap();
    assert_eq!(std::fs::read(&restored).unwrap(), original);
    let block = bitgolf(&["cat", s(&packed), "--block", "2"]).stdout;
    assert_eq!(block, &original[600..original.len().min(900)]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn heatmap_and_digest() {
    let dir = temp("heatmap");
    let input = fixtures().join("golden_text.txt");
    let page = dir.join("h.html");
    let model = fixtures().join("tiny.bgm");
    bitgolf(&["heatmap", s(&input), "-o", s(&page), "--model", s(&model)]);
    let html = std::fs::read_to_string(&page).unwrap();
    assert!(html.starts_with("<!doctype html>") && html.contains("data-b="));
    let digest = String::from_utf8(bitgolf(&["digest", s(&input), "--model", s(&model)]).stdout).unwrap();
    let goldens = std::fs::read_to_string(fixtures().join("goldens.json")).unwrap();
    let logits = serde_json::from_str::<serde_json::Value>(&digest).unwrap()["logits_sha256"].clone();
    assert!(goldens.contains(logits.as_str().unwrap()), "CLI digest matches the golden");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn store_commands() {
    let dir = temp("store");
    let kv = dir.join("kv");
    let model = fixtures().join("tiny.bgm");
    let lines = dir.join("lines.jsonl");
    std::fs::write(&lines, "{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n").unwrap();
    bitgolf(&["store", "import", s(&kv), s(&lines), "--prefix", "ev/", "--model", s(&model)]);
    bitgolf(&["store", "put", s(&kv), "note", "--value", "hello"]);
    assert_eq!(bitgolf(&["store", "get", s(&kv), "ev/00000001"]).stdout, b"{\"b\":2}\n");
    assert_eq!(bitgolf(&["store", "get", s(&kv), "note"]).stdout, b"hello");
    let keys = String::from_utf8(bitgolf(&["store", "scan", s(&kv), "ev/", "--keys-only"]).stdout).unwrap();
    assert_eq!(keys, "ev/00000000\nev/00000001\nev/00000002\n");
    let stats = String::from_utf8(bitgolf(&["store", "stats", s(&kv)]).stdout).unwrap();
    assert!(stats.contains("\"keys\": 4"), "{stats}");
    std::fs::remove_dir_all(&dir).unwrap();
}
