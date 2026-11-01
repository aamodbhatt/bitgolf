//! The Rust engine reproduces the PyTorch reference (int8 fake quantisation) on the tiny
//! fixture, across more positions than the attention window.

mod common;

use bitgolf_core::model::{Stream, Workspace};

#[derive(serde::Deserialize)]
struct Expected {
    model_sha256: String,
    text: String,
    logits: Vec<Vec<f32>>,
}

#[test]
fn engine_matches_pytorch_reference() {
    let model = common::tiny_model();
    let expected: Expected =
        serde_json::from_slice(&std::fs::read(common::fixture("tiny_expected.json")).unwrap()).unwrap();
    assert_eq!(model.id_hex(), expected.model_sha256);
    let text = expected.text.as_bytes();
    assert!(text.len() > model.config().window, "fixture must exceed the window");
    let mut stream = Stream::new(&model);
    let mut ws = Workspace::new();
    let mut logits = vec![0.0f32; 256];
    let mut worst = 0.0f32;
    for (pos, &b) in text.iter().enumerate() {
        model.step(&mut stream, b, &mut ws, &mut logits);
        let want = &expected.logits[pos];
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (got, want) in logits.iter().zip(want) {
            worst = worst.max((got - want).abs() / scale);
        }
    }
    // Rounding can flip an occasional int8 value between the two implementations; anything
    // structural (layout, window, scales) would show up as errors of order 1.
    assert!(worst < 0.02, "max relative logit error {worst}");
}
