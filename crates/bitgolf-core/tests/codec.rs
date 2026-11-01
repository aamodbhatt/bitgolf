//! End-to-end codec behaviour: round trips, random access, fallbacks, thread independence
//! and corruption detection.

mod common;

use bitgolf_core::codec::{self, CompressOptions};
use bitgolf_core::container::{self, BlockCodec, Container};
use bitgolf_core::{Error, compress, decompress};
use proptest::prelude::*;

fn opts(block_size: usize) -> CompressOptions {
    CompressOptions { block_size, preamble: Vec::new() }
}

#[test]
fn round_trips_text_in_many_blocks() {
    let model = common::tiny_model();
    let data = common::sample_text(20_000);
    let packed = compress(&model, &data, &opts(3000)).unwrap();
    let c = Container::parse(&packed).unwrap();
    assert_eq!(c.blocks().len(), 7);
    assert!(packed.len() < data.len(), "repetitive text must compress");
    assert_eq!(decompress(&model, &packed).unwrap(), data);
}

/// Builds a container whose blocks are all model-coded, whatever their size, so the model
/// path is exercised even with the untrained fixture model (which deflate usually beats).
fn model_container(model: &bitgolf_core::Model, data: &[u8], block: usize, preamble: &[u8]) -> Vec<u8> {
    let primer = codec::Primer::new(model, preamble);
    let blocks: Vec<container::EncodedBlock> = data
        .chunks(block)
        .map(|raw| container::EncodedBlock {
            codec: BlockCodec::Model,
            raw_len: raw.len() as u32,
            raw_crc: container::crc32c(raw),
            payload: codec::model_encode(model, &primer, raw).expect("not abandoned"),
        })
        .collect();
    let header = container::Header {
        model_id: model.id(),
        window: model.config().window as u32,
        block_size: block as u32,
        preamble: preamble.to_vec(),
    };
    container::write(&header, &blocks)
}

#[test]
fn model_coded_blocks_round_trip_in_parallel_groups() {
    let model = common::tiny_model();
    let data = common::sample_text(12_000);
    let packed = model_container(&model, &data, 700, b"prime");
    let c = Container::parse(&packed).unwrap();
    assert!(c.blocks().iter().all(|b| b.codec == BlockCodec::Model));
    assert_eq!(decompress(&model, &packed).unwrap(), data);
    let third = codec::decode_blocks(&model, &c, &[3]).unwrap().pop().unwrap();
    assert_eq!(third, &data[2100..2800]);
}

#[test]
fn round_trips_empty_input() {
    let model = common::tiny_model();
    let packed = compress(&model, &[], &opts(4096)).unwrap();
    assert_eq!(Container::parse(&packed).unwrap().blocks().len(), 0);
    assert!(decompress(&model, &packed).unwrap().is_empty());
}

#[test]
fn binary_blocks_fall_back_and_never_expand_much() {
    let model = common::tiny_model();
    let mut data = common::noise(10_000, 3);
    data.extend(std::iter::repeat_n(0u8, 10_000));
    let packed = compress(&model, &data, &opts(10_000)).unwrap();
    let c = Container::parse(&packed).unwrap();
    assert_eq!(c.blocks()[0].codec, BlockCodec::Stored, "noise is stored");
    assert_ne!(c.blocks()[1].codec, BlockCodec::Stored, "zeros compress");
    assert!(packed.len() < 10_000 + 400);
    assert_eq!(decompress(&model, &packed).unwrap(), data);
}

#[test]
fn every_block_decodes_independently() {
    let model = common::tiny_model();
    let data = common::sample_text(9_000);
    let packed = compress(&model, &data, &opts(2048)).unwrap();
    let c = Container::parse(&packed).unwrap();
    for i in (0..c.blocks().len()).rev() {
        let block = codec::decode_blocks(&model, &c, &[i]).unwrap().pop().unwrap();
        let start = i * 2048;
        assert_eq!(block, &data[start..(start + 2048).min(data.len())]);
    }
}

#[test]
fn preamble_round_trips_and_is_recorded() {
    let model = common::tiny_model();
    let data = common::sample_text(5_000);
    let preamble = b"{\"user\":\"ada\"}\n".to_vec();
    let packed =
        compress(&model, &data, &CompressOptions { block_size: 1000, preamble: preamble.clone() }).unwrap();
    assert_eq!(Container::parse(&packed).unwrap().header().preamble, preamble);
    assert_eq!(decompress(&model, &packed).unwrap(), data);
}

#[test]
fn output_is_independent_of_thread_count() {
    let model = common::tiny_model();
    let data = common::sample_text(30_000);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| compress(&model, &data, &opts(1500)).unwrap())
    };
    let one = run(1);
    assert_eq!(one, run(3));
    assert_eq!(one, run(8));
}

#[test]
fn byte_costs_match_the_coded_size() {
    let model = common::tiny_model();
    let data = common::sample_text(4_000);
    let bits: f64 = codec::byte_costs(&model, &data, &[]).iter().map(|&b| f64::from(b)).sum();
    let payload = codec::model_encode(&model, &codec::Primer::new(&model, &[]), &data).unwrap();
    let payload = payload.len() as f64 * 8.0;
    assert!((payload - bits).abs() < 64.0, "coded {payload} bits vs ideal {bits}");
}

#[test]
fn tampered_header_is_rejected() {
    let model = common::tiny_model();
    let mut packed = compress(&model, b"hello world", &opts(64)).unwrap();
    packed[12] ^= 1; // a model id byte, protected by the header checksum
    assert!(matches!(decompress(&model, &packed), Err(Error::Corrupt(_))));
}

#[test]
fn data_for_another_model_is_rejected() {
    let model = common::tiny_model();
    let data = b"hello world";
    let blocks = codec::encode_blocks(&model, data, &opts(64)).unwrap();
    let header = container::Header { model_id: [7; 32], window: 64, block_size: 64, preamble: Vec::new() };
    let packed = container::write(&header, &blocks);
    assert!(matches!(decompress(&model, &packed), Err(Error::ModelMismatch { .. })));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn any_input_round_trips_through_the_model(data in proptest::collection::vec(any::<u8>(), 0..600), block in 1usize..300) {
        let model = common::tiny_model();
        let packed = model_container(&model, &data, block, &[]);
        prop_assert_eq!(decompress(&model, &packed).unwrap(), data);
    }

    #[test]
    fn any_input_round_trips(data in proptest::collection::vec(any::<u8>(), 0..1500), block in 1usize..700) {
        let model = common::tiny_model();
        let packed = compress(&model, &data, &opts(block)).unwrap();
        prop_assert_eq!(decompress(&model, &packed).unwrap(), data);
    }

    #[test]
    fn corruption_is_detected_never_silent(pos_frac in 0.0f64..1.0, bit in 0u8..8) {
        let model = common::tiny_model();
        let data = common::sample_text(3000);
        let mut packed = model_container(&model, &data, 700, &[]);
        let pos = ((packed.len() - 1) as f64 * pos_frac) as usize;
        packed[pos] ^= 1 << bit;
        // Either an explicit error, or (if the flip hit a byte no structure checks, which the
        // format does not have) the original data. Never different data.
        match decompress(&model, &packed) {
            Ok(out) => prop_assert_eq!(out, data),
            Err(Error::Corrupt(_) | Error::ModelMismatch { .. }) => {}
            Err(e) => prop_assert!(false, "unexpected error {e}"),
        }
    }

    #[test]
    fn truncation_is_detected(cut in 1usize..400) {
        let model = common::tiny_model();
        let data = common::sample_text(2000);
        let packed = compress(&model, &data, &opts(500)).unwrap();
        let cut = cut.min(packed.len());
        prop_assert!(decompress(&model, &packed[..packed.len() - cut]).is_err());
    }
}
