//! Store behaviour, persistence, corruption detection and crash recovery.
//!
//! Crashes are simulated on real files: a flush is "interrupted" by restoring the old manifest
//! (crash before the rename) and/or truncating the data file at an arbitrary point inside the
//! new frame (torn append), plus leaving a partial `MANIFEST.tmp` behind.

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bitgolf_core::Error;
use bitgolf_core::store::{Store, StoreOptions};
use proptest::prelude::*;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "bitgolf-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn opts() -> StoreOptions {
    StoreOptions { block_size: 600, preamble: b"{\"k\":".to_vec() }
}

fn open(dir: &Path) -> Result<Store, Error> {
    Store::open(dir, common::tiny_model(), &opts())
}

fn value(i: usize) -> Vec<u8> {
    format!("{{\"k\":{i},\"name\":\"user-{i}\",\"note\":\"value number {i} of the test\"}}\n").into_bytes()
}

#[test]
fn put_get_scan_and_reopen() {
    let dir = temp_dir("basic");
    let mut s = open(&dir).unwrap();
    for i in 0..60 {
        s.put(format!("user/{i:03}").as_bytes(), &value(i)).unwrap();
    }
    s.put(b"user/007", b"replaced").unwrap();
    assert_eq!(s.get(b"user/007").unwrap().unwrap(), b"replaced");
    s.flush().unwrap();
    drop(s);

    let s = open(&dir).unwrap();
    assert_eq!(s.get(b"user/007").unwrap().unwrap(), b"replaced");
    assert_eq!(s.get(b"user/042").unwrap().unwrap(), value(42));
    assert!(s.get(b"user/999").unwrap().is_none());
    let scanned = s.scan(b"user/01").unwrap();
    assert_eq!(scanned.len(), 10);
    assert_eq!(scanned[3], (b"user/013".to_vec(), value(13)));
    let stats = s.stats();
    assert_eq!(stats.keys, 60);
    assert!(stats.blocks > 1);
    assert_eq!(stats.pending_bytes, 0);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn unflushed_values_are_visible_but_not_durable() {
    let dir = temp_dir("pending");
    let mut s = open(&dir).unwrap();
    s.put(b"a", b"1").unwrap();
    assert_eq!(s.get(b"a").unwrap().unwrap(), b"1");
    assert_eq!(s.scan(b"").unwrap().len(), 1);
    drop(s);
    assert!(open(&dir).unwrap().get(b"a").unwrap().is_none());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rejects_a_different_model_and_bad_arguments() {
    let dir = temp_dir("args");
    let mut s = open(&dir).unwrap();
    assert!(matches!(s.put(b"", b"x"), Err(Error::InvalidArgument(_))));
    s.put(b"k", b"v").unwrap();
    s.flush().unwrap();
    drop(s);
    // Rewrite the manifest's model id (and its checksum) to simulate a store made elsewhere.
    let path = dir.join("MANIFEST");
    let mut m = fs::read(&path).unwrap();
    m[10] ^= 0xFF;
    let n = m.len();
    let crc = bitgolf_core::container::crc32c(&m[..n - 4]);
    m[n - 4..].copy_from_slice(&crc.to_le_bytes());
    fs::write(&path, m).unwrap();
    assert!(matches!(open(&dir), Err(Error::ModelMismatch { .. })));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn corruption_is_reported() {
    let dir = temp_dir("corrupt");
    let mut s = open(&dir).unwrap();
    for i in 0..30 {
        s.put(format!("k{i:02}").as_bytes(), &value(i)).unwrap();
    }
    s.flush().unwrap();
    drop(s);
    let data_path = dir.join("data.bgs");
    let pristine = fs::read(&data_path).unwrap();

    // A flipped payload byte (the last byte of the file) fails that block's checksum on get.
    let mut bad = pristine.clone();
    *bad.last_mut().unwrap() ^= 0x10;
    fs::write(&data_path, &bad).unwrap();
    let s = open(&dir).unwrap();
    let results: Vec<_> = (0..30).map(|i| s.get(format!("k{i:02}").as_bytes())).collect();
    assert!(results.iter().any(|r| matches!(r, Err(Error::Corrupt(_)))));
    assert!(results.iter().all(|r| matches!(r, Err(Error::Corrupt(_))) || r.as_ref().unwrap().is_some()));
    drop(s);

    // A flipped key-table byte fails at open.
    let mut bad = pristine.clone();
    bad[20] ^= 0x01;
    fs::write(&data_path, &bad).unwrap();
    assert!(matches!(open(&dir), Err(Error::Corrupt(_))));

    // A data file shorter than the manifest commits is data loss, not a torn write.
    fs::write(&data_path, &pristine[..pristine.len() / 2]).unwrap();
    assert!(matches!(open(&dir), Err(Error::Corrupt(_))));

    // A damaged manifest is reported.
    fs::write(&data_path, &pristine).unwrap();
    let mut m = fs::read(dir.join("MANIFEST")).unwrap();
    m[44] ^= 0x01;
    fs::write(dir.join("MANIFEST"), m).unwrap();
    assert!(matches!(open(&dir), Err(Error::Corrupt(_))));
    fs::remove_dir_all(&dir).unwrap();
}

/// Writes `committed` values and flushes, then writes `extra` values and flushes again,
/// returning the file images after each flush.
struct Images {
    data_1: Vec<u8>,
    manifest_1: Vec<u8>,
    data_2: Vec<u8>,
    manifest_2: Vec<u8>,
}

fn two_flushes(
    dir: &Path,
    committed: &BTreeMap<Vec<u8>, Vec<u8>>,
    extra: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> Images {
    let mut s = open(dir).unwrap();
    for (k, v) in committed {
        s.put(k, v).unwrap();
    }
    s.flush().unwrap();
    let data_1 = fs::read(dir.join("data.bgs")).unwrap();
    let manifest_1 = fs::read(dir.join("MANIFEST")).unwrap();
    for (k, v) in extra {
        s.put(k, v).unwrap();
    }
    s.flush().unwrap();
    drop(s);
    Images {
        data_1,
        manifest_1,
        data_2: fs::read(dir.join("data.bgs")).unwrap(),
        manifest_2: fs::read(dir.join("MANIFEST")).unwrap(),
    }
}

fn entries(n: usize, tag: &str) -> BTreeMap<Vec<u8>, Vec<u8>> {
    (0..n).map(|i| (format!("{tag}/{i:04}").into_bytes(), value(i * 7 + tag.len()))).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Crash during the second flush, anywhere between "nothing written" and "frame fully
    /// written but manifest not yet renamed": the store must reopen at the first flush.
    #[test]
    fn torn_flush_recovers_to_last_commit(n1 in 1usize..40, n2 in 1usize..40, cut_frac in 0.0f64..=1.0, tmp_len in 0usize..80) {
        let dir = temp_dir("torn");
        let (a, b) = (entries(n1, "a"), entries(n2, "b"));
        let img = two_flushes(&dir, &a, &b);
        let torn_len = img.data_1.len() + ((img.data_2.len() - img.data_1.len()) as f64 * cut_frac) as usize;
        fs::write(dir.join("data.bgs"), &img.data_2[..torn_len]).unwrap();
        fs::write(dir.join("MANIFEST"), &img.manifest_1).unwrap();
        fs::write(dir.join("MANIFEST.tmp"), &img.manifest_2[..tmp_len.min(img.manifest_2.len())]).unwrap();

        let mut s = open(&dir).unwrap();
        prop_assert_eq!(s.stats().keys, a.len());
        prop_assert_eq!(s.stats().recovered_bytes as usize, torn_len - img.data_1.len());
        for (k, v) in &a {
            prop_assert_eq!(s.get(k).unwrap(), Some(v.clone()));
        }
        prop_assert!(s.get(b.keys().next().unwrap()).unwrap().is_none());
        prop_assert!(!dir.join("MANIFEST.tmp").exists());
        // The recovered store keeps working.
        s.put(b"after", b"crash").unwrap();
        s.flush().unwrap();
        drop(s);
        let s = open(&dir).unwrap();
        prop_assert_eq!(s.get(b"after").unwrap().unwrap(), b"crash".to_vec());
        prop_assert_eq!(s.stats().keys, a.len() + 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Crash after the rename: everything from both flushes is there.
    #[test]
    fn completed_flush_is_durable(n1 in 1usize..30, n2 in 1usize..30) {
        let dir = temp_dir("durable");
        let (a, b) = (entries(n1, "a"), entries(n2, "b"));
        let img = two_flushes(&dir, &a, &b);
        fs::write(dir.join("data.bgs"), &img.data_2).unwrap();
        fs::write(dir.join("MANIFEST"), &img.manifest_2).unwrap();
        let s = open(&dir).unwrap();
        for (k, v) in a.iter().chain(&b) {
            prop_assert_eq!(s.get(k).unwrap(), Some(v.clone()));
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
