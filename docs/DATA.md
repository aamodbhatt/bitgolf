# Data provenance

## Training mixture (`python -m bitgolf_train.prepare`)

| Source | License | Share of batches | Bytes (train) |
|---|---|---:|---:|
| FineWeb-Edu `sample/10BT`, shard `000_00000.parquet`, first 131,263 documents | ODC-By 1.0 | 70% | 623.6 MB |
| Go standard library sources (`$GOROOT/src`, no tests or testdata) | BSD-3-Clause | 15% (code) | 62.9 MB |
| CPython 3.12 standard library (`Lib/`, no tests) | PSF-2.0 | (code) | 15.4 MB |
| JSON Lines and CSV rendered from FineWeb-Edu row metadata (id, url, dump, score, ...) | ODC-By 1.0 | 15% (structured) | |
| Synthetic logs (Apache combined, application, JSON) from a seeded generator in `prepare.py` | this repository (MIT) | (structured) | 62.5 MB total structured |

1% of documents per source are held out as validation (`*_val.bin`). Exact byte counts and
sha256 digests are written to `data/train/manifest.json` by the preparation step.

## Evaluation corpora (`python -m bitgolf_train.corpora`), never trained on

| File | Bytes | Source |
|---|---:|---|
| `enwik8` | 100,000,000 | Hutter Prize, mattmahoney.net/dc/enwik8.zip |
| `alice29`, `lcet10`, `plrabn12` | 152,089 / 426,754 / 481,861 | Canterbury corpus |
| `dickens` | 10,192,446 | Silesia corpus |
| `code` | 1,084,532 | Go `net/http` and CPython `asyncio`, excluded from training by path (`prepare.HELD_OUT_CODE`) |
| `jsonlog` | 813,687 | npm's bundled `package.json` manifests as JSON Lines, followed by real `dpkg.log` and `apt/history.log` from the training machine (the log tail varies by machine; the 256 KB benchmark slice rebuilt on a second pod gave identical gzip, xz, zstd and brotli sizes) |
| `fineweb_val_1m` | 1,000,000 | first 1 MB of the held-out FineWeb-Edu validation split, `prose_val.bin` with its NUL document separators turned into newlines (in-distribution, unseen documents) |
| `enwik8_1m` | 1,000,000 | first 1 MB of enwik8 |

No evaluation file is read by the training pipeline. FineWeb-Edu is a web crawl, so incidental
overlap with public-domain books or Wikipedia text cannot be ruled out; the models' weaker
results on exactly those corpora suggest it is not material.
