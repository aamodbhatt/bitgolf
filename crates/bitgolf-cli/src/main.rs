//! `bitgolf`: compress text with a byte-level transformer and a range coder.

mod heatmap;
mod models;
mod serve;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use bitgolf_core::codec::{self, CompressOptions, DEFAULT_BLOCK_SIZE};
use bitgolf_core::container::{BlockCodec, Container};
use bitgolf_core::error::hex;
use bitgolf_core::store::{DEFAULT_STORE_BLOCK, Store, StoreOptions};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bitgolf", version, about = "Neural text compression with bit-exact inference")]
struct Cli {
    /// Worker threads (default: all available).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Directory holding .bgm models (default: $BITGOLF_MODELS or ./models).
    #[arg(long, global = true)]
    models_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct ModelArg {
    /// Model preset name (nano, small) or path to a .bgm file.
    #[arg(long, short)]
    model: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Compress a file into a .bg container.
    Compress {
        input: PathBuf,
        /// Output path (default: <input>.bg).
        #[arg(long, short)]
        output: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArg,
        /// Raw bytes per independently decodable block.
        #[arg(long, default_value_t = DEFAULT_BLOCK_SIZE)]
        block_size: usize,
        /// File whose bytes prime the model before every block.
        #[arg(long)]
        preamble: Option<PathBuf>,
    },
    /// Decompress a .bg container (the model is found by its recorded hash).
    Decompress {
        input: PathBuf,
        /// Output path (default: input without .bg, or stdout with "-").
        #[arg(long, short)]
        output: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Decode a single block to stdout (random access).
    Cat {
        input: PathBuf,
        /// Block index.
        #[arg(long)]
        block: usize,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Describe a .bg container.
    Info {
        input: PathBuf,
        /// List every block.
        #[arg(long)]
        blocks: bool,
    },
    /// Measure ratio and throughput, verifying the round trip.
    Bench {
        input: PathBuf,
        #[command(flatten)]
        model: ModelArg,
        /// Thread counts to measure.
        #[arg(long, value_delimiter = ',', default_value = "1")]
        thread_counts: Vec<usize>,
        #[arg(long, default_value_t = DEFAULT_BLOCK_SIZE)]
        block_size: usize,
        /// Only use the first N bytes of the input.
        #[arg(long)]
        limit: Option<usize>,
        /// Prime each block with the first N bytes of the input.
        #[arg(long, default_value_t = 0)]
        prime_head: usize,
    },
    /// Write an HTML page colouring each character by the bits it cost.
    Heatmap {
        input: PathBuf,
        #[arg(long, short)]
        output: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArg,
        /// Score at most this many bytes.
        #[arg(long, default_value_t = 32 << 10)]
        max_bytes: usize,
    },
    /// Print digests of the model's logits and of the compressed bytes (determinism checks).
    Digest {
        input: PathBuf,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Key/value store whose values are packed into compressed blocks.
    Store {
        #[command(subcommand)]
        command: StoreCommand,
    },
    /// Serve the browser demo.
    Serve {
        /// Address to bind.
        #[arg(long, default_value = "127.0.0.1:8091")]
        addr: String,
        /// Directory with index.html and the wasm package.
        #[arg(long, default_value = "web")]
        web: PathBuf,
    },
}

#[derive(Subcommand)]
enum StoreCommand {
    /// Store one value (from --value, --file or stdin) and flush.
    Put {
        dir: PathBuf,
        key: String,
        #[arg(long, conflicts_with = "file")]
        value: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Import every line of a file as a value under keys <prefix><line number>.
    Import {
        dir: PathBuf,
        file: PathBuf,
        #[arg(long, default_value = "")]
        prefix: String,
        #[arg(long, default_value_t = DEFAULT_STORE_BLOCK)]
        block_size: usize,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Print a value.
    Get {
        dir: PathBuf,
        key: String,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Print every key (and value) with a prefix.
    Scan {
        dir: PathBuf,
        #[arg(default_value = "")]
        prefix: String,
        /// Print keys only.
        #[arg(long)]
        keys_only: bool,
        #[command(flatten)]
        model: ModelArg,
    },
    /// Print size and compression statistics as JSON.
    Stats {
        dir: PathBuf,
        #[command(flatten)]
        model: ModelArg,
    },
}

fn read_input(path: &Path) -> Result<Vec<u8>> {
    if path.as_os_str() == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        return Ok(buf);
    }
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

fn write_output(path: &Path, data: &[u8]) -> Result<()> {
    if path.as_os_str() == "-" {
        std::io::stdout().write_all(data)?;
        return Ok(());
    }
    std::fs::write(path, data).with_context(|| format!("writing {}", path.display()))
}

fn bpb(stored: usize, raw: usize) -> f64 {
    stored as f64 * 8.0 / raw.max(1) as f64
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }
    let dir = models::models_dir(cli.models_dir.as_deref());
    let default_model = || models::DEFAULT_MODEL.to_string();
    match cli.command {
        Command::Compress { input, output, model, block_size, preamble } => {
            let model = models::load(&model.model.unwrap_or_else(default_model), &dir)?;
            let data = read_input(&input)?;
            let preamble = preamble.map(|p| read_input(&p)).transpose()?.unwrap_or_default();
            let started = Instant::now();
            let packed = codec::compress(&model, &data, &CompressOptions { block_size, preamble })?;
            let secs = started.elapsed().as_secs_f64();
            let out = output.unwrap_or_else(|| PathBuf::from(format!("{}.bg", input.display())));
            write_output(&out, &packed)?;
            eprintln!(
                "{} -> {}: {} -> {} bytes, {:.3} bits/byte, {:.1} KB/s",
                input.display(),
                out.display(),
                data.len(),
                packed.len(),
                bpb(packed.len(), data.len()),
                data.len() as f64 / 1e3 / secs
            );
        }
        Command::Decompress { input, output, model } => {
            let packed = read_input(&input)?;
            let c = Container::parse(&packed)?;
            let model = models::resolve(model.model.as_deref(), &dir, &c.header().model_id)?;
            let started = Instant::now();
            let data = codec::decompress(&model, &packed)?;
            let secs = started.elapsed().as_secs_f64();
            let out = output.unwrap_or_else(|| {
                let s = input.display().to_string();
                PathBuf::from(s.strip_suffix(".bg").map_or_else(|| format!("{s}.out"), str::to_string))
            });
            write_output(&out, &data)?;
            eprintln!(
                "{} -> {}: {} bytes, {:.1} KB/s",
                input.display(),
                out.display(),
                data.len(),
                data.len() as f64 / 1e3 / secs
            );
        }
        Command::Cat { input, block, model } => {
            let packed = read_input(&input)?;
            let c = Container::parse(&packed)?;
            let model = models::resolve(model.model.as_deref(), &dir, &c.header().model_id)?;
            let mut out = codec::decode_blocks(&model, &c, &[block])?;
            std::io::stdout().write_all(&out.pop().expect("one block"))?;
        }
        Command::Info { input, blocks } => info(&input, blocks)?,
        Command::Bench { input, model, thread_counts, block_size, limit, prime_head } => {
            let model = models::load(&model.model.unwrap_or_else(default_model), &dir)?;
            let mut data = read_input(&input)?;
            if let Some(n) = limit {
                data.truncate(n);
            }
            let preamble = data[..prime_head.min(data.len())].to_vec();
            let name = input.file_name().and_then(|n| n.to_str()).unwrap_or("input").to_string();
            for threads in thread_counts {
                let report = bench(&model, &name, &data, block_size, &preamble, threads)?;
                println!("{}", serde_json::to_string(&report)?);
            }
        }
        Command::Heatmap { input, output, model, max_bytes } => {
            let model = models::load(&model.model.unwrap_or_else(default_model), &dir)?;
            let mut data = read_input(&input)?;
            data.truncate(max_bytes);
            let costs = codec::byte_costs(&model, &data, &[]);
            let page = heatmap::render(&heatmap::Heatmap {
                title: &input.display().to_string(),
                model: format!("{} ({})", model.config().name, &model.id_hex()[..12]),
                data: &data,
                costs: &costs,
                deflate_bytes: codec::choose_codec(&data, None).payload.len(),
            });
            let out = output.unwrap_or_else(|| PathBuf::from(format!("{}.heatmap.html", input.display())));
            write_output(&out, page.as_bytes())?;
            let total: f64 = costs.iter().map(|&c| f64::from(c)).sum();
            eprintln!(
                "wrote {} ({:.3} bits/byte over {} bytes)",
                out.display(),
                total / data.len().max(1) as f64,
                data.len()
            );
        }
        Command::Digest { input, model } => {
            let model = models::load(&model.model.unwrap_or_else(default_model), &dir)?;
            let data = read_input(&input)?;
            let packed =
                codec::compress(&model, &data, &CompressOptions { block_size: 4096, preamble: Vec::new() })?;
            let packed_sha = bitgolf_core::container::crc32c(&packed);
            println!(
                "{}",
                serde_json::json!({
                    "model": model.id_hex(),
                    "logits_sha256": hex(&codec::logits_digest(&model, &data)),
                    "container_bytes": packed.len(),
                    "container_crc32c": format!("{packed_sha:08x}"),
                })
            );
        }
        Command::Store { command } => store(command, &dir)?,
        Command::Serve { addr, web } => serve::serve(&addr, &web, &dir)?,
    }
    Ok(())
}

fn info(input: &Path, list: bool) -> Result<()> {
    let packed = read_input(input)?;
    let c = Container::parse(&packed)?;
    let h = c.header();
    let mut counts = [0usize; 3];
    for b in c.blocks() {
        counts[match b.codec {
            BlockCodec::Model => 0,
            BlockCodec::Deflate => 1,
            BlockCodec::Stored => 2,
        }] += 1;
    }
    println!("container   {}", input.display());
    println!("model       {}", hex(&h.model_id));
    println!("window      {} bytes", h.window);
    println!("block size  {} bytes", h.block_size);
    println!("preamble    {} bytes", h.preamble.len());
    println!(
        "blocks      {} (model {}, deflate {}, stored {})",
        c.blocks().len(),
        counts[0],
        counts[1],
        counts[2]
    );
    println!("raw         {} bytes", c.total_len());
    println!(
        "stored      {} bytes ({:.3} bits/byte)",
        c.stored_len(),
        bpb(c.stored_len(), c.total_len() as usize)
    );
    if list {
        println!("{:>6} {:>8} {:>10} {:>9} {:>8}  crc32c", "block", "codec", "raw", "payload", "bits/B");
        for (i, b) in c.blocks().iter().enumerate() {
            println!(
                "{i:>6} {:>8} {:>10} {:>9} {:>8.3}  {:08x}",
                b.codec.name(),
                b.raw_len,
                b.len,
                bpb(b.len as usize, b.raw_len as usize),
                b.raw_crc
            );
        }
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct BenchReport {
    file: String,
    model: String,
    model_bytes: usize,
    model_params: usize,
    threads: usize,
    block_size: usize,
    prime_bytes: usize,
    raw_bytes: usize,
    stored_bytes: usize,
    bits_per_byte: f64,
    ratio: f64,
    encode_kb_s: f64,
    decode_kb_s: f64,
    model_blocks: usize,
    fallback_blocks: usize,
    verified: bool,
}

fn bench(
    model: &bitgolf_core::Model,
    name: &str,
    data: &[u8],
    block_size: usize,
    preamble: &[u8],
    threads: usize,
) -> Result<BenchReport> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let opts = CompressOptions { block_size, preamble: preamble.to_vec() };
    let started = Instant::now();
    let packed = pool.install(|| codec::compress(model, data, &opts))?;
    let enc = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let restored = pool.install(|| codec::decompress(model, &packed))?;
    let dec = started.elapsed().as_secs_f64();
    if restored != data {
        bail!("round trip mismatch on {name}");
    }
    let c = Container::parse(&packed)?;
    let model_blocks = c.blocks().iter().filter(|b| b.codec == BlockCodec::Model).count();
    Ok(BenchReport {
        file: name.to_string(),
        model: model.config().name.clone(),
        model_bytes: model.file_len(),
        model_params: model.num_params(),
        threads,
        block_size,
        prime_bytes: preamble.len(),
        raw_bytes: data.len(),
        stored_bytes: packed.len(),
        bits_per_byte: bpb(packed.len(), data.len()),
        ratio: data.len() as f64 / packed.len().max(1) as f64,
        encode_kb_s: data.len() as f64 / 1e3 / enc,
        decode_kb_s: data.len() as f64 / 1e3 / dec,
        model_blocks,
        fallback_blocks: c.blocks().len() - model_blocks,
        verified: true,
    })
}

fn open_store(dir: &Path, models_dir: &Path, spec: Option<&str>, opts: &StoreOptions) -> Result<Store> {
    // An existing store names its model in the manifest; a new one uses --model or the default.
    let manifest = dir.join("MANIFEST");
    let model = if spec.is_none() && manifest.exists() {
        let bytes = std::fs::read(&manifest)?;
        let id: [u8; 32] = bytes.get(10..42).context("manifest is truncated")?.try_into()?;
        models::resolve(None, models_dir, &id)?
    } else {
        models::load(spec.unwrap_or(models::DEFAULT_MODEL), models_dir)?
    };
    Ok(Store::open(dir, model, opts)?)
}

fn store(cmd: StoreCommand, models_dir: &Path) -> Result<()> {
    let defaults = StoreOptions::default();
    match cmd {
        StoreCommand::Put { dir, key, value, file, model } => {
            let value = match (value, file) {
                (Some(v), _) => v.into_bytes(),
                (None, Some(f)) => read_input(&f)?,
                (None, None) => read_input(Path::new("-"))?,
            };
            let mut s = open_store(&dir, models_dir, model.model.as_deref(), &defaults)?;
            s.put(key.as_bytes(), &value)?;
            s.flush()?;
        }
        StoreCommand::Import { dir, file, prefix, block_size, model } => {
            let opts = StoreOptions { block_size, preamble: Vec::new() };
            let mut s = open_store(&dir, models_dir, model.model.as_deref(), &opts)?;
            let data = read_input(&file)?;
            let started = Instant::now();
            let mut n = 0usize;
            for (i, line) in data.split(|&b| b == b'\n').enumerate().filter(|(_, l)| !l.is_empty()) {
                let mut value = line.to_vec();
                value.push(b'\n');
                s.put(format!("{prefix}{i:08}").as_bytes(), &value)?;
                n += 1;
            }
            s.flush()?;
            eprintln!("imported {n} values in {:.1}s", started.elapsed().as_secs_f64());
            println!("{}", serde_json::to_string_pretty(&s.stats())?);
        }
        StoreCommand::Get { dir, key, model } => {
            let s = open_store(&dir, models_dir, model.model.as_deref(), &defaults)?;
            match s.get(key.as_bytes())? {
                Some(v) => std::io::stdout().write_all(&v)?,
                None => bail!("key {key:?} not found"),
            }
        }
        StoreCommand::Scan { dir, prefix, keys_only, model } => {
            let s = open_store(&dir, models_dir, model.model.as_deref(), &defaults)?;
            let mut out = std::io::stdout().lock();
            for (k, v) in s.scan(prefix.as_bytes())? {
                out.write_all(&k)?;
                if !keys_only {
                    out.write_all(b"\t")?;
                    out.write_all(&v)?;
                    if !v.ends_with(b"\n") {
                        out.write_all(b"\n")?;
                    }
                } else {
                    out.write_all(b"\n")?;
                }
            }
        }
        StoreCommand::Stats { dir, model } => {
            let s = open_store(&dir, models_dir, model.model.as_deref(), &defaults)?;
            println!("{}", serde_json::to_string_pretty(&s.stats())?);
        }
    }
    Ok(())
}
