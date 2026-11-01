//! Locating `.bgm` model files: by explicit path, by preset name, or by the sha256 recorded
//! in a container or store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use bitgolf_core::Model;
use bitgolf_core::error::hex;

/// Model used when neither `--model` nor a recorded hash decides.
pub const DEFAULT_MODEL: &str = "small";

/// Where to look for models: `--models-dir`, then `$BITGOLF_MODELS`, then `./models`.
pub fn models_dir(flag: Option<&Path>) -> PathBuf {
    flag.map(Path::to_path_buf)
        .or_else(|| std::env::var_os("BITGOLF_MODELS").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Loads a model from a path, or from `<dir>/bitgolf-<name>.bgm` when given a preset name.
pub fn load(spec: &str, dir: &Path) -> Result<Arc<Model>> {
    let path = if spec.ends_with(".bgm") || spec.contains(['/', '\\']) {
        PathBuf::from(spec)
    } else {
        dir.join(format!("bitgolf-{spec}.bgm"))
    };
    let bytes = std::fs::read(&path).with_context(|| format!("reading model {}", path.display()))?;
    let model = Model::from_bytes(&bytes).with_context(|| format!("loading {}", path.display()))?;
    Ok(Arc::new(model))
}

/// Loads `spec` if given, otherwise the model in `dir` whose sha256 equals `id`.
pub fn resolve(spec: Option<&str>, dir: &Path, id: &[u8; 32]) -> Result<Arc<Model>> {
    if let Some(spec) = spec {
        return load(spec, dir);
    }
    let want = hex(id);
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("looking for model {want} in {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "bgm") {
            let bytes = std::fs::read(&path)?;
            if let Ok(model) = Model::from_bytes(&bytes) {
                if &model.id() == id {
                    return Ok(Arc::new(model));
                }
            }
        }
    }
    bail!("no model with sha256 {want} in {}; pass --model or --models-dir", dir.display())
}
