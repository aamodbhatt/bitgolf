//! Error type shared by every bitgolf-core module.

use std::io;

/// Everything that can go wrong while loading models, coding data or using a store.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The `.bgm` model file is malformed, corrupt or inconsistent.
    #[error("invalid model: {0}")]
    InvalidModel(String),
    /// A `.bg` container or store file failed validation (bad magic, checksum or bounds).
    #[error("corrupt data: {0}")]
    Corrupt(String),
    /// The data was encoded with a different model than the one supplied.
    #[error("model mismatch: data needs model {expected}, but model {actual} was supplied")]
    ModelMismatch {
        /// Hex sha256 of the model recorded in the data.
        expected: String,
        /// Hex sha256 of the supplied model.
        actual: String,
    },
    /// A caller-supplied argument is out of range.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// An underlying I/O operation failed.
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
}

/// Shorthand result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Lowercase hex encoding, used for model ids in messages and metadata.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}
