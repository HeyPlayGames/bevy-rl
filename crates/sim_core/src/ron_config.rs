//! Shared RON config loading/saving for serializable runtime assets.
//!
//! Used for creature morphology (and anything else that wants Rust-friendly
//! text assets with native Bevy math types).

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Serialize};

/// Error loading or saving a RON config file.
#[derive(Debug)]
pub enum RonConfigError {
    Io(io::Error),
    Parse {
        path: PathBuf,
        source: ron::error::SpannedError,
    },
    Serialize {
        path: PathBuf,
        source: ron::Error,
    },
}

impl std::fmt::Display for RonConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "ron config io error: {error}"),
            Self::Parse { path, source } => {
                write!(
                    formatter,
                    "failed to parse ron {}: {source}",
                    path.display()
                )
            }
            Self::Serialize { path, source } => {
                write!(
                    formatter,
                    "failed to serialize ron {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for RonConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse { source, .. } => Some(source),
            Self::Serialize { source, .. } => Some(source),
        }
    }
}

impl From<io::Error> for RonConfigError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Load a typed value from a RON file.
pub fn load_ron_config<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, RonConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    ron::from_str(&text).map_err(|source| RonConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

/// Serialize `value` to pretty RON and write it to `path`.
///
/// Creates parent directories when needed.
pub fn save_ron_config<T: Serialize>(
    path: impl AsRef<Path>,
    value: &T,
) -> Result<(), RonConfigError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let pretty = ron::ser::PrettyConfig::new()
        .depth_limit(8)
        .indentor("  ".to_string())
        .struct_names(true);
    let text = ron::ser::to_string_pretty(value, pretty).map_err(|source| {
        RonConfigError::Serialize {
            path: path.to_path_buf(),
            source,
        }
    })?;
    fs::write(path, text)?;
    Ok(())
}
