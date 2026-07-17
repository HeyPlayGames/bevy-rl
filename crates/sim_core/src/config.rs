//! Shared JSON config load/save for runtime-tunable parameters.
//!
//! Creature packs and training tools can keep reward / hyperparameter knobs in
//! `.json` files and edit them without recompiling.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Serialize};

/// Error loading or writing a JSON config file.
#[derive(Debug)]
pub enum JsonConfigError {
    Io(io::Error),
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    Write {
        path: PathBuf,
        source: serde_json::Error,
    },
}

impl std::fmt::Display for JsonConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "config io error: {error}"),
            Self::Parse { path, source } => {
                write!(
                    formatter,
                    "failed to parse config {}: {source}",
                    path.display()
                )
            }
            Self::Write { path, source } => {
                write!(
                    formatter,
                    "failed to serialize config {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for JsonConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse { source, .. } | Self::Write { source, .. } => Some(source),
        }
    }
}

impl From<io::Error> for JsonConfigError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Load a typed config from a JSON file.
pub fn load_json_config<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> Result<T, JsonConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|source| JsonConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

/// Load a JSON config, or return `T::default()` when the file is missing.
///
/// Parse errors still fail so typos are not silently ignored.
pub fn load_json_config_or_default<T: DeserializeOwned + Default>(
    path: impl AsRef<Path>,
) -> Result<T, JsonConfigError> {
    let path = path.as_ref();
    match load_json_config(path) {
        Ok(config) => Ok(config),
        Err(JsonConfigError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            Ok(T::default())
        }
        Err(error) => Err(error),
    }
}

/// Write a typed config as pretty JSON, creating parent directories as needed.
pub fn save_json_config<T: Serialize>(
    path: impl AsRef<Path>,
    value: &T,
) -> Result<(), JsonConfigError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(value).map_err(|source| JsonConfigError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, text)?;
    Ok(())
}
