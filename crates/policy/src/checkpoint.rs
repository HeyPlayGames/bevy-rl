//! Policy weight checkpoints and sidecar metadata under the OS app data directory.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError},
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::actor_critic::{ActorCritic, ActorCriticConfig};

const APP_NAME: &str = "bevy-rl";
const WEIGHTS_EXTENSION: &str = "mpk";
const META_EXTENSION: &str = "json";
const LATEST_STEM: &str = "latest";

/// Sidecar metadata stored next to policy weights.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyCheckpointMeta {
    pub creature_id: String,
    pub observation_dim: usize,
    pub action_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub update_index: usize,
    /// Mean per-step reward for each PPO update in this training run.
    pub mean_rewards: Vec<f32>,
    /// Mean episode length (steps) for each PPO update in this training run.
    pub mean_episode_lengths: Vec<f32>,
}

impl PolicyCheckpointMeta {
    pub fn from_config(
        creature_id: impl Into<String>,
        config: &ActorCriticConfig,
        update_index: usize,
        mean_rewards: Vec<f32>,
        mean_episode_lengths: Vec<f32>,
    ) -> Self {
        Self {
            creature_id: creature_id.into(),
            observation_dim: config.observation_dim,
            action_dim: config.action_dim,
            hidden_dims: config.hidden_dims.clone(),
            update_index,
            mean_rewards,
            mean_episode_lengths,
        }
    }

    pub fn to_config(&self) -> ActorCriticConfig {
        ActorCriticConfig::new(self.observation_dim, self.action_dim)
            .with_hidden_dims(self.hidden_dims.clone())
    }
}

/// Paths written for one checkpoint save (weights stem; recorder adds `.mpk`).
#[derive(Clone, Debug)]
pub struct CheckpointPaths {
    pub directory: PathBuf,
    pub latest_weights_stem: PathBuf,
    pub latest_meta_path: PathBuf,
    pub step_weights_stem: PathBuf,
    pub step_meta_path: PathBuf,
}

#[derive(Debug)]
pub enum CheckpointError {
    DataDirUnavailable,
    Io(io::Error),
    Serde(serde_json::Error),
    Recorder(RecorderError),
    DimMismatch {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    ArchMismatch {
        field: &'static str,
        expected: String,
        found: String,
    },
    CreatureMismatch {
        expected: String,
        found: String,
    },
    MissingFile(PathBuf),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataDirUnavailable => {
                write!(
                    formatter,
                    "platform data directory is unavailable (dirs::data_dir returned None)"
                )
            }
            Self::Io(error) => write!(formatter, "checkpoint io error: {error}"),
            Self::Serde(error) => write!(formatter, "checkpoint metadata error: {error}"),
            Self::Recorder(error) => write!(formatter, "checkpoint weight recorder error: {error}"),
            Self::DimMismatch {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "checkpoint {field} mismatch: expected {expected}, found {found}"
            ),
            Self::ArchMismatch {
                field,
                expected,
                found,
            } => write!(
                formatter,
                "checkpoint {field} mismatch: expected {expected}, found {found}"
            ),
            Self::CreatureMismatch { expected, found } => write!(
                formatter,
                "checkpoint creature_id mismatch: expected {expected}, found {found}"
            ),
            Self::MissingFile(path) => {
                write!(formatter, "checkpoint file missing: {}", path.display())
            }
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serde(error) => Some(error),
            Self::Recorder(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for CheckpointError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serde(error)
    }
}

impl From<RecorderError> for CheckpointError {
    fn from(error: RecorderError) -> Self {
        Self::Recorder(error)
    }
}

/// `%APPDATA%/bevy-rl/checkpoints` on Windows; OS equivalent elsewhere.
pub fn checkpoint_root() -> Result<PathBuf, CheckpointError> {
    let data_directory = dirs::data_dir().ok_or(CheckpointError::DataDirUnavailable)?;
    Ok(data_directory.join(APP_NAME).join("checkpoints"))
}

pub fn creature_checkpoint_dir(creature_id: &str) -> Result<PathBuf, CheckpointError> {
    Ok(checkpoint_root()?.join(creature_id))
}

pub fn latest_checkpoint_stem(creature_id: &str) -> Result<PathBuf, CheckpointError> {
    Ok(creature_checkpoint_dir(creature_id)?.join(LATEST_STEM))
}

pub fn step_checkpoint_stem(
    creature_id: &str,
    update_index: usize,
) -> Result<PathBuf, CheckpointError> {
    Ok(creature_checkpoint_dir(creature_id)?.join(format!("step_{update_index:06}")))
}

fn weights_file_path(stem: &Path) -> PathBuf {
    stem.with_extension(WEIGHTS_EXTENSION)
}

fn meta_file_path(stem: &Path) -> PathBuf {
    stem.with_extension(META_EXTENSION)
}

fn write_meta(path: &Path, meta: &PolicyCheckpointMeta) -> Result<(), CheckpointError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(meta)?;
    fs::write(path, json)?;
    Ok(())
}

fn read_meta(path: &Path) -> Result<PolicyCheckpointMeta, CheckpointError> {
    if !path.exists() {
        return Err(CheckpointError::MissingFile(path.to_path_buf()));
    }
    let json = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

fn validate_meta(
    meta: &PolicyCheckpointMeta,
    expected_creature_id: &str,
    expected_config: &ActorCriticConfig,
) -> Result<(), CheckpointError> {
    if meta.creature_id != expected_creature_id {
        return Err(CheckpointError::CreatureMismatch {
            expected: expected_creature_id.to_string(),
            found: meta.creature_id.clone(),
        });
    }
    if meta.observation_dim != expected_config.observation_dim {
        return Err(CheckpointError::DimMismatch {
            field: "observation_dim",
            expected: expected_config.observation_dim,
            found: meta.observation_dim,
        });
    }
    if meta.action_dim != expected_config.action_dim {
        return Err(CheckpointError::DimMismatch {
            field: "action_dim",
            expected: expected_config.action_dim,
            found: meta.action_dim,
        });
    }
    if meta.hidden_dims != expected_config.hidden_dims {
        return Err(CheckpointError::ArchMismatch {
            field: "hidden_dims",
            expected: format!("{:?}", expected_config.hidden_dims),
            found: format!("{:?}", meta.hidden_dims),
        });
    }
    Ok(())
}

fn save_weights_and_meta<B: Backend>(
    model: ActorCritic<B>,
    stem: &Path,
    meta: &PolicyCheckpointMeta,
) -> Result<(), CheckpointError> {
    if let Some(parent) = stem.parent() {
        fs::create_dir_all(parent)?;
    }
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.save_file(stem, &recorder)?;
    write_meta(&meta_file_path(stem), meta)?;
    Ok(())
}

/// Save inference-ready weights + JSON metadata as `latest` and a step-numbered pair.
///
/// Prefer passing a non-autodiff model (e.g. `trained_model.valid()`) so viewers can load on Wgpu.
pub fn save_creature_checkpoint<B: Backend>(
    model: &ActorCritic<B>,
    meta: &PolicyCheckpointMeta,
) -> Result<CheckpointPaths, CheckpointError> {
    let directory = creature_checkpoint_dir(&meta.creature_id)?;
    fs::create_dir_all(&directory)?;

    let latest_weights_stem = directory.join(LATEST_STEM);
    let step_weights_stem = directory.join(format!("step_{:06}", meta.update_index));

    save_weights_and_meta(model.clone(), &latest_weights_stem, meta)?;
    save_weights_and_meta(model.clone(), &step_weights_stem, meta)?;

    Ok(CheckpointPaths {
        directory,
        latest_meta_path: meta_file_path(&latest_weights_stem),
        latest_weights_stem,
        step_meta_path: meta_file_path(&step_weights_stem),
        step_weights_stem,
    })
}

/// Resolve a user path to a weights stem (no extension).
///
/// Accepts a directory (uses `latest`), a stem, or a `.mpk` / `.json` file path.
pub fn resolve_checkpoint_stem(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.join(LATEST_STEM);
    }
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(WEIGHTS_EXTENSION) | Some(META_EXTENSION) => path.with_extension(""),
        _ => path.to_path_buf(),
    }
}

/// Load policy weights + metadata from a stem (or directory / file path).
///
/// Fails loudly if creature id or obs/action/hidden dims do not match `expected_config`.
pub fn load_policy_checkpoint<B: Backend>(
    device: &B::Device,
    checkpoint_path: &Path,
    expected_creature_id: &str,
    expected_config: &ActorCriticConfig,
) -> Result<(ActorCritic<B>, PolicyCheckpointMeta), CheckpointError> {
    let stem = resolve_checkpoint_stem(checkpoint_path);
    let meta_path = meta_file_path(&stem);
    let weights_path = weights_file_path(&stem);

    let meta = read_meta(&meta_path)?;
    validate_meta(&meta, expected_creature_id, expected_config)?;

    if !weights_path.exists() {
        return Err(CheckpointError::MissingFile(weights_path));
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = expected_config
        .init::<B>(device)
        .load_file(stem, &recorder, device)?;

    Ok((model, meta))
}

/// Load the `latest` checkpoint for a creature under the platform data directory.
pub fn load_latest_creature_checkpoint<B: Backend>(
    device: &B::Device,
    creature_id: &str,
    expected_config: &ActorCriticConfig,
) -> Result<(ActorCritic<B>, PolicyCheckpointMeta), CheckpointError> {
    let stem = latest_checkpoint_stem(creature_id)?;
    load_policy_checkpoint(device, &stem, creature_id, expected_config)
}
