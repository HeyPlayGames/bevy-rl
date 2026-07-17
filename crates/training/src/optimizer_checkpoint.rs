//! Save/load Adam optimizer records next to policy weight checkpoints.

use std::path::{Path, PathBuf};

use bevy::prelude::{info, warn};
use burn::{
    module::AutodiffModule,
    optim::Optimizer,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use policy::{resolve_checkpoint_stem, ActorCritic};

use crate::ppo::PpoOptimizer;

const OPTIM_SUFFIX: &str = "_optim";

/// `latest` → `latest_optim`, `step_000049` → `step_000049_optim`.
pub fn optimizer_stem_from_weights_stem(weights_stem: &Path) -> PathBuf {
    let file_name = weights_stem
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    weights_stem.with_file_name(format!("{file_name}{OPTIM_SUFFIX}"))
}

fn optimizer_file_path(optim_stem: &Path) -> PathBuf {
    optim_stem.with_extension("mpk")
}

/// Write optimizer moments beside a weights stem (recorder adds `.mpk`).
pub fn save_optimizer_checkpoint<B>(
    optimizer: &PpoOptimizer<B>,
    weights_stem: &Path,
) -> Result<PathBuf, String>
where
    B: AutodiffBackend,
    ActorCritic<B>: AutodiffModule<B>,
{
    let optim_stem = optimizer_stem_from_weights_stem(weights_stem);
    if let Some(parent) = optim_stem.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(optimizer.to_record(), optim_stem.clone())
        .map_err(|error| error.to_string())?;

    Ok(optimizer_file_path(&optim_stem))
}

/// Load optimizer moments for a policy checkpoint path, or keep `optimizer` if missing/unreadable.
pub fn load_optimizer_checkpoint_or_fresh<B>(
    optimizer: PpoOptimizer<B>,
    device: &B::Device,
    checkpoint_path: &Path,
) -> PpoOptimizer<B>
where
    B: AutodiffBackend,
    ActorCritic<B>: AutodiffModule<B>,
{
    let weights_stem = resolve_checkpoint_stem(checkpoint_path);
    let optim_stem = optimizer_stem_from_weights_stem(&weights_stem);
    let optim_file = optimizer_file_path(&optim_stem);

    if !optim_file.exists() {
        warn!(
            "optimizer checkpoint missing at {}, continuing with empty Adam",
            optim_file.display()
        );
        return optimizer;
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load(optim_stem.clone(), device) {
        Ok(record) => {
            info!("loaded optimizer checkpoint from {}", optim_file.display());
            optimizer.load_record(record)
        }
        Err(error) => {
            warn!(
                "failed to load optimizer checkpoint from {}: {error}; continuing with empty Adam",
                optim_file.display()
            );
            optimizer
        }
    }
}
