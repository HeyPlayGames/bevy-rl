use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use sim_core::{load_json_config_or_default, reward, JsonConfigError};

/// Weights for the dog balance objective.
///
/// Tunable at runtime via `config/reward.json` (no recompile required).
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct DogBalanceConfig {
    pub target_height: f32,
    pub height_tolerance: f32,
    pub upright_weight: f32,
    pub height_weight: f32,
    /// Episode ends when torso height drops below this.
    pub min_height: f32,
    /// Episode ends when uprightness (`up · world_up`) drops below this.
    pub min_upright: f32,
    /// Added to the step reward on the fall-terminal step.
    pub fall_penalty: f32,
}

impl Default for DogBalanceConfig {
    fn default() -> Self {
        Self {
            target_height: 0.73,
            height_tolerance: 0.25,
            upright_weight: 1.0,
            height_weight: 0.5,
            min_height: 0.2,
            min_upright: 0.2,
            fall_penalty: -1.0,
        }
    }
}

impl DogBalanceConfig {
    /// Default path shipped with the dog pack (`config/reward.json`).
    pub fn default_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/reward.json")
    }

    /// Load from JSON, or defaults when the file is missing.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, JsonConfigError> {
        load_json_config_or_default(path)
    }

    /// Load from [`Self::default_path`].
    pub fn load_default() -> Result<Self, JsonConfigError> {
        Self::load_from_path(Self::default_path())
    }
}

/// Fixed-horizon balance reward: stay upright near target height.
pub fn dog_balance_reward(config: &DogBalanceConfig, up: Vec3, height: f32) -> f32 {
    let upright = reward::uprightness(up, Vec3::Y);
    let height_term = reward::height_band(height, config.target_height, config.height_tolerance);

    config.upright_weight * upright + config.height_weight * height_term
}

/// True when torso is too low or tipped past the fall thresholds.
pub fn dog_has_fallen(config: &DogBalanceConfig, up: Vec3, height: f32) -> bool {
    let upright = reward::uprightness(up, Vec3::Y);
    height < config.min_height || upright < config.min_upright
}
