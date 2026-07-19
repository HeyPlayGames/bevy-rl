//! Dog RL wiring + morphology asset path.
//!
//! Body graph comes from `config/morphology.ron` only — no hardcoded creature.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use sim_core::{load_ron_config, CreatureDesc, RonConfigError};

/// Observation: projected gravity (3) + root lin/ang vel (6) + joint angles (12) + joint ang vels (12)
/// + torso height (1) + foot contacts (4) + previous actions (12).
pub const DOG_OBS_DIM: usize = 50;
/// One normalized target-angle command per revolute leg joint.
pub const DOG_ACTION_DIM: usize = 12;

/// Standing morphology loaded from RON (spawn / reset source of truth).
#[derive(Resource, Clone, Debug)]
pub struct DogMorphology(pub CreatureDesc);

/// Stable action order for dog revolute joints.
pub fn actuated_joint_names() -> [&'static str; DOG_ACTION_DIM] {
    [
        "fl_hip_abduct",
        "fl_hip_flex",
        "fl_knee",
        "fr_hip_abduct",
        "fr_hip_flex",
        "fr_knee",
        "bl_hip_abduct",
        "bl_hip_flex",
        "bl_knee",
        "br_hip_abduct",
        "br_hip_flex",
        "br_knee",
    ]
}

/// Path to the dog morphology RON asset (`examples/dog/config/morphology.ron`).
pub fn dog_morphology_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/morphology.ron")
}

/// Load [`CreatureDesc`] from [`dog_morphology_path`].
pub fn load_dog_morphology() -> Result<CreatureDesc, RonConfigError> {
    load_dog_morphology_from(&dog_morphology_path())
}

/// Load [`CreatureDesc`] from an explicit RON path.
pub fn load_dog_morphology_from(path: &Path) -> Result<CreatureDesc, RonConfigError> {
    load_ron_config(path)
}
