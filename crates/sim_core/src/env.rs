use avian3d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Identifies one isolated environment instance inside a shared Avian world.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnvId(pub u32);

impl EnvId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Root entity for an environment instance (ground, creatures, etc. hang under or reference it).
#[derive(Component, Clone, Copy, Debug)]
pub struct EnvRoot {
    pub env_id: EnvId,
}

/// Marks a rigid body belonging to an environment.
#[derive(Component, Clone, Copy, Debug)]
pub struct SimBody {
    pub env_id: EnvId,
}

/// Marks a joint entity belonging to an environment.
#[derive(Component, Clone, Copy, Debug)]
pub struct SimJoint {
    pub env_id: EnvId,
}

/// Layout + isolation settings for packing many envs into one world.
#[derive(Resource, Clone, Debug)]
pub struct EnvIsolationConfig {
    /// World-space spacing between env origins on the XZ grid.
    pub spacing: f32,
    /// Number of columns in the env placement grid.
    pub grid_columns: u32,
}

impl Default for EnvIsolationConfig {
    fn default() -> Self {
        Self {
            spacing: 40.0,
            grid_columns: 16,
        }
    }
}

/// World origin for an environment (floor center).
pub fn env_origin(env_id: EnvId, config: &EnvIsolationConfig) -> Vec3 {
    let columns = config.grid_columns.max(1);
    let column = env_id.0 % columns;
    let row = env_id.0 / columns;
    Vec3::new(
        column as f32 * config.spacing,
        0.0,
        row as f32 * config.spacing,
    )
}

/// Collision layers so bodies only interact within the same env slot.
///
/// Avian exposes 32 layers. Layer 0 is left unused (default). Envs map to
/// layers 1..=31 via `env_id % 31`. Spatial separation covers the wrap case.
pub fn env_collision_layers(env_id: EnvId) -> CollisionLayers {
    let slot = (env_id.0 % 31) + 1;
    let bit = 1u32 << slot;
    CollisionLayers::from_bits(bit, bit)
}
