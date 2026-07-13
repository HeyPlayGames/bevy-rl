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

/// Collision layer pair for one env slot: creature bodies vs world geometry.
///
/// Avian has 32 layers; layer 0 is left unused. Each env uses two layers
/// (`creature` then `world`) via `env_id % 15`, so creatures collide with
/// ground but not with other bodies of the same creature. Spatial separation
/// covers the wrap case when more than 15 envs share a world.
fn env_layer_bits(env_id: EnvId) -> (u32, u32) {
    let slot = env_id.0 % 15;
    let creature_bit = 1u32 << (slot * 2 + 1);
    let world_bit = 1u32 << (slot * 2 + 2);
    (creature_bit, world_bit)
}

/// Layers for articulated creature bodies: membership on the env creature layer,
/// filtering only the env world layer (no self-collision between body parts).
pub fn env_creature_collision_layers(env_id: EnvId) -> CollisionLayers {
    let (creature_bit, world_bit) = env_layer_bits(env_id);
    CollisionLayers::from_bits(creature_bit, world_bit)
}

/// Layers for env world geometry (ground, etc.): membership on the env world
/// layer, filtering the env creature layer.
pub fn env_world_collision_layers(env_id: EnvId) -> CollisionLayers {
    let (creature_bit, world_bit) = env_layer_bits(env_id);
    CollisionLayers::from_bits(world_bit, creature_bit)
}
