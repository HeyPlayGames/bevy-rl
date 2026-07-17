//! Flat ground plane helpers shared by creature packs.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::env::{env_origin, env_world_collision_layers, EnvId, EnvIsolationConfig, SimBody};

/// Half-thickness of the default flat ground cuboid.
pub const GROUND_HALF_THICKNESS: f32 = 0.25;

/// Marks the static ground body for an env (debug meshes, queries).
#[derive(Component, Clone, Copy, Debug)]
pub struct FlatGround {
    pub env_id: EnvId,
}

/// Half-extents for the flat ground cuboid in an env.
pub fn ground_half_extents(isolation: &EnvIsolationConfig) -> Vec3 {
    let half_size = isolation.spacing * 0.9 * 0.5;
    Vec3::new(half_size, GROUND_HALF_THICKNESS, half_size)
}

/// Spawns a static friction ground cuboid centered under the env origin.
pub fn spawn_flat_ground(
    commands: &mut Commands,
    env_id: EnvId,
    isolation: &EnvIsolationConfig,
) -> Entity {
    let origin = env_origin(env_id, isolation);
    let layers = env_world_collision_layers(env_id);
    let half_extents = ground_half_extents(isolation);
    let collider = Collider::cuboid(
        half_extents.x * 2.0,
        half_extents.y * 2.0,
        half_extents.z * 2.0,
    );
    let ground_translation = origin - Vec3::Y * half_extents.y;
    commands
        .spawn((
            Name::new(format!("ground_{}", env_id.index())),
            FlatGround { env_id },
            SimBody { env_id },
            RigidBody::Static,
            collider,
            layers,
            Friction::new(0.9),
            Transform::from_translation(ground_translation),
        ))
        .id()
}
