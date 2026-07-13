//! Parallel headless physics simulation core.
//!
//! Owns env isolation (spatial + collision layers), fixed-timestep schedule
//! helpers, and a thin creature articulation description that spawns Avian
//! rigid bodies + joints.

mod creature;
mod env;
mod plugin;
mod spawn;

pub use creature::{
    BodyDesc, BodyShape, CreatureDesc, CreatureInstance, JointDesc, JointKind, PoseDesc,
};
pub use env::{
    env_collision_layers, env_origin, EnvId, EnvIsolationConfig, EnvRoot, SimBody, SimJoint,
};
pub use plugin::{configure_headless_app, HeadlessSimConfig, SimCorePlugin, SimTick};
pub use spawn::{
    debug_mesh_for_shape, despawn_env, reset_env, spawn_creature, spawn_env_root, DebugMeshKind,
};

pub mod prelude {
    pub use crate::{
        despawn_env, env_collision_layers, env_origin, reset_env, spawn_creature, spawn_env_root,
        BodyDesc, BodyShape, CreatureDesc, CreatureInstance, EnvId, EnvIsolationConfig, EnvRoot,
        HeadlessSimConfig, JointDesc, JointKind, PoseDesc, SimBody, SimCorePlugin, SimJoint,
        SimTick, configure_headless_app,
    };
    pub use avian3d::prelude::*;
    pub use bevy::prelude::{
        App, Commands, Component, Entity, Plugin, Query, Res, ResMut, Resource, Transform, Vec3,
        With, Without,
    };
}
