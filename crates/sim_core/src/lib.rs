//! Parallel headless physics simulation core.
//!
//! Owns env isolation (spatial + collision layers), fixed-timestep schedule
//! helpers, a thin creature articulation description, and the shared RL step
//! buffer contract used by trainers and viewers.

mod config;
mod control;
mod creature;
mod env;
mod ground;
mod plugin;
mod rl;
mod spawn;

pub use config::{load_json_config, load_json_config_or_default, JsonConfigError};
pub use control::reward;
pub use control::{
    apply_joint_targets, revolute_angle, revolute_angular_velocity, ActuatedRevolute, CreatureRoot,
    JointTargetAngle,
};
pub use creature::{
    apply_revolute_angles, transform_creature_poses, BodyDesc, BodyShape, CreatureDesc,
    CreatureInstance, JointDesc, JointKind, PoseDesc,
};
pub use env::{
    env_creature_collision_layers, env_origin, env_world_collision_layers, EnvId,
    EnvIsolationConfig, EnvRoot, SimBody, SimJoint,
};
pub use ground::{ground_half_extents, spawn_flat_ground, FlatGround, GROUND_HALF_THICKNESS};
pub use plugin::{configure_headless_app, HeadlessSimConfig, SimCorePlugin, SimTick};
pub use rl::{
    apply_buffered_actions, configure_control_systems, ControlSystems, CreatureSpec,
    EpisodeResetPolicy, RespawnAllEnvs, RlBuffers, SpawnEnvBatch,
};
pub use spawn::{despawn_env, soft_reset_creature, spawn_creature, spawn_env_root, CreaturePart};

pub mod prelude {
    pub use crate::{
        apply_buffered_actions, apply_joint_targets, apply_revolute_angles,
        configure_control_systems, configure_headless_app, despawn_env,
        env_creature_collision_layers, env_origin, env_world_collision_layers, ground_half_extents,
        revolute_angle, revolute_angular_velocity, soft_reset_creature, spawn_creature,
        spawn_env_root, spawn_flat_ground, transform_creature_poses, ActuatedRevolute, BodyDesc,
        BodyShape, ControlSystems, CreatureDesc, CreatureInstance, CreaturePart, CreatureRoot,
        CreatureSpec, EnvId, EnvIsolationConfig, EnvRoot, FlatGround, EpisodeResetPolicy,
        HeadlessSimConfig, JointDesc, JointKind, JointTargetAngle, PoseDesc, JsonConfigError,
        RespawnAllEnvs, RlBuffers, SimBody, SimCorePlugin, SimJoint, SimTick, SpawnEnvBatch,
        GROUND_HALF_THICKNESS, load_json_config, load_json_config_or_default,
    };
    pub use avian3d::prelude::*;
    pub use bevy::prelude::{
        App, Commands, Component, Entity, Plugin, Query, Res, ResMut, Resource, Transform, Vec3,
        With, Without,
    };
}
