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
mod ron_config;
mod spawn;

pub use config::{load_json_config, load_json_config_or_default, JsonConfigError};
pub use ron_config::{load_ron_config, save_ron_config, RonConfigError};
pub use control::reward;
pub use control::{
    apply_joint_targets, revolute_angle, revolute_angular_velocity, ActuatedRevolute, CreatureRoot,
    JointTargetAngle,
};
pub use creature::{
    articulation_root_name, attach_default_revolute_actuation, compute_body_poses,
    compute_default_body_poses, compute_zero_body_poses, default_revolute_angles,
    set_body_pose_at_zero, transform_body_poses, BodyDesc, BodyPoseMap, BodyShape, CreatureDesc,
    CreatureInstance, JointDesc, JointKind,
};
pub use env::{
    env_creature_collision_layers, env_origin, env_world_collision_layers, EnvId,
    EnvIsolationConfig, EnvRoot, SimBody, SimJoint,
};
pub use ground::{ground_half_extents, spawn_flat_ground, FlatGround, GROUND_HALF_THICKNESS};
pub use plugin::{configure_headless_app, HeadlessSimConfig, SimCorePlugin, SimTick};
pub use rl::{
    apply_buffered_actions, configure_control_systems, ControlSystems, CreatureSpec,
    EpisodeResetPolicy, PolicyControl, PolicyDecimation, RespawnAllEnvs, RlBuffers, SpawnEnvBatch,
};
pub use spawn::{despawn_env, soft_reset_creature, spawn_creature, spawn_env_root, CreaturePart};

pub mod prelude {
    pub use crate::{
        apply_buffered_actions, apply_joint_targets, articulation_root_name,
        attach_default_revolute_actuation, compute_body_poses, compute_default_body_poses,
        compute_zero_body_poses, configure_control_systems, configure_headless_app,
        default_revolute_angles, despawn_env, env_creature_collision_layers, env_origin,
        env_world_collision_layers, ground_half_extents, load_json_config,
        load_json_config_or_default, load_ron_config, revolute_angle, revolute_angular_velocity,
        save_ron_config, set_body_pose_at_zero, soft_reset_creature, spawn_creature,
        spawn_env_root, spawn_flat_ground, transform_body_poses, ActuatedRevolute, BodyDesc,
        BodyPoseMap, BodyShape, ControlSystems, CreatureDesc, CreatureInstance, CreaturePart,
        CreatureRoot, CreatureSpec, EnvId, EnvIsolationConfig, EnvRoot, EpisodeResetPolicy,
        FlatGround, HeadlessSimConfig, JointDesc, JointKind, JointTargetAngle, JsonConfigError,
        PolicyControl, PolicyDecimation, RespawnAllEnvs, RlBuffers, RonConfigError, SimBody,
        SimCorePlugin, SimJoint, SimTick, SpawnEnvBatch, GROUND_HALF_THICKNESS,
    };
    pub use avian3d::prelude::*;
    pub use bevy::prelude::{
        App, Commands, Component, Entity, Plugin, Query, Res, ResMut, Resource, Transform, Vec3,
        With, Without,
    };
}
