use bevy::prelude::*;
use sim_core::prelude::*;

use crate::{
    apply_dog_spawn_noise, attach_dog_actuation, dog_quadruped_desc, mark_dog_root, DogSpawnNoise,
};

/// Flat ground + dog quadruped batch spawning.
pub struct DogGroundPlugin;

impl Plugin for DogGroundPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SubstepCount(6))
            .init_resource::<DogSpawnNoise>()
            .add_systems(Startup, spawn_requested_batch)
            .add_systems(Update, handle_respawn_all_envs);
    }
}

fn spawn_requested_batch(
    mut commands: Commands,
    batch: Option<Res<SpawnEnvBatch>>,
    isolation: Res<EnvIsolationConfig>,
    spawn_noise: Res<DogSpawnNoise>,
) {
    let Some(batch) = batch else {
        return;
    };
    for index in 0..batch.count {
        spawn_dog_ground_env(
            &mut commands,
            EnvId::new(index),
            &isolation,
            batch.interpolate,
            &spawn_noise,
        );
    }
}

fn handle_respawn_all_envs(
    mut commands: Commands,
    mut messages: MessageReader<RespawnAllEnvs>,
    isolation: Res<EnvIsolationConfig>,
    spawn_noise: Res<DogSpawnNoise>,
    roots: Query<(Entity, &EnvRoot)>,
    bodies: Query<(Entity, &SimBody)>,
    joints: Query<(Entity, &SimJoint)>,
) {
    let Some(request) = messages.read().last().copied() else {
        return;
    };

    let mut env_indices = std::collections::BTreeSet::new();
    for (_, root) in &roots {
        env_indices.insert(root.env_id.index());
    }
    for (_, body) in &bodies {
        env_indices.insert(body.env_id.index());
    }
    for (_, joint) in &joints {
        env_indices.insert(joint.env_id.index());
    }

    for index in env_indices {
        despawn_env(&mut commands, EnvId::new(index), &roots, &bodies, &joints);
    }

    for index in 0..request.count {
        spawn_dog_ground_env(
            &mut commands,
            EnvId::new(index),
            &isolation,
            request.interpolate,
            &spawn_noise,
        );
    }
}

/// Spawns one isolated flat-ground + dog environment with randomized start pose.
pub fn spawn_dog_ground_env(
    commands: &mut Commands,
    env_id: EnvId,
    isolation: &EnvIsolationConfig,
    interpolate: bool,
    spawn_noise: &DogSpawnNoise,
) {
    let origin = env_origin(env_id, isolation);

    let _root = spawn_env_root(commands, env_id, isolation);
    spawn_flat_ground(commands, env_id, isolation);

    let mut dog = dog_quadruped_desc();
    apply_dog_spawn_noise(&mut dog, spawn_noise);
    let instance = spawn_creature(commands, env_id, origin, &dog, interpolate);
    attach_dog_actuation(commands, &instance);
    mark_dog_root(commands, env_id, &instance);
}
