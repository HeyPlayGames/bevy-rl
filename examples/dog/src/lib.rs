//! Example creature pack: dog quadruped on flat ground.

mod contacts;
mod env;
mod morphology;
mod reward;
mod spawn_noise;

pub use env::{spawn_dog_ground_env, DogGroundPlugin};
pub use morphology::{
    actuated_joint_names, dog_morphology_path, load_dog_morphology, load_dog_morphology_from,
    DogMorphology, DOG_ACTION_DIM, DOG_OBS_DIM,
};
pub use reward::{dog_balance_reward, dog_has_fallen, DogBalanceConfig};
pub use spawn_noise::{sample_dog_spawn_poses, DogSpawnNoise};

use avian3d::prelude::*;
use bevy::prelude::*;
use sim_core::prelude::*;

use contacts::{update_dog_contacts, DogContactBuffer, DogGroundContacts};

/// Creature id used for checkpoints and policy metadata.
pub const CREATURE_ID: &str = "dog";

/// Full dog setup: ground env batch + obs/reward/episode systems.
///
/// Action → joint target mapping comes from [`sim_core::SimCorePlugin`] via [`RlBuffers`].
pub struct DogPlugin;

impl Plugin for DogPlugin {
    fn build(&self, app: &mut App) {
        let morphology = match load_dog_morphology() {
            Ok(creature) => DogMorphology(creature),
            Err(error) => {
                panic!(
                    "failed to load dog morphology from {}: {error}",
                    dog_morphology_path().display()
                );
            }
        };

        let reward_config = match DogBalanceConfig::load_default() {
            Ok(config) => config,
            Err(error) => {
                bevy::log::error!("failed to load dog reward config: {error}");
                DogBalanceConfig::default()
            }
        };

        app.insert_resource(morphology)
            .insert_resource(CreatureSpec {
                id: CREATURE_ID,
                observation_dim: DOG_OBS_DIM,
                action_dim: DOG_ACTION_DIM,
            })
            .insert_resource(reward_config)
            .init_resource::<DogContactBuffer>()
            .add_plugins(DogGroundPlugin)
            .add_systems(
                FixedLast,
                (
                    update_dog_contacts,
                    write_dog_observations,
                    write_dog_rewards,
                    advance_and_reset_episodes,
                    // Soft resets rewrite poses in-place; refresh obs so the next
                    // policy step sees the post-reset standing state.
                    refresh_dog_observations_after_reset,
                )
                    .chain(),
            );
    }
}

pub fn attach_dog_actuation(
    commands: &mut Commands,
    instance: &CreatureInstance,
    morphology: &CreatureDesc,
) {
    for (action_index, joint_name) in actuated_joint_names().iter().enumerate() {
        let Some(&joint_entity) = instance.joints.get(*joint_name) else {
            continue;
        };
        let Some(joint) = morphology
            .joints
            .iter()
            .find(|joint| joint.name == *joint_name)
        else {
            continue;
        };
        let Some(default_angle) = joint.kind.default_angle() else {
            continue;
        };
        commands.entity(joint_entity).insert((
            ActuatedRevolute {
                action_index,
                rest_angle: default_angle,
            },
            JointTargetAngle(0.0),
        ));
    }
}

pub fn mark_dog_root(commands: &mut Commands, env_id: EnvId, instance: &CreatureInstance) {
    if let Some(&torso) = instance.bodies.get("torso") {
        commands.entity(torso).insert(CreatureRoot { env_id });
    }
}

fn write_dog_observations(
    mut buffers: ResMut<RlBuffers>,
    contacts: Res<DogContactBuffer>,
    roots: Query<(Entity, &CreatureRoot)>,
    bodies: Query<(&Transform, &LinearVelocity, &AngularVelocity)>,
    joints: Query<(&RevoluteJoint, &ActuatedRevolute, &SimJoint)>,
    transforms: Query<&Transform>,
    angular_velocities: Query<&AngularVelocity>,
) {
    let env_count = buffers.observations.len();
    if env_count == 0 {
        return;
    }

    for (torso_entity, root) in &roots {
        let env_index = root.env_id.index() as usize;
        if env_index >= env_count {
            continue;
        }

        let Ok((transform, linear_velocity, angular_velocity)) = bodies.get(torso_entity) else {
            continue;
        };

        let rotation = transform.rotation;
        let inverse_rotation = rotation.inverse();
        let projected_gravity = inverse_rotation * Vec3::NEG_Y;
        let local_linear_velocity = inverse_rotation * linear_velocity.0;
        let local_angular_velocity = inverse_rotation * angular_velocity.0;

        let previous_actions = buffers.actions.get(env_index).cloned().unwrap_or_default();
        let ground_contacts = contacts
            .envs
            .get(env_index)
            .copied()
            .unwrap_or_default();

        let observation = &mut buffers.observations[env_index];
        observation.fill(0.0);

        observation[0] = projected_gravity.x;
        observation[1] = projected_gravity.y;
        observation[2] = projected_gravity.z;
        observation[3] = local_linear_velocity.x;
        observation[4] = local_linear_velocity.y;
        observation[5] = local_linear_velocity.z;
        observation[6] = local_angular_velocity.x;
        observation[7] = local_angular_velocity.y;
        observation[8] = local_angular_velocity.z;

        for (joint, actuated, sim_joint) in &joints {
            if sim_joint.env_id != root.env_id {
                continue;
            }
            let angle_index = 9 + actuated.action_index;
            let velocity_index = 9 + DOG_ACTION_DIM + actuated.action_index;
            if velocity_index >= observation.len() {
                continue;
            }

            let Ok(transform_a) = transforms.get(joint.body1) else {
                continue;
            };
            let Ok(transform_b) = transforms.get(joint.body2) else {
                continue;
            };
            let Ok(angular_velocity_a) = angular_velocities.get(joint.body1) else {
                continue;
            };
            let Ok(angular_velocity_b) = angular_velocities.get(joint.body2) else {
                continue;
            };

            observation[angle_index] =
                revolute_angle(joint, transform_a.rotation, transform_b.rotation);
            observation[velocity_index] = revolute_angular_velocity(
                joint,
                transform_a.rotation,
                angular_velocity_a.0,
                angular_velocity_b.0,
            );
        }

        let height_index = 9 + 2 * DOG_ACTION_DIM;
        if height_index < observation.len() {
            observation[height_index] = transform.translation.y;
        }

        let contact_offset = height_index + 1;
        ground_contacts.write_observation(observation, contact_offset);

        let previous_action_offset = contact_offset + 4;
        if previous_action_offset + DOG_ACTION_DIM <= observation.len() {
            let copy_length = DOG_ACTION_DIM.min(previous_actions.len());
            observation[previous_action_offset..previous_action_offset + copy_length]
                .copy_from_slice(&previous_actions[..copy_length]);
        }
    }
}

/// Re-write observations after mid-episode fall resets so the next action uses
/// the post-reset standing state instead of the pre-reset fallen state.
fn refresh_dog_observations_after_reset(
    buffers: ResMut<RlBuffers>,
    contacts: Res<DogContactBuffer>,
    roots: Query<(Entity, &CreatureRoot)>,
    bodies: Query<(&Transform, &LinearVelocity, &AngularVelocity)>,
    joints: Query<(&RevoluteJoint, &ActuatedRevolute, &SimJoint)>,
    transforms: Query<&Transform>,
    angular_velocities: Query<&AngularVelocity>,
) {
    write_dog_observations(
        buffers,
        contacts,
        roots,
        bodies,
        joints,
        transforms,
        angular_velocities,
    );
}

fn write_dog_rewards(
    mut buffers: ResMut<RlBuffers>,
    contacts: Res<DogContactBuffer>,
    config: Res<DogBalanceConfig>,
    roots: Query<(Entity, &CreatureRoot)>,
    bodies: Query<&Transform>,
) {
    let env_count = buffers.rewards.len();
    if env_count == 0 {
        return;
    }

    for (torso_entity, root) in &roots {
        let env_index = root.env_id.index() as usize;
        if env_index >= env_count {
            continue;
        }

        let Ok(transform) = bodies.get(torso_entity) else {
            continue;
        };

        let ground_contacts = contacts
            .envs
            .get(env_index)
            .copied()
            .unwrap_or_default();

        buffers.rewards[env_index] = dog_balance_reward(
            &config,
            transform.rotation * Vec3::Y,
            transform.translation.y,
            ground_contacts,
        );
    }
}

fn soft_reset_dog_env(
    commands: &mut Commands,
    env_id: EnvId,
    isolation: &EnvIsolationConfig,
    spawn_noise: &DogSpawnNoise,
    morphology: &CreatureDesc,
    bodies: &mut Query<(
        Entity,
        &SimBody,
        &CreaturePart,
        &mut Transform,
        &mut Position,
        &mut Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
    joint_targets: &mut Query<(&SimJoint, &mut JointTargetAngle)>,
) {
    let poses = sample_dog_spawn_poses(morphology, spawn_noise);
    let world_origin = env_origin(env_id, isolation);
    soft_reset_creature(commands, env_id, world_origin, &poses, bodies, joint_targets);
}

/// Soft-reset every dog env with spawn-pose noise.
///
/// Used by the trainer at the start of each PPO update so rollouts begin
/// from a fresh randomized standing pose rather than continuing fallen states.
pub fn reset_all_envs(
    mut commands: Commands,
    mut buffers: ResMut<RlBuffers>,
    mut contacts: ResMut<DogContactBuffer>,
    isolation: Res<EnvIsolationConfig>,
    spawn_noise: Res<DogSpawnNoise>,
    morphology: Res<DogMorphology>,
    mut bodies: Query<(
        Entity,
        &SimBody,
        &CreaturePart,
        &mut Transform,
        &mut Position,
        &mut Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
    mut joint_targets: Query<(&SimJoint, &mut JointTargetAngle)>,
) {
    let env_count = buffers.episode_steps.len();
    contacts.envs.resize(env_count, DogGroundContacts::default());
    for env_index in 0..env_count {
        let env_id = EnvId::new(env_index as u32);
        soft_reset_dog_env(
            &mut commands,
            env_id,
            &isolation,
            &spawn_noise,
            &morphology.0,
            &mut bodies,
            &mut joint_targets,
        );
        buffers.episode_steps[env_index] = 0;
        buffers.episode_terminated[env_index] = false;
        buffers.episode_truncated[env_index] = false;
        buffers.episode_done[env_index] = false;
        if let Some(actions) = buffers.actions.get_mut(env_index) {
            actions.fill(0.0);
        }
        if let Some(env_contacts) = contacts.envs.get_mut(env_index) {
            *env_contacts = DogGroundContacts::default();
        }
    }
}

fn advance_and_reset_episodes(
    mut commands: Commands,
    mut buffers: ResMut<RlBuffers>,
    mut contacts: ResMut<DogContactBuffer>,
    reset_policy: Res<EpisodeResetPolicy>,
    config: Res<DogBalanceConfig>,
    isolation: Res<EnvIsolationConfig>,
    spawn_noise: Res<DogSpawnNoise>,
    morphology: Res<DogMorphology>,
    mut pose_queries: ParamSet<(
        Query<(&CreatureRoot, &Transform)>,
        Query<(
            Entity,
            &SimBody,
            &CreaturePart,
            &mut Transform,
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        )>,
    )>,
    mut joint_targets: Query<(&SimJoint, &mut JointTargetAngle)>,
) {
    let env_count = buffers.episode_steps.len();
    if env_count == 0 {
        return;
    }

    let mut fallen_by_env = vec![false; env_count];
    for (root, transform) in pose_queries.p0().iter() {
        let env_index = root.env_id.index() as usize;
        if env_index >= env_count {
            continue;
        }
        let up = transform.rotation * Vec3::Y;
        let height = transform.translation.y;
        if dog_has_fallen(&config, up, height) {
            fallen_by_env[env_index] = true;
        }
    }

    let horizon = buffers.episode_horizon;
    for (env_index, &fallen) in fallen_by_env.iter().enumerate() {
        buffers.episode_terminated[env_index] = false;
        buffers.episode_truncated[env_index] = false;
        buffers.episode_done[env_index] = false;
        buffers.episode_steps[env_index] = buffers.episode_steps[env_index].saturating_add(1);

        if fallen {
            buffers.rewards[env_index] += config.fall_penalty;
        }

        let timed_out = buffers.episode_steps[env_index] >= horizon;
        if fallen {
            buffers.episode_terminated[env_index] = true;
            buffers.episode_done[env_index] = true;
        } else if timed_out {
            buffers.episode_truncated[env_index] = true;
            buffers.episode_done[env_index] = true;
        } else {
            continue;
        }

        buffers.episode_steps[env_index] = 0;
        if let Some(actions) = buffers.actions.get_mut(env_index) {
            actions.fill(0.0);
        }

        // True terminals always reset. Truncations only reset when enabled so
        // trainers can bootstrap from the final pre-reset state.
        let should_reset = fallen || reset_policy.reset_on_truncate;
        if !should_reset {
            continue;
        }

        // Contact graph is stale until the next physics step; clear so the
        // post-reset observation does not keep fallen-body contacts.
        if let Some(env_contacts) = contacts.envs.get_mut(env_index) {
            *env_contacts = DogGroundContacts::default();
        }

        let env_id = EnvId::new(env_index as u32);
        soft_reset_dog_env(
            &mut commands,
            env_id,
            &isolation,
            &spawn_noise,
            &morphology.0,
            &mut pose_queries.p1(),
            &mut joint_targets,
        );
    }
}
