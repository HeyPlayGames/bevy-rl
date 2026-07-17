//! Example creature pack: dog quadruped on flat ground.

mod env;
mod morphology;
mod reward;
mod spawn_noise;

pub use env::{spawn_dog_ground_env, DogGroundPlugin};
pub use morphology::{dog_quadruped_desc, DOG_ACTION_DIM, DOG_OBS_DIM};
pub use reward::{dog_balance_reward, dog_has_fallen, DogBalanceConfig};
pub use spawn_noise::{apply_dog_spawn_noise, DogSpawnNoise};

use avian3d::prelude::*;
use bevy::prelude::*;
use sim_core::prelude::*;

use morphology::actuated_joint_names;

/// Creature id used for checkpoints and policy metadata.
pub const CREATURE_ID: &str = "dog";

/// Full dog setup: ground env batch + obs/reward/episode systems.
///
/// Action → joint target mapping comes from [`sim_core::SimCorePlugin`] via [`RlBuffers`].
pub struct DogPlugin;

impl Plugin for DogPlugin {
    fn build(&self, app: &mut App) {
        let reward_config = match DogBalanceConfig::load_default() {
            Ok(config) => config,
            Err(error) => {
                bevy::log::error!("failed to load dog reward config: {error}");
                DogBalanceConfig::default()
            }
        };

        app.insert_resource(CreatureSpec {
            id: CREATURE_ID,
            observation_dim: DOG_OBS_DIM,
            action_dim: DOG_ACTION_DIM,
        })
        .insert_resource(reward_config)
        .init_resource::<DogSpawnNoise>()
        .add_plugins(DogGroundPlugin)
        .add_systems(
            FixedLast,
            (
                write_dog_observations,
                write_dog_rewards,
                advance_and_reset_episodes,
                // Fall resets despawn/respawn via Commands; flush before rewriting
                // observations so the next policy step sees the post-reset spawn.
                ApplyDeferred,
                refresh_dog_observations_after_reset,
            )
                .chain(),
        );
    }
}

/// Tags a dog agent root for env-scoped RL bookkeeping.
#[derive(Component, Clone, Copy, Debug)]
pub struct DogAgent {
    pub env_id: EnvId,
}

pub fn attach_dog_actuation(commands: &mut Commands, instance: &CreatureInstance) {
    for (action_index, joint_name) in actuated_joint_names().iter().enumerate() {
        let Some(&joint_entity) = instance.joints.get(*joint_name) else {
            continue;
        };
        commands
            .entity(joint_entity)
            .insert((
                ActuatedRevolute {
                    action_index,
                    rest_angle: 0.0,
                },
                JointTargetAngle(0.0),
            ));
    }
}

pub fn mark_dog_root(commands: &mut Commands, env_id: EnvId, instance: &CreatureInstance) {
    commands.entity(instance.root).insert(DogAgent { env_id });
    if let Some(&torso) = instance.bodies.get("torso") {
        commands.entity(torso).insert(CreatureRoot { env_id });
    }
}

fn write_dog_observations(
    mut buffers: ResMut<RlBuffers>,
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

        let previous_action_offset = height_index + 1;
        if previous_action_offset + DOG_ACTION_DIM <= observation.len() {
            let copy_length = DOG_ACTION_DIM.min(previous_actions.len());
            observation[previous_action_offset..previous_action_offset + copy_length]
                .copy_from_slice(&previous_actions[..copy_length]);
        }
    }
}

/// Re-write observations after mid-episode fall resets so the next action uses
/// the post-respawn state instead of the pre-reset fallen state.
fn refresh_dog_observations_after_reset(
    buffers: ResMut<RlBuffers>,
    roots: Query<(Entity, &CreatureRoot)>,
    bodies: Query<(&Transform, &LinearVelocity, &AngularVelocity)>,
    joints: Query<(&RevoluteJoint, &ActuatedRevolute, &SimJoint)>,
    transforms: Query<&Transform>,
    angular_velocities: Query<&AngularVelocity>,
) {
    write_dog_observations(buffers, roots, bodies, joints, transforms, angular_velocities);
}

fn write_dog_rewards(
    mut buffers: ResMut<RlBuffers>,
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

        buffers.rewards[env_index] = dog_balance_reward(
            &config,
            transform.rotation * Vec3::Y,
            transform.translation.y,
        );
    }
}

/// Despawn and respawn every dog env with spawn-pose noise.
///
/// Used by the trainer at the start of each PPO update so rollouts begin
/// from a fresh randomized standing pose rather than continuing fallen states.
pub fn reset_all_envs(
    mut commands: Commands,
    mut buffers: ResMut<RlBuffers>,
    isolation: Res<EnvIsolationConfig>,
    spawn_noise: Res<DogSpawnNoise>,
    roots: Query<(Entity, &EnvRoot)>,
    bodies: Query<(Entity, &SimBody)>,
    joints: Query<(Entity, &SimJoint)>,
) {
    let env_count = buffers.episode_steps.len();
    for env_index in 0..env_count {
        let env_id = EnvId::new(env_index as u32);
        reset_env(
            &mut commands,
            env_id,
            &roots,
            &bodies,
            &joints,
            |commands, env_id| {
                spawn_dog_ground_env(commands, env_id, &isolation, false, &spawn_noise);
            },
        );
        buffers.episode_steps[env_index] = 0;
        buffers.episode_terminated[env_index] = false;
        buffers.episode_truncated[env_index] = false;
        buffers.episode_done[env_index] = false;
        if let Some(actions) = buffers.actions.get_mut(env_index) {
            actions.fill(0.0);
        }
    }
}

fn advance_and_reset_episodes(
    mut commands: Commands,
    mut buffers: ResMut<RlBuffers>,
    reset_policy: Res<EpisodeResetPolicy>,
    config: Res<DogBalanceConfig>,
    isolation: Res<EnvIsolationConfig>,
    spawn_noise: Res<DogSpawnNoise>,
    roots: Query<(Entity, &EnvRoot)>,
    creature_roots: Query<(Entity, &CreatureRoot)>,
    transforms: Query<&Transform>,
    bodies: Query<(Entity, &SimBody)>,
    joints: Query<(Entity, &SimJoint)>,
) {
    let env_count = buffers.episode_steps.len();
    if env_count == 0 {
        return;
    }

    let horizon = buffers.episode_horizon;
    for env_index in 0..env_count {
        buffers.episode_terminated[env_index] = false;
        buffers.episode_truncated[env_index] = false;
        buffers.episode_done[env_index] = false;
        buffers.episode_steps[env_index] = buffers.episode_steps[env_index].saturating_add(1);

        let mut fallen = false;
        for (torso_entity, root) in &creature_roots {
            if root.env_id.index() as usize != env_index {
                continue;
            }
            if let Ok(transform) = transforms.get(torso_entity) {
                let up = transform.rotation * Vec3::Y;
                let height = transform.translation.y;
                if dog_has_fallen(&config, up, height) {
                    fallen = true;
                    buffers.rewards[env_index] += config.fall_penalty;
                }
            }
            break;
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

        let env_id = EnvId::new(env_index as u32);
        reset_env(
            &mut commands,
            env_id,
            &roots,
            &bodies,
            &joints,
            |commands, env_id| {
                spawn_dog_ground_env(commands, env_id, &isolation, false, &spawn_noise);
            },
        );
    }
}
