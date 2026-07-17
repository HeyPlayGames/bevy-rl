//! End-to-end PPO training loop over a Bevy + Avian headless sim.

use std::path::PathBuf;
use std::time::Instant;

use avian3d::prelude::*;
use bevy::ecs::system::RunSystemError;
use bevy::prelude::*;
use burn::{
    backend::{Autodiff, Wgpu},
    module::AutodiffModule,
    tensor::Tensor,
};
use policy::{
    load_policy_checkpoint, save_creature_checkpoint, ActorCritic, ActorCriticArchConfig,
    ActorCriticConfig, PolicyCheckpointMeta,
};
use sim_core::prelude::*;

use crate::{
    adam_optimizer, load_optimizer_checkpoint_or_fresh, ppo_update, save_optimizer_checkpoint,
    PpoConfig, RolloutBatch, TrainerDashboard,
};

type TrainBackend = Autodiff<Wgpu>;
type InferenceBackend = Wgpu;

/// Knobs for [`run_ppo`].
#[derive(Clone, Debug)]
pub struct PpoTrainConfig {
    pub creature_id: &'static str,
    pub observation_dim: usize,
    pub action_dim: usize,
    pub env_count: usize,
    pub total_updates: usize,
    /// Full episode length; PPO trains on every step of the episode.
    pub episode_horizon: usize,
    pub load_path: Option<PathBuf>,
    pub fixed_hz: f64,
    pub isolation: EnvIsolationConfig,
    pub gravity: Vec3,
    pub ppo: PpoConfig,
}

impl Default for PpoTrainConfig {
    fn default() -> Self {
        Self {
            creature_id: "creature",
            observation_dim: 1,
            action_dim: 1,
            env_count: 16,
            total_updates: 50,
            episode_horizon: 300,
            load_path: None,
            fixed_hz: 60.0,
            isolation: EnvIsolationConfig {
                spacing: 40.0,
                grid_columns: 16,
            },
            gravity: Vec3::NEG_Y * 9.81,
            ppo: PpoConfig::default(),
        }
    }
}

/// Run PPO training against a creature pack already registered via `add_plugins`.
///
/// `reset_all_envs` is called at the start of each update so rollouts begin from
/// a consistent spawn (typically a pack-provided despawn+respawn system).
pub fn run_ppo(
    config: PpoTrainConfig,
    add_plugins: impl FnOnce(&mut App),
    mut reset_all_envs: impl FnMut(&mut World) -> Result<(), RunSystemError>,
) {
    let env_count = config.env_count;
    let total_updates = config.total_updates;
    let observation_dim = config.observation_dim;
    let action_dim = config.action_dim;
    let episode_horizon = config.episode_horizon;
    let creature_id = config.creature_id;

    let headless = HeadlessSimConfig {
        fixed_hz: config.fixed_hz,
        runner_wait: std::time::Duration::ZERO,
        max_ticks: None,
    };

    let mut app = App::new();
    configure_headless_app(&mut app, &headless);

    app.add_plugins(PhysicsPlugins::default())
        .add_plugins(SimCorePlugin {
            fixed_hz: headless.fixed_hz,
            isolation: config.isolation.clone(),
            interpolate_transforms: false,
        })
        .insert_resource(Gravity(config.gravity))
        .insert_resource(EpisodeResetPolicy {
            // Keep final states for value bootstrap; truncations must not autoreset.
            reset_on_truncate: false,
        })
        .insert_resource(SpawnEnvBatch {
            count: env_count as u32,
            interpolate: false,
        });
    add_plugins(&mut app);

    // Manual update loops skip `App::run`, so finish plugins (Avian registers diagnostics here).
    app.finish();
    app.cleanup();
    app.update();

    {
        let world = app.world_mut();
        let mut buffers = world.resource_mut::<RlBuffers>();
        buffers.resize(
            env_count,
            observation_dim,
            action_dim,
            episode_horizon as u32,
        );
    }

    app.update();

    let device = Default::default();
    let policy_config = match ActorCriticConfig::from_arch_file(observation_dim, action_dim, None) {
        Ok(config) => config,
        Err(error) => {
            eprintln!(
                "failed to load actor-critic config from {}: {error}",
                ActorCriticArchConfig::default_path().display()
            );
            std::process::exit(1);
        }
    };
    println!(
        "policy arch: hidden_dims={:?} initial_log_std={}",
        policy_config.hidden_dims, policy_config.initial_log_std
    );

    let mut model = match &config.load_path {
        Some(path) => {
            let (loaded, meta) = match load_policy_checkpoint::<InferenceBackend>(
                &device,
                path,
                creature_id,
                &policy_config,
            ) {
                Ok(loaded) => loaded,
                Err(error) => {
                    eprintln!("failed to load checkpoint from {}: {error}", path.display());
                    std::process::exit(1);
                }
            };

            println!(
                "loaded checkpoint update_index={} mean_rewards={} mean_episode_lengths={} from {}",
                meta.update_index,
                meta.mean_rewards.len(),
                meta.mean_episode_lengths.len(),
                path.display()
            );
            ActorCritic::<TrainBackend>::from_inner(loaded)
        }
        None => policy_config.init::<TrainBackend>(&device),
    };

    let mut optimizer = adam_optimizer::<TrainBackend>(&config.ppo);
    if let Some(path) = &config.load_path {
        optimizer = load_optimizer_checkpoint_or_fresh(optimizer, &device, path);
    }

    println!(
        "trainer start: creature={creature_id} envs={env_count} updates={total_updates} episode_horizon={episode_horizon} obs={observation_dim} action={action_dim}"
    );

    let wall_clock = Instant::now();
    let mut dashboard = TrainerDashboard::new(total_updates);
    let mut mean_rewards = Vec::with_capacity(total_updates);
    let mut mean_episode_lengths = Vec::with_capacity(total_updates);
    let mut last_update_index = 0_usize;
    let mut completed_any_update = false;

    for update_index in 0..total_updates {
        if dashboard.should_stop() {
            break;
        }

        if let Err(error) = reset_all_envs(app.world_mut()) {
            eprintln!("failed to reset envs before update {update_index}: {error}");
            std::process::exit(1);
        }
        // Apply respawns, then clear episode counters so the horizon run is clean.
        app.update();
        {
            let mut buffers = app.world_mut().resource_mut::<RlBuffers>();
            for steps in &mut buffers.episode_steps {
                *steps = 0;
            }
            buffers.episode_terminated.fill(false);
            buffers.episode_truncated.fill(false);
            buffers.episode_done.fill(false);
        }

        let mut rollout =
            RolloutBatch::new(env_count, episode_horizon, observation_dim, action_dim);
        let mut step_log_probs = vec![0.0; env_count];
        let mut step_values = vec![0.0; env_count];

        for step in 0..episode_horizon {
            let observations = {
                let world = app.world();
                world.resource::<RlBuffers>().observations.clone()
            };

            let flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let observation_tensor =
                Tensor::<TrainBackend, 1>::from_floats(flat.as_slice(), &device)
                    .reshape([env_count, observation_dim]);

            let (actions_tensor, log_probs_tensor, values_tensor) = model.act(observation_tensor);
            let action_values = actions_tensor.to_data().to_vec::<f32>().unwrap_or_default();
            let log_prob_values = log_probs_tensor
                .to_data()
                .to_vec::<f32>()
                .unwrap_or_default();
            let value_values = values_tensor.to_data().to_vec::<f32>().unwrap_or_default();

            let stored_actions = {
                let world = app.world_mut();
                let mut buffers = world.resource_mut::<RlBuffers>();
                for env_index in 0..env_count {
                    let offset = env_index * action_dim;
                    if offset + action_dim <= action_values.len() {
                        buffers.actions[env_index]
                            .copy_from_slice(&action_values[offset..offset + action_dim]);
                    }
                    step_log_probs[env_index] =
                        log_prob_values.get(env_index).copied().unwrap_or(0.0);
                    step_values[env_index] = value_values.get(env_index).copied().unwrap_or(0.0);
                }
                buffers.actions.clone()
            };

            app.update();

            let (rewards, terminations, truncations) = {
                let world = app.world();
                let buffers = world.resource::<RlBuffers>();
                (
                    buffers.rewards.clone(),
                    buffers.episode_terminated.clone(),
                    buffers.episode_truncated.clone(),
                )
            };

            rollout.store_step(
                step,
                &observations,
                &stored_actions,
                &step_log_probs,
                &rewards,
                &step_values,
                &terminations,
                &truncations,
            );
        }

        let last_values = {
            let observations = {
                let world = app.world();
                world.resource::<RlBuffers>().observations.clone()
            };
            let flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let observation_tensor =
                Tensor::<TrainBackend, 1>::from_floats(flat.as_slice(), &device)
                    .reshape([env_count, observation_dim]);
            let output = model.forward(observation_tensor);
            output
                .value
                .reshape([env_count])
                .to_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; env_count])
        };

        let (updated_model, updated_optimizer, loss) = ppo_update(
            model,
            optimizer,
            &rollout,
            &last_values,
            &config.ppo,
            &device,
        );

        model = updated_model;
        optimizer = updated_optimizer;

        let mean_reward = rollout.rewards.iter().sum::<f32>() / rollout.rewards.len().max(1) as f32;
        let mean_episode_return = rollout.mean_episode_return();
        let mean_episode_length = rollout.mean_episode_length();

        mean_rewards.push(mean_reward);
        mean_episode_lengths.push(mean_episode_length);
        last_update_index = update_index;
        completed_any_update = true;
        dashboard.record_update(update_index, loss, mean_reward, mean_episode_return);
    }

    dashboard.finish();

    if completed_any_update {
        let meta = PolicyCheckpointMeta::from_config(
            creature_id,
            &policy_config,
            last_update_index,
            mean_rewards,
            mean_episode_lengths,
        );

        let inference_model = model.valid();

        match save_creature_checkpoint(&inference_model, &meta) {
            Ok(paths) => {
                println!(
                    "checkpoint saved: {} and {}",
                    paths.latest_weights_stem.with_extension("mpk").display(),
                    paths.step_weights_stem.with_extension("mpk").display()
                );

                for weights_stem in [&paths.latest_weights_stem, &paths.step_weights_stem] {
                    match save_optimizer_checkpoint(&optimizer, weights_stem) {
                        Ok(optim_path) => {
                            println!("optimizer checkpoint saved: {}", optim_path.display());
                        }
                        Err(error) => {
                            eprintln!(
                                "failed to save optimizer checkpoint for {}: {error}",
                                weights_stem.display()
                            );
                            std::process::exit(1);
                        }
                    }
                }
            }
            Err(error) => {
                eprintln!("failed to save checkpoint at end of training: {error}");
                std::process::exit(1);
            }
        }
    }

    println!("trainer done in {:.1}s", wall_clock.elapsed().as_secs_f64());
}

/// Resolve `--load` with no path to the creature's latest checkpoint stem.
pub fn resolve_latest_checkpoint(creature_id: &str) -> PathBuf {
    match policy::latest_checkpoint_stem(creature_id) {
        Ok(stem) => stem,
        Err(error) => {
            eprintln!("failed to resolve latest checkpoint path: {error}");
            std::process::exit(1);
        }
    }
}
