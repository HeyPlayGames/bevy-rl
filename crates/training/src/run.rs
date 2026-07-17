//! End-to-end PPO training loop over a Bevy + Avian headless sim.

use std::path::PathBuf;
use std::time::Instant;

use avian3d::prelude::*;
use bevy::ecs::system::RunSystemError;
use bevy::prelude::*;
use burn::{
    backend::{Autodiff, Wgpu},
    module::AutodiffModule,
    tensor::{Transaction, Tensor},
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
/// a consistent spawn (typically a pack-provided soft-reset system).
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
        // Apply resets, then clear episode counters so the horizon run is clean.
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
        // Rollout inference skips Autodiff; only actions sync each step (Bevy needs them).
        // Log-probs / values stay on device and download once after the horizon.
        let inference_model = model.valid();
        let mut pending_log_probs =
            Vec::with_capacity(episode_horizon);
        let mut pending_values = Vec::with_capacity(episode_horizon);

        for step in 0..episode_horizon {
            let observations = {
                let world = app.world();
                world.resource::<RlBuffers>().observations.clone()
            };

            let flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let observation_tensor =
                Tensor::<InferenceBackend, 1>::from_floats(flat.as_slice(), &device)
                    .reshape([env_count, observation_dim]);

            let (actions_tensor, log_probs_tensor, values_tensor) =
                inference_model.act(observation_tensor);
            pending_log_probs.push(log_probs_tensor);
            pending_values.push(values_tensor);

            let action_values = match Transaction::<InferenceBackend>::default()
                .register(actions_tensor)
                .try_execute()
            {
                Ok(data) => data
                    .into_iter()
                    .next()
                    .and_then(|tensor_data| tensor_data.to_vec::<f32>().ok())
                    .unwrap_or_default(),
                Err(error) => {
                    eprintln!("failed to read actions at step {step}: {error}");
                    std::process::exit(1);
                }
            };

            let stored_actions = {
                let world = app.world_mut();
                let mut buffers = world.resource_mut::<RlBuffers>();
                for env_index in 0..env_count {
                    let offset = env_index * action_dim;
                    if offset + action_dim <= action_values.len() {
                        buffers.actions[env_index]
                            .copy_from_slice(&action_values[offset..offset + action_dim]);
                    }
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
                &rewards,
                &terminations,
                &truncations,
            );
        }

        let last_values_tensor = {
            let observations = {
                let world = app.world();
                world.resource::<RlBuffers>().observations.clone()
            };
            let flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let observation_tensor =
                Tensor::<InferenceBackend, 1>::from_floats(flat.as_slice(), &device)
                    .reshape([env_count, observation_dim]);
            inference_model
                .forward(observation_tensor)
                .value
                .reshape([env_count])
        };

        let log_probs_tensor = Tensor::cat(pending_log_probs, 0);
        let values_tensor = Tensor::cat(pending_values, 0);
        let (log_prob_values, value_values, last_values) =
            match Transaction::<InferenceBackend>::default()
                .register(log_probs_tensor)
                .register(values_tensor)
                .register(last_values_tensor)
                .try_execute()
            {
                Ok(data) => {
                    let mut data = data.into_iter();
                    let log_probs = data
                        .next()
                        .and_then(|tensor_data| tensor_data.to_vec::<f32>().ok())
                        .unwrap_or_default();
                    let values = data
                        .next()
                        .and_then(|tensor_data| tensor_data.to_vec::<f32>().ok())
                        .unwrap_or_default();
                    let bootstrap = data
                        .next()
                        .and_then(|tensor_data| tensor_data.to_vec::<f32>().ok())
                        .unwrap_or_else(|| vec![0.0; env_count]);
                    (log_probs, values, bootstrap)
                }
                Err(error) => {
                    eprintln!("failed to read rollout policy outputs: {error}");
                    std::process::exit(1);
                }
            };
        rollout.fill_policy_outputs(&log_prob_values, &value_values);

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
