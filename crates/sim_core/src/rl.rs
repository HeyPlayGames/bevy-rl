//! Shared RL step buffers and control ordering.
//!
//! Creature packs fill observations / rewards; trainers and viewers write
//! actions. [`apply_buffered_actions`] maps actions onto [`JointTargetAngle`]
//! before physics.

use bevy::prelude::*;

use crate::control::{apply_joint_targets, ActuatedRevolute, JointTargetAngle};
use crate::env::SimJoint;

/// Identity and dimensions for the creature pack currently in the app.
#[derive(Resource, Clone, Debug)]
pub struct CreatureSpec {
    pub id: &'static str,
    pub observation_dim: usize,
    pub action_dim: usize,
}

/// How many envs to spawn at startup (handled by the creature pack plugin).
#[derive(Resource, Clone, Copy, Debug)]
pub struct SpawnEnvBatch {
    pub count: u32,
    pub interpolate: bool,
}

impl Default for SpawnEnvBatch {
    fn default() -> Self {
        Self {
            count: 8,
            interpolate: false,
        }
    }
}

/// Ordering hook so controllers can write actions before target application.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlSystems {
    /// Reads [`RlBuffers::actions`] into joint targets. Write actions before this set.
    ApplyActions,
}

/// Controls whether time-limit truncations reset the env.
///
/// Trainers should disable this so the final observation stays valid for
/// value bootstrapping. Viewers/headless can leave the default (`true`) so
/// long-running sims periodically get a fresh episode.
#[derive(Resource, Clone, Copy, Debug)]
pub struct EpisodeResetPolicy {
    pub reset_on_truncate: bool,
}

impl Default for EpisodeResetPolicy {
    fn default() -> Self {
        Self {
            reset_on_truncate: true,
        }
    }
}

/// Per-env observation / action / reward buffers used by trainers and viewers.
#[derive(Resource, Clone, Debug, Default)]
pub struct RlBuffers {
    pub observations: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub rewards: Vec<f32>,
    pub episode_steps: Vec<u32>,
    /// True terminal (e.g. fall). PPO/GAE must not bootstrap across these.
    pub episode_terminated: Vec<bool>,
    /// Time-limit cut. Bootstrap value across these; do not treat as failure.
    pub episode_truncated: Vec<bool>,
    /// `terminated || truncated` — episode boundary for bookkeeping / UI.
    pub episode_done: Vec<bool>,
    pub episode_horizon: u32,
}

impl RlBuffers {
    pub fn resize(
        &mut self,
        env_count: usize,
        observation_dim: usize,
        action_dim: usize,
        horizon: u32,
    ) {
        self.observations = (0..env_count)
            .map(|_| vec![0.0; observation_dim])
            .collect();
        self.actions = (0..env_count).map(|_| vec![0.0; action_dim]).collect();
        self.rewards = vec![0.0; env_count];
        self.episode_steps = vec![0; env_count];
        self.episode_terminated = vec![false; env_count];
        self.episode_truncated = vec![false; env_count];
        self.episode_done = vec![false; env_count];
        self.episode_horizon = horizon.max(1);
    }
}

/// Request that every env be despawned and respawned (viewer env-count changes).
#[derive(Message, Clone, Copy, Debug)]
pub struct RespawnAllEnvs {
    pub count: u32,
    pub interpolate: bool,
}

/// Maps [`RlBuffers::actions`] onto [`JointTargetAngle`] for all actuated revolutes.
pub fn apply_buffered_actions(
    buffers: Res<RlBuffers>,
    mut joints: Query<(&ActuatedRevolute, &SimJoint, &mut JointTargetAngle)>,
) {
    for (actuated, sim_joint, mut target) in &mut joints {
        let env_index = sim_joint.env_id.index() as usize;
        let Some(actions) = buffers.actions.get(env_index) else {
            target.0 = 0.0;
            continue;
        };
        target.0 = actions
            .get(actuated.action_index)
            .copied()
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0);
    }
}

/// Registers control set ordering and the shared action → target system.
pub fn configure_control_systems(app: &mut App) {
    app.configure_sets(
        FixedUpdate,
        ControlSystems::ApplyActions.before(apply_joint_targets),
    )
    .add_systems(
        FixedUpdate,
        apply_buffered_actions.in_set(ControlSystems::ApplyActions),
    );
}
