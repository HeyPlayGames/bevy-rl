use burn::tensor::{backend::Backend, Tensor};

/// On-policy rollout storage for continuous-control PPO.
pub struct RolloutBatch {
    pub observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub log_probs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    /// True terminals only (not time-limit truncations). Used by GAE bootstrap.
    pub terminations: Vec<f32>,
    /// Time-limit cuts. Used for episode-return metrics, not GAE bootstrap.
    pub truncations: Vec<f32>,
    pub env_count: usize,
    pub step_count: usize,
    pub observation_dim: usize,
    pub action_dim: usize,
}

impl RolloutBatch {
    pub fn new(
        env_count: usize,
        step_count: usize,
        observation_dim: usize,
        action_dim: usize,
    ) -> Self {
        let transitions = env_count * step_count;
        Self {
            observations: vec![0.0; transitions * observation_dim],
            actions: vec![0.0; transitions * action_dim],
            log_probs: vec![0.0; transitions],
            rewards: vec![0.0; transitions],
            values: vec![0.0; transitions],
            terminations: vec![0.0; transitions],
            truncations: vec![0.0; transitions],
            env_count,
            step_count,
            observation_dim,
            action_dim,
        }
    }

    pub fn store_step(
        &mut self,
        step: usize,
        observations: &[Vec<f32>],
        actions: &[Vec<f32>],
        log_probs: &[f32],
        rewards: &[f32],
        values: &[f32],
        terminations: &[bool],
        truncations: &[bool],
    ) {
        for env_index in 0..self.env_count {
            let flat = step * self.env_count + env_index;
            let observation_offset = flat * self.observation_dim;
            let action_offset = flat * self.action_dim;
            self.observations[observation_offset..observation_offset + self.observation_dim]
                .copy_from_slice(&observations[env_index]);
            self.actions[action_offset..action_offset + self.action_dim]
                .copy_from_slice(&actions[env_index]);
            self.log_probs[flat] = log_probs[env_index];
            self.rewards[flat] = rewards[env_index];
            self.values[flat] = values[env_index];
            self.terminations[flat] = if terminations[env_index] { 1.0 } else { 0.0 };
            self.truncations[flat] = if truncations[env_index] { 1.0 } else { 0.0 };
        }
    }

    pub fn transition_count(&self) -> usize {
        self.env_count * self.step_count
    }

    /// Mean sum of rewards per episode (terminated, truncated, or open at rollout end).
    pub fn mean_episode_return(&self) -> f32 {
        mean_episode_return(
            &self.rewards,
            &self.terminations,
            &self.truncations,
            self.env_count,
            self.step_count,
        )
    }

    /// Mean episode length in steps (terminated, truncated, or open at rollout end).
    pub fn mean_episode_length(&self) -> f32 {
        mean_episode_length(
            &self.terminations,
            &self.truncations,
            self.env_count,
            self.step_count,
        )
    }
}

/// Average undiscounted episode return over a rollout.
///
/// Episodes end on termination or truncation. Any still-open trajectory at the
/// end of the rollout is counted as its own (partial) episode.
pub fn mean_episode_return(
    rewards: &[f32],
    terminations: &[f32],
    truncations: &[f32],
    env_count: usize,
    step_count: usize,
) -> f32 {
    if env_count == 0 || step_count == 0 {
        return 0.0;
    }

    let mut episode_returns = Vec::new();
    for env_index in 0..env_count {
        let mut episode_return = 0.0;
        for step in 0..step_count {
            let index = step * env_count + env_index;
            episode_return += rewards[index];
            let ended = terminations[index] > 0.5 || truncations[index] > 0.5;
            let last_step = step + 1 == step_count;
            if ended || last_step {
                episode_returns.push(episode_return);
                episode_return = 0.0;
            }
        }
    }

    if episode_returns.is_empty() {
        return 0.0;
    }
    episode_returns.iter().sum::<f32>() / episode_returns.len() as f32
}

/// Average episode length in steps over a rollout.
///
/// Episodes end on termination or truncation. Any still-open trajectory at the
/// end of the rollout is counted as its own (partial) episode.
pub fn mean_episode_length(
    terminations: &[f32],
    truncations: &[f32],
    env_count: usize,
    step_count: usize,
) -> f32 {
    if env_count == 0 || step_count == 0 {
        return 0.0;
    }

    let mut episode_lengths = Vec::new();
    for env_index in 0..env_count {
        let mut episode_length = 0_u32;
        for step in 0..step_count {
            let index = step * env_count + env_index;
            episode_length = episode_length.saturating_add(1);
            let ended = terminations[index] > 0.5 || truncations[index] > 0.5;
            let last_step = step + 1 == step_count;
            if ended || last_step {
                episode_lengths.push(episode_length as f32);
                episode_length = 0;
            }
        }
    }

    if episode_lengths.is_empty() {
        return 0.0;
    }
    episode_lengths.iter().sum::<f32>() / episode_lengths.len() as f32
}

/// Generalized Advantage Estimation.
///
/// `terminations` must be true terminals only. Time-limit truncations should be
/// left false so values can bootstrap across the cut.
pub fn gae(
    rewards: &[f32],
    values: &[f32],
    terminations: &[f32],
    last_values: &[f32],
    env_count: usize,
    step_count: usize,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut advantages = vec![0.0; env_count * step_count];
    let mut returns = vec![0.0; env_count * step_count];
    let mut last_advantage = vec![0.0; env_count];

    for step in (0..step_count).rev() {
        for env_index in 0..env_count {
            let index = step * env_count + env_index;
            let next_value = if step + 1 < step_count {
                values[index + env_count]
            } else {
                last_values[env_index]
            };
            let next_nonterminal = 1.0 - terminations[index];
            let delta = rewards[index] + gamma * next_value * next_nonterminal - values[index];
            last_advantage[env_index] =
                delta + gamma * lambda * next_nonterminal * last_advantage[env_index];
            advantages[index] = last_advantage[env_index];
            returns[index] = advantages[index] + values[index];
        }
    }

    (advantages, returns)
}

pub fn tensor_from_slice<B: Backend, const D: usize>(
    data: &[f32],
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, 1>::from_floats(data, device).reshape(shape)
}
