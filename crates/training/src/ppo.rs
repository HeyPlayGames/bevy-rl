use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{backend::AutodiffBackend, Tensor},
};
use policy::ActorCritic;

use crate::rollout::{gae, tensor_from_slice, RolloutBatch};

#[derive(Clone, Debug)]
pub struct PpoConfig {
    pub learning_rate: f64,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_epsilon: f32,
    pub entropy_coef: f32,
    pub value_coef: f32,
    pub update_epochs: usize,
    pub minibatch_size: usize,
    pub max_grad_norm: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            entropy_coef: 0.01,
            value_coef: 0.5,
            update_epochs: 4,
            minibatch_size: 256,
            max_grad_norm: 0.5,
        }
    }
}

pub type PpoOptimizer<B> = OptimizerAdaptor<Adam, ActorCritic<B>, B>;

pub fn adam_optimizer<B: AutodiffBackend>(config: &PpoConfig) -> PpoOptimizer<B>
where
    ActorCritic<B>: AutodiffModule<B>,
{
    let mut adam_config = AdamConfig::new().with_epsilon(1e-5);
    if config.max_grad_norm > 0.0 {
        adam_config = adam_config
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)));
    }
    adam_config.init()
}

pub fn ppo_update<B: AutodiffBackend>(
    mut model: ActorCritic<B>,
    mut optimizer: PpoOptimizer<B>,
    rollout: &RolloutBatch,
    last_values: &[f32],
    config: &PpoConfig,
    device: &B::Device,
) -> (ActorCritic<B>, PpoOptimizer<B>, f32)
where
    ActorCritic<B>: AutodiffModule<B>,
{
    let (advantages, returns) = gae(
        &rollout.rewards,
        &rollout.values,
        &rollout.terminations,
        last_values,
        rollout.env_count,
        rollout.step_count,
        config.gamma,
        config.gae_lambda,
    );

    let transition_count = rollout.transition_count();
    let observations = tensor_from_slice::<B, 2>(
        &rollout.observations,
        [transition_count, rollout.observation_dim],
        device,
    );
    let actions = tensor_from_slice::<B, 2>(
        &rollout.actions,
        [transition_count, rollout.action_dim],
        device,
    );
    let old_log_probs = tensor_from_slice::<B, 1>(&rollout.log_probs, [transition_count], device);
    let mut advantages_tensor = tensor_from_slice::<B, 1>(&advantages, [transition_count], device);
    let returns_tensor = tensor_from_slice::<B, 1>(&returns, [transition_count], device);

    let advantage_mean = advantages_tensor.clone().mean();
    let advantage_var = advantages_tensor.clone().var(0);
    let advantage_std = advantage_var.sqrt().clamp_min(1e-8);
    advantages_tensor = (advantages_tensor - advantage_mean) / advantage_std;

    let mut last_loss = 0.0_f32;
    let minibatch_size = config.minibatch_size.min(transition_count).max(1);

    for _ in 0..config.update_epochs {
        let mut indices: Vec<usize> = (0..transition_count).collect();
        let mut state = transition_count as u64 + 0x9E37_79B9;
        for index in (1..indices.len()).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let swap_with = (state as usize) % (index + 1);
            indices.swap(index, swap_with);
        }

        for start in (0..transition_count).step_by(minibatch_size) {
            let end = (start + minibatch_size).min(transition_count);
            let batch_indices = &indices[start..end];
            let index_data: Vec<i32> = batch_indices.iter().map(|value| *value as i32).collect();
            let index_tensor =
                Tensor::<B, 1, burn::tensor::Int>::from_ints(index_data.as_slice(), device);

            let batch_observations = observations.clone().select(0, index_tensor.clone());
            let batch_actions = actions.clone().select(0, index_tensor.clone());
            let batch_old_log_probs = old_log_probs.clone().select(0, index_tensor.clone());
            let batch_advantages = advantages_tensor.clone().select(0, index_tensor.clone());
            let batch_returns = returns_tensor.clone().select(0, index_tensor);

            let (new_log_probs, values, entropy) =
                model.evaluate(batch_observations, batch_actions);
            let ratio = (new_log_probs - batch_old_log_probs).exp();
            let surrogate_1 = ratio.clone() * batch_advantages.clone();
            let clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon);
            let surrogate_2 = clipped_ratio * batch_advantages;
            let policy_loss = -surrogate_1.min_pair(surrogate_2).mean();
            let value_loss = (batch_returns - values).powf_scalar(2.0).mean();
            let loss = policy_loss + value_loss.mul_scalar(config.value_coef)
                - entropy.mul_scalar(config.entropy_coef);

            last_loss = loss.clone().into_scalar().elem::<f32>();
            let gradients = loss.backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(config.learning_rate, model, gradients);
        }
    }

    (model, optimizer, last_loss)
}
