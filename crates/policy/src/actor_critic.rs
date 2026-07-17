use std::path::{Path, PathBuf};

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

use crate::gaussian::{
    diagonal_gaussian_entropy, diagonal_gaussian_log_prob, sample_diagonal_gaussian,
};

/// Tunable actor-critic architecture (loaded from JSON; does not include env dims).
///
/// `hidden_dims` is the MLP backbone width sequence, e.g. `[512, 512, 512]` for three
/// hidden layers. Empty means no hidden layers (linear heads on raw observations).
#[derive(Config, Debug)]
pub struct ActorCriticArchConfig {
    #[config(default = "vec![512, 512, 512]")]
    pub hidden_dims: Vec<usize>,
    #[config(default = "-0.5")]
    pub initial_log_std: f32,
}

impl ActorCriticArchConfig {
    /// Default path shipped with the `policy` crate (`config/actor_critic.json`).
    pub fn default_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/actor_critic.json")
    }

    /// Load from `path`, or fall back to Burn defaults when the file is missing.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self, burn::config::ConfigError> {
        let path = path.as_ref();
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::new())
        }
    }
}

/// Full actor-critic config: env dims + architecture hyperparameters.
#[derive(Config, Debug)]
pub struct ActorCriticConfig {
    pub observation_dim: usize,
    pub action_dim: usize,
    #[config(default = "vec![512, 512, 512]")]
    pub hidden_dims: Vec<usize>,
    #[config(default = "-0.5")]
    pub initial_log_std: f32,
}

impl ActorCriticConfig {
    /// Build from observation/action dims and a JSON architecture file.
    ///
    /// When `arch_path` is `None`, uses [`ActorCriticArchConfig::default_path`].
    pub fn from_arch_file(
        observation_dim: usize,
        action_dim: usize,
        arch_path: Option<&Path>,
    ) -> Result<Self, burn::config::ConfigError> {
        let path = arch_path
            .map(Path::to_path_buf)
            .unwrap_or_else(ActorCriticArchConfig::default_path);
        let arch = ActorCriticArchConfig::load_or_default(&path)?;
        Ok(Self::from_arch(observation_dim, action_dim, &arch))
    }

    pub fn from_arch(
        observation_dim: usize,
        action_dim: usize,
        arch: &ActorCriticArchConfig,
    ) -> Self {
        Self::new(observation_dim, action_dim)
            .with_hidden_dims(arch.hidden_dims.clone())
            .with_initial_log_std(arch.initial_log_std)
    }

    /// Feature width feeding the actor/critic heads (last hidden dim, or obs dim).
    pub fn feature_dim(&self) -> usize {
        self.hidden_dims
            .last()
            .copied()
            .unwrap_or(self.observation_dim)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ActorCritic<B> {
        let mut backbone = Vec::with_capacity(self.hidden_dims.len());
        let mut input_dim = self.observation_dim;
        for &output_dim in &self.hidden_dims {
            backbone.push(LinearConfig::new(input_dim, output_dim).init(device));
            input_dim = output_dim;
        }

        let feature_dim = self.feature_dim();
        let log_std = Tensor::<B, 1>::full([self.action_dim], self.initial_log_std, device);
        ActorCritic {
            backbone,
            activation: Relu::new(),
            actor_mean: LinearConfig::new(feature_dim, self.action_dim).init(device),
            critic: LinearConfig::new(feature_dim, 1).init(device),
            log_std: Param::from_tensor(log_std),
        }
    }
}

#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    backbone: Vec<Linear<B>>,
    activation: Relu,
    actor_mean: Linear<B>,
    critic: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
}

pub struct ActorCriticOutput<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub log_std: Tensor<B, 1>,
    pub value: Tensor<B, 2>,
}

impl<B: Backend> ActorCritic<B> {
    pub fn forward(&self, observations: Tensor<B, 2>) -> ActorCriticOutput<B> {
        let mut hidden = observations;
        for layer in &self.backbone {
            hidden = self.activation.forward(layer.forward(hidden));
        }
        let mean = self.actor_mean.forward(hidden.clone()).tanh();
        let value = self.critic.forward(hidden);
        ActorCriticOutput {
            mean,
            log_std: self.log_std.val(),
            value,
        }
    }

    /// Sample actions for a batch of observations. Returns (actions, log_probs, values).
    ///
    /// Actions are clamped to `[-1, 1]` before the log-prob is computed so the
    /// stored probability matches what the env applies after clamping.
    pub fn act(&self, observations: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>) {
        let output = self.forward(observations);
        let actions = sample_diagonal_gaussian(output.mean.clone(), output.log_std.clone());
        let actions = actions.clamp(-1.0, 1.0);
        let log_probs =
            diagonal_gaussian_log_prob(actions.clone(), output.mean, output.log_std.clone());
        let values = output.value.reshape([actions.dims()[0]]);
        (actions, log_probs, values)
    }

    /// Evaluate log-probs / values / entropy for PPO (actions already clamped).
    pub fn evaluate(
        &self,
        observations: Tensor<B, 2>,
        actions: Tensor<B, 2>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let output = self.forward(observations);
        let log_probs =
            diagonal_gaussian_log_prob(actions, output.mean, output.log_std.clone());
        let values = output.value.reshape([log_probs.dims()[0]]);
        let entropy = diagonal_gaussian_entropy(output.log_std);
        (log_probs, values, entropy)
    }
}
