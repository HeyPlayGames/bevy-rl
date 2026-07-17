//! Shared PPO utilities (rollouts, GAE, update, dashboard) and a ready-made train loop.

mod dashboard;
mod optimizer_checkpoint;
mod ppo;
mod rollout;
mod run;

pub use dashboard::TrainerDashboard;
pub use optimizer_checkpoint::{
    load_optimizer_checkpoint_or_fresh, optimizer_stem_from_weights_stem, save_optimizer_checkpoint,
};
pub use ppo::{adam_optimizer, ppo_update, PpoConfig};
pub use rollout::{gae, mean_episode_length, mean_episode_return, RolloutBatch};
pub use run::{resolve_latest_checkpoint, run_ppo, PpoTrainConfig};
