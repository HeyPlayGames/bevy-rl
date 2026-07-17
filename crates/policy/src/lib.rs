//! Actor-critic MLP used for training and later in-sim inference.

mod actor_critic;
mod checkpoint;
mod gaussian;

pub use actor_critic::{ActorCritic, ActorCriticArchConfig, ActorCriticConfig, ActorCriticOutput};
pub use checkpoint::{
    checkpoint_root, creature_checkpoint_dir, latest_checkpoint_stem,
    load_latest_creature_checkpoint, load_policy_checkpoint, resolve_checkpoint_stem,
    save_creature_checkpoint, step_checkpoint_stem, CheckpointError, CheckpointPaths,
    PolicyCheckpointMeta,
};
pub use gaussian::{
    diagonal_gaussian_entropy, diagonal_gaussian_log_prob, sample_diagonal_gaussian,
};
