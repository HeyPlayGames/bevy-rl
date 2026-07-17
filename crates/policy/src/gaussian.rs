use burn::tensor::{backend::Backend, Distribution, Tensor};

const LOG_SQRT_2_PI: f32 = 0.918_938_5;

/// Sample `mean + std * noise` with independent Normal(0,1) noise per element.
pub fn sample_diagonal_gaussian<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let std = log_std.clone().exp().unsqueeze_dim(0);
    let noise = mean.random_like(Distribution::Normal(0.0, 1.0));
    mean + noise * std
}

/// Summed log-probability of a diagonal Gaussian over the action dimension.
pub fn diagonal_gaussian_log_prob<B: Backend>(
    actions: Tensor<B, 2>,
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let log_std = log_std.unsqueeze_dim(0);
    let std = log_std.clone().exp();
    let variance = std.clone().powf_scalar(2.0);
    let log_prob = (actions - mean).powf_scalar(2.0) / variance * -0.5 - log_std - LOG_SQRT_2_PI;
    log_prob.sum_dim(1).squeeze_dim(1)
}

/// Entropy of a diagonal Gaussian (summed over action dims), broadcast to batch.
pub fn diagonal_gaussian_entropy<B: Backend>(log_std: Tensor<B, 1>) -> Tensor<B, 1> {
    // 0.5 + 0.5*ln(2π) + log_std, summed over actions → scalar, then we'll expand in caller
    let per_dim = log_std.add_scalar(0.5 + LOG_SQRT_2_PI);
    per_dim.sum()
}
