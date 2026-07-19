//! Burn train TUI metrics dashboard for the custom PPO loop.

use std::io::IsTerminal;
use std::sync::Arc;

use burn::data::dataloader::Progress;
use burn::train::{
    metric::{
        MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAttributes, NumericEntry,
        SerializedEntry,
    },
    renderer::{
        tui::TuiMetricsRendererWrapper, MetricState, MetricsRenderer, MetricsRendererTraining,
        ProgressType, TrainingProgress,
    },
    Interrupter,
};

/// Live Burn TUI dashboard for PPO loss / reward metrics (no-op when stdout is not a terminal).
pub struct TrainerDashboard {
    renderer: Option<TuiMetricsRendererWrapper>,
    interrupter: Interrupter,
    loss_metric_id: MetricId,
    reward_metric_id: MetricId,
    episode_return_metric_id: MetricId,
    update_time_metric_id: MetricId,
    total_updates: usize,
}

impl TrainerDashboard {
    /// Starts the Burn metrics TUI when stdout is a terminal.
    pub fn new(total_updates: usize) -> Self {
        let interrupter = Interrupter::new();
        let loss_metric_id = MetricId::new(Arc::new("Loss".to_string()));
        let reward_metric_id = MetricId::new(Arc::new("Mean Reward".to_string()));
        let episode_return_metric_id = MetricId::new(Arc::new("Mean Episode Return".to_string()));
        let update_time_metric_id = MetricId::new(Arc::new("Update Time".to_string()));

        let mut renderer = if std::io::stdout().is_terminal() {
            Some(TuiMetricsRendererWrapper::new(interrupter.clone(), None))
        } else {
            None
        };

        if let Some(renderer) = renderer.as_mut() {
            renderer.register_metric(MetricDefinition {
                metric_id: loss_metric_id.clone(),
                name: "Loss".to_string(),
                description: Some("PPO loss from the last minibatch of the update".to_string()),
                attributes: MetricAttributes::Numeric(NumericAttributes {
                    unit: None,
                    higher_is_better: false,
                }),
            });
            renderer.register_metric(MetricDefinition {
                metric_id: reward_metric_id.clone(),
                name: "Mean Reward".to_string(),
                description: Some("Mean per-step reward across the rollout batch".to_string()),
                attributes: MetricAttributes::Numeric(NumericAttributes {
                    unit: None,
                    higher_is_better: true,
                }),
            });
            renderer.register_metric(MetricDefinition {
                metric_id: episode_return_metric_id.clone(),
                name: "Mean Episode Return".to_string(),
                description: Some(
                    "Mean undiscounted return per episode (falls / timeouts / open at end)"
                        .to_string(),
                ),
                attributes: MetricAttributes::Numeric(NumericAttributes {
                    unit: None,
                    higher_is_better: true,
                }),
            });
            renderer.register_metric(MetricDefinition {
                metric_id: update_time_metric_id.clone(),
                name: "Update Time".to_string(),
                description: Some(
                    "Wall-clock seconds for one PPO update (rollout + policy update)".to_string(),
                ),
                attributes: MetricAttributes::Numeric(NumericAttributes {
                    unit: Some("s".into()),
                    higher_is_better: false,
                }),
            });
        }

        Self {
            renderer,
            interrupter,
            loss_metric_id,
            reward_metric_id,
            episode_return_metric_id,
            update_time_metric_id,
            total_updates,
        }
    }

    /// True when the user requested stop from the TUI (`q` then `s`).
    pub fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    /// Push loss / reward / timing metrics and refresh the TUI progress bar for this update.
    pub fn record_update(
        &mut self,
        update_index: usize,
        loss: f32,
        mean_reward: f32,
        mean_episode_return: f32,
        update_time_secs: f64,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let loss_value = f64::from(loss);
        let reward_value = f64::from(mean_reward);
        let episode_return_value = f64::from(mean_episode_return);

        let loss_numeric = NumericEntry::Value(loss_value);
        renderer.update_train(MetricState::Numeric(
            MetricEntry::new(
                self.loss_metric_id.clone(),
                SerializedEntry::new(format!("{loss_value:.4}"), loss_numeric.serialize()),
            ),
            loss_numeric,
        ));

        let reward_numeric = NumericEntry::Value(reward_value);
        renderer.update_train(MetricState::Numeric(
            MetricEntry::new(
                self.reward_metric_id.clone(),
                SerializedEntry::new(format!("{reward_value:.4}"), reward_numeric.serialize()),
            ),
            reward_numeric,
        ));

        let episode_return_numeric = NumericEntry::Value(episode_return_value);
        renderer.update_train(MetricState::Numeric(
            MetricEntry::new(
                self.episode_return_metric_id.clone(),
                SerializedEntry::new(
                    format!("{episode_return_value:.4}"),
                    episode_return_numeric.serialize(),
                ),
            ),
            episode_return_numeric,
        ));

        let update_time_numeric = NumericEntry::Value(update_time_secs);
        renderer.update_train(MetricState::Numeric(
            MetricEntry::new(
                self.update_time_metric_id.clone(),
                SerializedEntry::new(
                    format!("{update_time_secs:.2}"),
                    update_time_numeric.serialize(),
                ),
            ),
            update_time_numeric,
        ));

        let completed = update_index.saturating_add(1);
        let update_progress = Progress::new(completed, self.total_updates.max(1));
        renderer.render_train(
            TrainingProgress {
                progress: None,
                global_progress: update_progress.clone(),
                iteration: Some(completed),
            },
            vec![ProgressType::Detailed {
                tag: String::from("Update"),
                progress: update_progress,
            }],
        );
    }

    /// Signal end-of-training to the TUI and tear it down.
    pub fn finish(mut self) {
        if let Some(mut renderer) = self.renderer.take() {
            let _ = renderer.on_train_end(None);
        }
    }
}
