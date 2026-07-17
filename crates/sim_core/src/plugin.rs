use bevy::app::{RunFixedMainLoop, ScheduleRunnerPlugin};
use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;
use bevy::time::{Fixed, TimeUpdateStrategy};
use bevy_transform_interpolation::prelude::*;
use std::time::Duration;

use crate::env::EnvIsolationConfig;

/// Counts completed fixed simulation steps.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct SimTick(pub u64);

/// Headless runner knobs.
#[derive(Clone, Debug)]
pub struct HeadlessSimConfig {
    pub fixed_hz: f64,
    /// Wall-clock wait between app updates. `Duration::ZERO` = as fast as possible.
    pub runner_wait: Duration,
    /// Exit after this many fixed ticks (`None` = run forever).
    pub max_ticks: Option<u64>,
}

impl Default for HeadlessSimConfig {
    fn default() -> Self {
        Self {
            fixed_hz: 60.0,
            runner_wait: Duration::ZERO,
            max_ticks: None,
        }
    }
}

/// Core plugin: isolation config, fixed timestep, tick counter.
///
/// Does **not** add `PhysicsPlugins` — callers choose headless vs rendered feature sets.
pub struct SimCorePlugin {
    pub fixed_hz: f64,
    pub isolation: EnvIsolationConfig,
    pub interpolate_transforms: bool,
}

impl Default for SimCorePlugin {
    fn default() -> Self {
        Self {
            fixed_hz: 60.0,
            isolation: EnvIsolationConfig::default(),
            interpolate_transforms: false,
        }
    }
}

impl Plugin for SimCorePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.isolation.clone())
            .insert_resource(Time::<Fixed>::from_hz(self.fixed_hz))
            .init_resource::<SimTick>()
            .init_resource::<crate::rl::RlBuffers>()
            .init_resource::<crate::rl::EpisodeResetPolicy>()
            .add_message::<crate::rl::RespawnAllEnvs>()
            .add_systems(FixedUpdate, crate::control::apply_joint_targets)
            .add_systems(FixedLast, bump_sim_tick);

        crate::rl::configure_control_systems(app);

        if self.interpolate_transforms && !app.is_plugin_added::<TransformInterpolationPlugin>() {
            app.add_plugins(TransformInterpolationPlugin::default());
        }
    }
}

fn bump_sim_tick(mut tick: ResMut<SimTick>) {
    tick.0 = tick.0.saturating_add(1);
}

/// Minimal headless Bevy + schedule runner. Add physics / env plugins after this.
///
/// Uses [`TimeUpdateStrategy::ManualDuration`] so each app update advances one
/// fixed step regardless of wall clock — throughput-first when `runner_wait` is zero.
pub fn configure_headless_app(app: &mut App, config: &HeadlessSimConfig) {
    let step = Duration::from_secs_f64(1.0 / config.fixed_hz);
    app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(config.runner_wait)))
        .add_plugins(TransformPlugin)
        .insert_resource(Time::<Fixed>::from_hz(config.fixed_hz))
        .insert_resource(TimeUpdateStrategy::ManualDuration(step));

    if let Some(max_ticks) = config.max_ticks {
        app.insert_resource(MaxTicks(max_ticks))
            .add_systems(RunFixedMainLoop, exit_after_max_ticks);
    }
}

#[derive(Resource, Clone, Copy)]
struct MaxTicks(u64);

fn exit_after_max_ticks(
    tick: Res<SimTick>,
    max_ticks: Res<MaxTicks>,
    mut exit: MessageWriter<AppExit>,
) {
    if tick.0 >= max_ticks.0 {
        exit.write(AppExit::Success);
    }
}
