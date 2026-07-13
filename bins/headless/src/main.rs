use std::time::Instant;

use avian3d::prelude::*;
use bevy::prelude::*;
use sim_core::prelude::*;
use sim_envs::{DogGroundPlugin, SpawnDogGroundBatch};

fn main() {
    let env_count: u32 = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);

    let max_ticks: Option<u64> = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok());

    let headless = HeadlessSimConfig {
        fixed_hz: 60.0,
        runner_wait: std::time::Duration::ZERO,
        max_ticks: max_ticks.or(Some(600)),
    };

    let mut app = App::new();
    configure_headless_app(&mut app, &headless);

    app.add_plugins(PhysicsPlugins::default())
        .add_plugins(SimCorePlugin {
            fixed_hz: headless.fixed_hz,
            isolation: EnvIsolationConfig {
                spacing: 40.0,
                grid_columns: 16,
            },
            interpolate_transforms: false,
        })
        .insert_resource(Gravity(Vec3::NEG_Y * 9.81))
        .insert_resource(SpawnDogGroundBatch {
            count: env_count,
            interpolate: false,
        })
        .insert_resource(WallClockStart(Instant::now()))
        .add_plugins(DogGroundPlugin)
        .add_systems(Startup, log_startup)
        .add_systems(Update, report_progress)
        .run();
}

#[derive(Resource)]
struct WallClockStart(Instant);

fn log_startup(batch: Res<SpawnDogGroundBatch>) {
    println!(
        "headless start: envs={} interpolate={}",
        batch.count, batch.interpolate
    );
}

fn report_progress(
    tick: Res<SimTick>,
    wall_clock: Res<WallClockStart>,
    bodies: Query<(&SimBody, &Transform, &Name)>,
) {
    if tick.0 == 0 || !tick.is_changed() {
        return;
    }
    if tick.0 % 60 != 0 {
        return;
    }

    let elapsed = wall_clock.0.elapsed().as_secs_f64();
    let hz = if elapsed > 0.0 {
        tick.0 as f64 / elapsed
    } else {
        0.0
    };

    let mut sample_y = None;
    for (body, transform, name) in &bodies {
        if body.env_id.index() == 0 && name.as_str().starts_with("torso_") {
            sample_y = Some(transform.translation.y);
            break;
        }
    }

    println!(
        "sim_tick={} wall_secs={:.3} effective_fixed_hz={:.1} env0_torso_y={:?}",
        tick.0, elapsed, hz, sample_y
    );
}
