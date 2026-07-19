//! Multi-view client for the dog example pack.

use bevy::prelude::*;
use dog::{dog_morphology_path, load_dog_morphology, DogPlugin};
use sim_viewer::{run_viewer, ViewerCreatureVisuals};

fn parse_env_count() -> u32 {
    let mut env_count = 4_u32;
    for argument in std::env::args().skip(1) {
        if let Ok(value) = argument.parse::<u32>() {
            env_count = value.clamp(1, 64);
        } else {
            warn!("ignoring unrecognized argument '{argument}'");
        }
    }
    env_count
}

fn main() {
    // So CLI arg warnings work before the viewer app is built.
    App::new().add_plugins(bevy::log::LogPlugin::default());

    let creature = load_dog_morphology().unwrap_or_else(|error| {
        panic!(
            "failed to load dog morphology from {}: {error}",
            dog_morphology_path().display()
        );
    });

    run_viewer(
        parse_env_count(),
        ViewerCreatureVisuals {
            creature,
            creature_color: Color::srgb(0.75, 0.55, 0.35),
            ground_color: Color::srgb(0.62, 0.62, 0.66),
        },
        |app| {
            app.add_plugins(DogPlugin);
        },
    );
}
