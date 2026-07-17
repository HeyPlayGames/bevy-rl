//! PPO trainer for the dog example pack.

use std::path::PathBuf;

use bevy::ecs::system::RunSystemOnce;
use dog::{reset_all_envs, DogPlugin, CREATURE_ID, DOG_ACTION_DIM, DOG_OBS_DIM};
use training::{resolve_latest_checkpoint, run_ppo, PpoTrainConfig};

struct TrainerArgs {
    env_count: usize,
    total_updates: usize,
    load_path: Option<PathBuf>,
}

fn parse_args() -> TrainerArgs {
    let mut env_count = 16_usize;
    let mut total_updates = 50_usize;
    let mut load_path = None;
    let mut positionals = Vec::new();
    let mut arguments = std::env::args().skip(1).peekable();

    while let Some(argument) = arguments.next() {
        if argument == "--load" {
            match arguments.peek() {
                Some(next) if !next.starts_with('-') && next.parse::<usize>().is_err() => {
                    if let Some(path) = arguments.next() {
                        load_path = Some(PathBuf::from(path));
                    }
                }
                _ => {
                    load_path = Some(resolve_latest_checkpoint(CREATURE_ID));
                }
            }
            continue;
        }

        if let Ok(value) = argument.parse::<usize>() {
            positionals.push(value);
        } else {
            eprintln!("warning: ignoring unrecognized argument '{argument}'");
        }
    }

    if let Some(value) = positionals.first().copied() {
        env_count = value;
    }
    if let Some(value) = positionals.get(1).copied() {
        total_updates = value;
    }

    TrainerArgs {
        env_count,
        total_updates,
        load_path,
    }
}

fn main() {
    let args = parse_args();

    let config = PpoTrainConfig {
        creature_id: CREATURE_ID,
        observation_dim: DOG_OBS_DIM,
        action_dim: DOG_ACTION_DIM,
        env_count: args.env_count,
        total_updates: args.total_updates,
        load_path: args.load_path,
        ..PpoTrainConfig::default()
    };

    run_ppo(
        config,
        |app| {
            app.add_plugins(DogPlugin);
        },
        |world| world.run_system_once(reset_all_envs).map(|_| ()),
    );
}
