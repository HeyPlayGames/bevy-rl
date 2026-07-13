//! Concrete environments built on `sim_core`.

mod dog_ground;

pub use dog_ground::{
    dog_quadruped_desc, ground_half_extents, spawn_dog_ground_env, DogGroundEnv, DogGroundPlugin,
    SpawnDogGroundBatch, GROUND_HALF_THICKNESS,
};
