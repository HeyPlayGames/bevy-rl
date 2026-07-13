//! Concrete environments built on `sim_core`.

mod dog_ground;

pub use dog_ground::{
    dog_quadruped_desc, spawn_dog_ground_env, DogGroundEnv, DogGroundPlugin, GroundMeshData,
    SpawnDogGroundBatch,
};
