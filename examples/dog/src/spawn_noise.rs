use std::collections::HashMap;

use bevy::prelude::*;
use sim_core::prelude::*;

/// Small/medium domain randomization applied on every dog spawn/reset.
#[derive(Resource, Clone, Copy, Debug)]
pub struct DogSpawnNoise {
    /// Half-range for root X/Z translation (meters).
    pub position_xy: f32,
    /// Half-range for yaw about world up (radians).
    pub yaw: f32,
    /// Half-range for pitch (radians).
    pub pitch: f32,
    /// Half-range for roll (radians).
    pub roll: f32,
    /// Half-range for each revolute joint angle (radians), clamped to limits.
    pub joint_angle: f32,
}

impl Default for DogSpawnNoise {
    fn default() -> Self {
        Self {
            position_xy: 0.08,
            yaw: 0.25,
            pitch: 0.08,
            roll: 0.08,
            joint_angle: 0.15,
        }
    }
}

/// Samples pose noise and applies it to a standing [`CreatureDesc`].
pub fn apply_dog_spawn_noise(creature: &mut CreatureDesc, noise: &DogSpawnNoise) {
    let mut joint_angles = HashMap::new();
    for joint in &creature.joints {
        if !matches!(joint.kind, JointKind::Revolute { .. }) {
            continue;
        }
        let angle = rand::random_range(-noise.joint_angle..noise.joint_angle);
        joint_angles.insert(joint.name.clone(), angle);
    }
    apply_revolute_angles(creature, &joint_angles);

    let yaw = rand::random_range(-noise.yaw..noise.yaw);
    let pitch = rand::random_range(-noise.pitch..noise.pitch);
    let roll = rand::random_range(-noise.roll..noise.roll);
    let rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);

    let translation = Vec3::new(
        rand::random_range(-noise.position_xy..noise.position_xy),
        0.0,
        rand::random_range(-noise.position_xy..noise.position_xy),
    );

    let pivot = creature
        .bodies
        .iter()
        .find(|body| body.name == "torso")
        .map(|body| body.pose.translation)
        .unwrap_or(Vec3::ZERO);

    transform_creature_poses(creature, pivot, translation, rotation);
}
