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

/// Samples pose noise and returns absolute body poses for spawn / soft reset.
///
/// Joint angles are sampled around each revolute's `default_angle`, then a small
/// rigid root transform is applied about the torso.
pub fn sample_dog_spawn_poses(
    morphology: &CreatureDesc,
    noise: &DogSpawnNoise,
) -> BodyPoseMap {
    let mut joint_angles = HashMap::new();
    for joint in &morphology.joints {
        let JointKind::Revolute {
            default_angle,
            angle_limits,
            ..
        } = joint.kind
        else {
            continue;
        };
        let mut angle = default_angle + rand::random_range(-noise.joint_angle..noise.joint_angle);
        if let Some((min, max)) = angle_limits {
            angle = angle.clamp(min, max);
        }
        joint_angles.insert(joint.name.clone(), angle);
    }

    let mut poses = compute_body_poses(morphology, &joint_angles);

    let yaw = rand::random_range(-noise.yaw..noise.yaw);
    let pitch = rand::random_range(-noise.pitch..noise.pitch);
    let roll = rand::random_range(-noise.roll..noise.roll);
    let rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);

    let translation = Vec3::new(
        rand::random_range(-noise.position_xy..noise.position_xy),
        0.0,
        rand::random_range(-noise.position_xy..noise.position_xy),
    );

    let pivot = poses
        .get("torso")
        .map(|pose| pose.translation)
        .unwrap_or(Vec3::ZERO);

    transform_body_poses(&mut poses, pivot, translation, rotation);
    poses
}
