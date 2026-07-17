//! Generic creature actuation and observation helpers.
//!
//! Creatures mark revolute joints as [`ActuatedRevolute`] and write normalized
//! target angles into [`JointTargetAngle`]. Observations and rewards are
//! assembled by creature-specific code using the shared helpers here.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::env::EnvId;

/// Marks the primary body used for root-relative observations (e.g. torso).
#[derive(Component, Clone, Copy, Debug)]
pub struct CreatureRoot {
    pub env_id: EnvId,
}

/// One controlled revolute DoF. `action_index` is stable across resets.
///
/// `rest_angle` is the joint angle commanded when the normalized action is `0`
/// (typically the spawn / standing pose). Action `-1` / `+1` still map to the
/// joint angle limits.
#[derive(Component, Clone, Copy, Debug)]
pub struct ActuatedRevolute {
    pub action_index: usize,
    pub rest_angle: f32,
}

/// Normalized joint target in `[-1, 1]`, mapped into the revolute angle limits.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct JointTargetAngle(pub f32);

/// Piecewise-linear map: `-1 → min`, `0 → rest`, `+1 → max`.
pub fn action_to_target_angle(action: f32, min: f32, max: f32, rest: f32) -> f32 {
    let action = action.clamp(-1.0, 1.0);
    let rest = rest.clamp(min, max);
    if action >= 0.0 {
        rest + action * (max - rest)
    } else {
        rest + action * (rest - min)
    }
}

/// Writes each actuated revolute's motor target from [`JointTargetAngle`].
///
/// Action `0` maps to [`ActuatedRevolute::rest_angle`]; `-1` / `+1` map to the
/// joint's min / max angle limits. Uses Avian's default spring-damper motor
/// model (no custom PD gains).
///
/// Runs in [`FixedUpdate`] so Avian's following physics step sees the targets.
pub fn apply_joint_targets(
    mut joints: Query<(&mut RevoluteJoint, &ActuatedRevolute, &JointTargetAngle)>,
) {
    for (mut joint, actuated, command) in &mut joints {
        let Some(limits) = joint.angle_limit else {
            continue;
        };

        let target_angle = action_to_target_angle(
            command.0,
            limits.min,
            limits.max,
            actuated.rest_angle,
        );

        if !joint.motor.enabled {
            joint.motor = AngularMotor::new(MotorModel::DEFAULT);
        }
        joint.motor.enabled = true;
        joint.motor.target_position = target_angle;
        joint.motor.target_velocity = 0.0;
    }
}

/// Relative hinge angle (radians) of a revolute joint from body rotations.
pub fn revolute_angle(joint: &RevoluteJoint, rotation_a: Quat, rotation_b: Quat) -> f32 {
    let local_axis = joint
        .local_hinge_axis1()
        .unwrap_or(joint.hinge_axis)
        .normalize_or_zero();
    if local_axis.length_squared() < 1e-8 {
        return 0.0;
    }

    let relative = rotation_a.inverse() * rotation_b;
    twist_angle(relative, local_axis)
}

/// Relative angular velocity about the hinge axis (rad/s).
pub fn revolute_angular_velocity(
    joint: &RevoluteJoint,
    rotation_a: Quat,
    angular_velocity_a: Vec3,
    angular_velocity_b: Vec3,
) -> f32 {
    let local_axis = joint
        .local_hinge_axis1()
        .unwrap_or(joint.hinge_axis)
        .normalize_or_zero();
    if local_axis.length_squared() < 1e-8 {
        return 0.0;
    }
    let world_axis = (rotation_a * local_axis).normalize_or_zero();
    (angular_velocity_b - angular_velocity_a).dot(world_axis)
}

fn twist_angle(rotation: Quat, axis: Vec3) -> f32 {
    let axis = axis.normalize_or_zero();
    if axis.length_squared() < 1e-8 {
        return 0.0;
    }
    let vector = Vec3::new(rotation.x, rotation.y, rotation.z);
    let projected = axis * vector.dot(axis);
    let twist = Quat::from_xyzw(projected.x, projected.y, projected.z, rotation.w).normalize();
    let signed = if Vec3::new(twist.x, twist.y, twist.z).dot(axis) < 0.0 {
        -1.0
    } else {
        1.0
    };
    2.0 * twist.w.clamp(-1.0, 1.0).acos() * signed
}

/// Shared reward building blocks for creature objectives.
pub mod reward {
    use bevy::prelude::Vec3;

    /// `1` when fully upright, `-1` when upside down.
    pub fn uprightness(up: Vec3, world_up: Vec3) -> f32 {
        up.normalize_or_zero()
            .dot(world_up.normalize_or_zero())
            .clamp(-1.0, 1.0)
    }

    /// Peaks at `1` when `height == target`, falls off over `tolerance`.
    pub fn height_band(height: f32, target: f32, tolerance: f32) -> f32 {
        let tolerance = tolerance.max(1e-4);
        1.0 - ((height - target).abs() / tolerance).min(1.0)
    }
}
