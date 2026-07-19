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

/// One controlled revolute degree of freedom. `action_index` is stable across resets.
///
/// `rest_angle` is the joint angle commanded when the normalized action is `0`
/// (from morphology [`crate::JointKind::Revolute::default_angle`]). Action `-1` /
/// `+1` still map to the joint angle limits.
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
/// joint's min / max angle limits. Uses a critically damped spring-damper motor
/// at 20 Hz (stiffer than Avian's 5 Hz default) for snappier joint tracking.
///
/// Runs in [`FixedUpdate`] so Avian's following physics step sees the targets.
pub fn apply_joint_targets(
    mut joints: Query<(&mut RevoluteJoint, &ActuatedRevolute, &JointTargetAngle)>,
) {
    const MOTOR_MODEL: MotorModel = MotorModel::SpringDamper {
        frequency: 20.0,
        damping_ratio: 1.0,
    };

    for (mut joint, actuated, command) in &mut joints {
        let Some(limits) = joint.angle_limit else {
            continue;
        };

        let target_angle =
            action_to_target_angle(command.0, limits.min, limits.max, actuated.rest_angle);

        if !joint.motor.enabled {
            joint.motor = AngularMotor::new(MOTOR_MODEL);
        }
        joint.motor.enabled = true;
        joint.motor.motor_model = MOTOR_MODEL;
        joint.motor.target_position = target_angle;
        joint.motor.target_velocity = 0.0;
    }
}

/// Relative hinge angle (radians) of a revolute joint from body rotations.
///
/// Matches Avian's motor / limit angle: frame bases from spawn (morphology zero)
/// are included so angle 0 is the authored relative pose, not identity orientations.
pub fn revolute_angle(joint: &RevoluteJoint, rotation_a: Quat, rotation_b: Quat) -> f32 {
    let basis1 = joint.local_basis1().unwrap_or(Quat::IDENTITY);
    let basis2 = joint.local_basis2().unwrap_or(Quat::IDENTITY);
    let hinge = joint.hinge_axis.normalize_or_zero();
    if hinge.length_squared() < 1e-8 {
        return 0.0;
    }

    let axis1 = (rotation_a * basis1 * hinge).normalize_or_zero();
    let ortho = hinge.any_orthonormal_vector();
    let direction1 = (rotation_a * basis1 * ortho).normalize_or_zero();
    let direction2 = (rotation_b * basis2 * ortho).normalize_or_zero();
    let sin_angle = direction1.cross(direction2).dot(axis1);
    let cos_angle = direction1.dot(direction2);
    sin_angle.atan2(cos_angle)
}

/// Relative angular velocity about the hinge axis (rad/s).
pub fn revolute_angular_velocity(
    joint: &RevoluteJoint,
    rotation_a: Quat,
    angular_velocity_a: Vec3,
    angular_velocity_b: Vec3,
) -> f32 {
    let basis1 = joint.local_basis1().unwrap_or(Quat::IDENTITY);
    let hinge = joint.hinge_axis.normalize_or_zero();
    if hinge.length_squared() < 1e-8 {
        return 0.0;
    }
    let world_axis = (rotation_a * basis1 * hinge).normalize_or_zero();
    (angular_velocity_b - angular_velocity_a).dot(world_axis)
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
