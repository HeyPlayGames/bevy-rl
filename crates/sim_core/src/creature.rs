//! Creature articulation description.
//!
//! Bodies are authored as a kinematic tree: a root pose plus per-body bind
//! rotations, with child placement implied by joint anchors and angles.
//! Absolute poses are computed via forward kinematics at spawn / edit time.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A spawned creature: root entity plus named body and joint entities.
#[derive(Clone, Debug)]
pub struct CreatureInstance {
    pub root: Entity,
    pub bodies: HashMap<String, Entity>,
    pub joints: HashMap<String, Entity>,
}

/// Absolute body poses in creature-local space (from [`compute_body_poses`]).
pub type BodyPoseMap = HashMap<String, Transform>;

/// Serializable / builder description of an articulated creature.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreatureDesc {
    pub name: String,
    /// Pose of the articulation root body (creature-local, before world origin).
    #[serde(default)]
    pub root_pose: Transform,
    pub bodies: Vec<BodyDesc>,
    pub joints: Vec<JointDesc>,
}

impl CreatureDesc {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            root_pose: Transform::IDENTITY,
            bodies: Vec::new(),
            joints: Vec::new(),
        }
    }

    pub fn root_pose(mut self, translation: Vec3, rotation: Quat) -> Self {
        self.root_pose = Transform::from_translation(translation).with_rotation(rotation);
        self
    }

    pub fn body(mut self, body: BodyDesc) -> Self {
        self.bodies.push(body);
        self
    }

    pub fn joint(mut self, joint: JointDesc) -> Self {
        self.joints.push(joint);
        self
    }
}

fn quat_identity() -> Quat {
    Quat::IDENTITY
}

/// Name of the articulation root (body that never appears as a joint child).
pub fn articulation_root_name(creature: &CreatureDesc) -> Option<&str> {
    let mut child_bodies = std::collections::HashSet::new();
    for joint in &creature.joints {
        child_bodies.insert(joint.body_b.as_str());
    }
    creature
        .bodies
        .iter()
        .find(|body| !child_bodies.contains(body.name.as_str()))
        .map(|body| body.name.as_str())
}

/// Revolute angles used as the standing / zero-action pose.
pub fn default_revolute_angles(creature: &CreatureDesc) -> HashMap<String, f32> {
    creature
        .joints
        .iter()
        .filter_map(|joint| {
            joint
                .kind
                .default_angle()
                .map(|angle| (joint.name.clone(), angle))
        })
        .collect()
}

/// Forward-kinematics absolute poses from the joint tree.
///
/// Missing angle entries are treated as `0`. Revolute angles are clamped to
/// limits when present. Child translation is solved so joint anchors coincide;
/// child orientation is `parent * hinge(angle) * bind_rotation`.
pub fn compute_body_poses(creature: &CreatureDesc, angles: &HashMap<String, f32>) -> BodyPoseMap {
    let mut poses = BodyPoseMap::with_capacity(creature.bodies.len());
    let Some(root_name) = articulation_root_name(creature) else {
        return poses;
    };
    poses.insert(root_name.to_string(), creature.root_pose.with_scale(Vec3::ONE));

    let order = joint_topology_order(creature);
    for joint_index in order {
        let joint = &creature.joints[joint_index];
        let Some(pose_a) = poses.get(&joint.body_a).copied() else {
            continue;
        };
        let Some(body_b) = creature.bodies.iter().find(|body| body.name == joint.body_b) else {
            continue;
        };

        let hinge = match &joint.kind {
            JointKind::Revolute {
                axis,
                angle_limits,
                ..
            } => {
                let requested = angles.get(&joint.name).copied().unwrap_or(0.0);
                let angle = match *angle_limits {
                    Some((min, max)) => requested.clamp(min, max),
                    None => requested,
                };
                let axis = axis.normalize_or_zero();
                if axis.length_squared() < 1e-8 || angle.abs() < 1e-8 {
                    Quat::IDENTITY
                } else {
                    Quat::from_axis_angle(axis, angle)
                }
            }
            JointKind::Spherical { .. } | JointKind::Fixed => Quat::IDENTITY,
        };

        let rotation_b = (pose_a.rotation * hinge * body_b.bind_rotation).normalize();
        let translation_b =
            pose_a.translation + pose_a.rotation * joint.anchor_a - rotation_b * joint.anchor_b;
        poses.insert(
            joint.body_b.clone(),
            Transform {
                translation: translation_b,
                rotation: rotation_b,
                scale: Vec3::ONE,
            },
        );
    }

    // Bodies with no path from the root keep identity (should not happen for a tree).
    for body in &creature.bodies {
        poses
            .entry(body.name.clone())
            .or_insert(Transform::IDENTITY);
    }
    poses
}

/// Standing poses: FK with each revolute's [`JointKind::Revolute::default_angle`].
pub fn compute_default_body_poses(creature: &CreatureDesc) -> BodyPoseMap {
    compute_body_poses(creature, &default_revolute_angles(creature))
}

/// Joint-zero poses: FK with all revolute angles at 0.
pub fn compute_zero_body_poses(creature: &CreatureDesc) -> BodyPoseMap {
    compute_body_poses(creature, &HashMap::new())
}

/// Applies a rigid transform to every pose about `pivot` (creature-local).
pub fn transform_body_poses(
    poses: &mut BodyPoseMap,
    pivot: Vec3,
    translation: Vec3,
    rotation: Quat,
) {
    let rotation = rotation.normalize();
    for pose in poses.values_mut() {
        pose.translation = pivot + rotation * (pose.translation - pivot) + translation;
        pose.rotation = (rotation * pose.rotation).normalize();
    }
}

/// Writes an absolute body pose back into the authoring graph at joint angle 0.
///
/// For the root, updates [`CreatureDesc::root_pose`]. For a child, updates that
/// body's [`BodyDesc::bind_rotation`] and the parent joint's `anchor_a` so the
/// anchors stay coincident at the new pose.
pub fn set_body_pose_at_zero(creature: &mut CreatureDesc, body_name: &str, pose: Transform) {
    let pose = Transform {
        translation: pose.translation,
        rotation: pose.rotation.normalize(),
        scale: Vec3::ONE,
    };

    let Some(root_name) = articulation_root_name(creature).map(str::to_string) else {
        return;
    };
    if body_name == root_name {
        creature.root_pose = pose;
        return;
    }

    let Some(joint_index) = creature
        .joints
        .iter()
        .position(|joint| joint.body_b == body_name)
    else {
        return;
    };

    let zero_poses = compute_zero_body_poses(creature);
    let Some(pose_a) = zero_poses.get(&creature.joints[joint_index].body_a).copied() else {
        return;
    };

    let anchor_b = creature.joints[joint_index].anchor_b;
    let bind_rotation = (pose_a.rotation.inverse() * pose.rotation).normalize();
    let anchor_a =
        pose_a.rotation.inverse() * (pose.translation + pose.rotation * anchor_b - pose_a.translation);

    if let Some(body) = creature.bodies.iter_mut().find(|body| body.name == body_name) {
        body.bind_rotation = bind_rotation;
    }
    creature.joints[joint_index].anchor_a = anchor_a;
}

fn joint_topology_order(creature: &CreatureDesc) -> Vec<usize> {
    let mut child_bodies = std::collections::HashSet::new();
    let mut joints_by_parent: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, joint) in creature.joints.iter().enumerate() {
        child_bodies.insert(joint.body_b.clone());
        joints_by_parent
            .entry(joint.body_a.clone())
            .or_default()
            .push(index);
    }

    let mut queue: Vec<String> = creature
        .bodies
        .iter()
        .filter(|body| !child_bodies.contains(&body.name))
        .map(|body| body.name.clone())
        .collect();

    let mut order = Vec::new();
    let mut visited_joints = std::collections::HashSet::new();
    while let Some(body_name) = queue.pop() {
        let Some(joint_indices) = joints_by_parent.get(&body_name) else {
            continue;
        };
        for &joint_index in joint_indices {
            if !visited_joints.insert(joint_index) {
                continue;
            }
            order.push(joint_index);
            queue.push(creature.joints[joint_index].body_b.clone());
        }
    }
    order
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyDesc {
    pub name: String,
    pub shape: BodyShape,
    /// Density used for mass properties (kg / m^3-ish; Avian density units).
    pub density: f32,
    /// Orientation relative to the parent body at revolute angle 0.
    /// Ignored for the articulation root ([`CreatureDesc::root_pose`] owns that).
    #[serde(default = "quat_identity")]
    pub bind_rotation: Quat,
}

impl BodyDesc {
    pub fn new(name: impl Into<String>, shape: BodyShape) -> Self {
        Self {
            name: name.into(),
            shape,
            density: 200.0,
            bind_rotation: Quat::IDENTITY,
        }
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    pub fn bind_rotation(mut self, rotation: Quat) -> Self {
        self.bind_rotation = rotation.normalize();
        self
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BodyShape {
    /// Capsule along local Y. `length` is the cylindrical section height.
    Capsule {
        radius: f32,
        length: f32,
    },
    Cylinder {
        radius: f32,
        height: f32,
    },
    Cuboid {
        half_extents: Vec3,
    },
    Sphere {
        radius: f32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointDesc {
    pub name: String,
    pub body_a: String,
    pub body_b: String,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub kind: JointKind,
}

impl JointDesc {
    pub fn revolute(
        name: impl Into<String>,
        body_a: impl Into<String>,
        body_b: impl Into<String>,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> Self {
        Self {
            name: name.into(),
            body_a: body_a.into(),
            body_b: body_b.into(),
            anchor_a,
            anchor_b,
            kind: JointKind::Revolute {
                axis,
                angle_limits: None,
                default_angle: 0.0,
            },
        }
    }

    pub fn spherical(
        name: impl Into<String>,
        body_a: impl Into<String>,
        body_b: impl Into<String>,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> Self {
        Self {
            name: name.into(),
            body_a: body_a.into(),
            body_b: body_b.into(),
            anchor_a,
            anchor_b,
            kind: JointKind::Spherical {
                twist_axis: Vec3::Y,
                swing_limits: None,
                twist_limits: None,
            },
        }
    }

    pub fn with_angle_limits(mut self, min: f32, max: f32) -> Self {
        if let JointKind::Revolute {
            ref mut angle_limits,
            ref mut default_angle,
            ..
        } = self.kind
        {
            *angle_limits = Some((min, max));
            *default_angle = 0.5 * (min + max);
        }
        self
    }

    pub fn with_default_angle(mut self, angle: f32) -> Self {
        if let JointKind::Revolute {
            ref mut default_angle,
            ..
        } = self.kind
        {
            *default_angle = angle;
        }
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JointKind {
    Revolute {
        axis: Vec3,
        angle_limits: Option<(f32, f32)>,
        /// Standing / zero-action pose. Mapped to [`crate::ActuatedRevolute::rest_angle`].
        default_angle: f32,
    },
    Spherical {
        twist_axis: Vec3,
        swing_limits: Option<(f32, f32)>,
        twist_limits: Option<(f32, f32)>,
    },
    Fixed,
}

impl JointKind {
    /// Revolute default angle, if this is a revolute joint.
    pub fn default_angle(&self) -> Option<f32> {
        match self {
            Self::Revolute { default_angle, .. } => Some(*default_angle),
            _ => None,
        }
    }
}

/// Marks every revolute joint as actuated, holding `default_angle` at action `0`.
///
/// Action indices follow morphology joint order (revolutes only). Used by the
/// morphology studio physics preview; training creatures may attach a subset.
pub fn attach_default_revolute_actuation(
    commands: &mut Commands,
    instance: &CreatureInstance,
    creature: &CreatureDesc,
) {
    let mut action_index = 0usize;
    for joint in &creature.joints {
        let JointKind::Revolute { default_angle, .. } = joint.kind else {
            continue;
        };
        let Some(&joint_entity) = instance.joints.get(&joint.name) else {
            continue;
        };
        commands.entity(joint_entity).insert((
            crate::ActuatedRevolute {
                action_index,
                rest_angle: default_angle,
            },
            crate::JointTargetAngle(0.0),
        ));
        action_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_leg() -> CreatureDesc {
        CreatureDesc::new("test")
            .root_pose(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY)
            .body(BodyDesc::new(
                "torso",
                BodyShape::Cuboid {
                    half_extents: Vec3::splat(0.1),
                },
            ))
            .body(
                BodyDesc::new(
                    "upper",
                    BodyShape::Capsule {
                        radius: 0.05,
                        length: 0.2,
                    },
                )
                .bind_rotation(Quat::IDENTITY),
            )
            .body(BodyDesc::new(
                "lower",
                BodyShape::Capsule {
                    radius: 0.04,
                    length: 0.2,
                },
            ))
            .joint(
                JointDesc::revolute(
                    "hip",
                    "torso",
                    "upper",
                    Vec3::new(0.0, -0.1, 0.0),
                    Vec3::new(0.0, 0.1, 0.0),
                    Vec3::Z,
                )
                .with_default_angle(0.5),
            )
            .joint(JointDesc::revolute(
                "knee",
                "upper",
                "lower",
                Vec3::new(0.0, -0.1, 0.0),
                Vec3::new(0.0, 0.1, 0.0),
                Vec3::Z,
            ))
    }

    #[test]
    fn zero_poses_keep_anchors_coincident() {
        let creature = sample_leg();
        let poses = compute_zero_body_poses(&creature);
        for joint in &creature.joints {
            let pose_a = poses[&joint.body_a];
            let pose_b = poses[&joint.body_b];
            let world_a = pose_a.translation + pose_a.rotation * joint.anchor_a;
            let world_b = pose_b.translation + pose_b.rotation * joint.anchor_b;
            assert!(
                world_a.distance(world_b) < 1e-5,
                "joint {} anchors separated by {}",
                joint.name,
                world_a.distance(world_b)
            );
        }
    }

    #[test]
    fn default_angle_rotates_about_parent_axis() {
        let creature = sample_leg();
        let zero = compute_zero_body_poses(&creature);
        let posed = compute_default_body_poses(&creature);
        let upper_zero = zero["upper"].rotation;
        let upper_posed = posed["upper"].rotation;
        let expected = Quat::from_axis_angle(Vec3::Z, 0.5) * upper_zero;
        let delta = (upper_posed.inverse() * expected).normalize();
        assert!(delta.xyz().length() < 1e-4, "unexpected upper rotation {delta:?}");
    }
}
