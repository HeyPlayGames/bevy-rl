//! Creature articulation description.
//!
//! Avian solves joints; this format owns the authoring graph (bodies + joints)
//! that we expand into rigid bodies at spawn time.

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

/// Serializable / builder description of an articulated creature.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreatureDesc {
    pub name: String,
    pub bodies: Vec<BodyDesc>,
    pub joints: Vec<JointDesc>,
}

impl CreatureDesc {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bodies: Vec::new(),
            joints: Vec::new(),
        }
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

/// Applies revolute joint angles (radians) to body poses via forward kinematics.
///
/// Joints are processed parent-before-child. Missing names are skipped. When a
/// joint has angle limits, the requested angle is clamped into that range.
pub fn apply_revolute_angles(creature: &mut CreatureDesc, angles: &HashMap<String, f32>) {
    if angles.is_empty() {
        return;
    }

    let order = revolute_joint_topology_order(creature);
    for joint_index in order {
        let joint = &creature.joints[joint_index];
        let Some(&requested) = angles.get(&joint.name) else {
            continue;
        };
        let JointKind::Revolute {
            axis,
            angle_limits,
        } = &joint.kind
        else {
            continue;
        };

        let angle = match *angle_limits {
            Some((min, max)) => requested.clamp(min, max),
            None => requested,
        };
        if angle.abs() < 1e-8 {
            continue;
        }

        let body_a_name = joint.body_a.clone();
        let body_b_name = joint.body_b.clone();
        let anchor_a = joint.anchor_a;
        let axis = *axis;

        let body_a_index = body_index(creature, &body_a_name);
        let Some(body_a_index) = body_a_index else {
            continue;
        };
        let pose_a = creature.bodies[body_a_index].pose;
        let pivot = pose_a.translation + pose_a.rotation * anchor_a;
        let world_axis = (pose_a.rotation * axis).normalize_or_zero();
        if world_axis.length_squared() < 1e-8 {
            continue;
        }
        let delta = Quat::from_axis_angle(world_axis, angle);

        let descendants = body_descendants(creature, &body_b_name);
        for body_name in descendants {
            let Some(index) = body_index(creature, &body_name) else {
                continue;
            };
            let pose = &mut creature.bodies[index].pose;
            pose.translation = pivot + delta * (pose.translation - pivot);
            pose.rotation = (delta * pose.rotation).normalize();
        }
    }
}

/// Applies a rigid transform to every body about `pivot` (creature-local).
pub fn transform_creature_poses(
    creature: &mut CreatureDesc,
    pivot: Vec3,
    translation: Vec3,
    rotation: Quat,
) {
    let rotation = rotation.normalize();
    for body in &mut creature.bodies {
        body.pose.translation =
            pivot + rotation * (body.pose.translation - pivot) + translation;
        body.pose.rotation = (rotation * body.pose.rotation).normalize();
    }
}

fn body_index(creature: &CreatureDesc, name: &str) -> Option<usize> {
    creature.bodies.iter().position(|body| body.name == name)
}

fn revolute_joint_topology_order(creature: &CreatureDesc) -> Vec<usize> {
    let mut child_bodies = std::collections::HashSet::new();
    let mut joints_by_parent: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, joint) in creature.joints.iter().enumerate() {
        if !matches!(joint.kind, JointKind::Revolute { .. }) {
            continue;
        }
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

fn body_descendants(creature: &CreatureDesc, root: &str) -> Vec<String> {
    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    for joint in &creature.joints {
        children
            .entry(joint.body_a.clone())
            .or_default()
            .push(joint.body_b.clone());
    }

    let mut out = Vec::new();
    let mut stack = vec![root.to_string()];
    let mut seen = std::collections::HashSet::new();
    while let Some(name) = stack.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        out.push(name.clone());
        if let Some(child_names) = children.get(&name) {
            stack.extend(child_names.iter().cloned());
        }
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyDesc {
    pub name: String,
    pub shape: BodyShape,
    /// Density used for mass properties (kg / m^3-ish; Avian density units).
    pub density: f32,
    /// Pose relative to the creature root (env-local, before world origin).
    pub pose: PoseDesc,
}

impl BodyDesc {
    pub fn new(name: impl Into<String>, shape: BodyShape) -> Self {
        Self {
            name: name.into(),
            shape,
            density: 200.0,
            pose: PoseDesc::default(),
        }
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    pub fn pose(mut self, translation: Vec3, rotation: Quat) -> Self {
        self.pose = PoseDesc {
            translation,
            rotation,
        };
        self
    }

    pub fn at(mut self, translation: Vec3) -> Self {
        self.pose.translation = translation;
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PoseDesc {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Default for PoseDesc {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
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
            ..
        } = self.kind
        {
            *angle_limits = Some((min, max));
        }
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JointKind {
    Revolute {
        axis: Vec3,
        angle_limits: Option<(f32, f32)>,
    },
    Spherical {
        twist_axis: Vec3,
        swing_limits: Option<(f32, f32)>,
        twist_limits: Option<(f32, f32)>,
    },
    Fixed,
}
