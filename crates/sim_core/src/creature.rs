//! Creature articulation description.
//!
//! Avian solves joints; this format owns the authoring graph (bodies + joints)
//! that we expand into rigid bodies at spawn time.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A spawned creature: root entity plus named body entities.
#[derive(Clone, Debug)]
pub struct CreatureInstance {
    pub root: Entity,
    pub bodies: HashMap<String, Entity>,
    pub joints: Vec<Entity>,
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
    Capsule { radius: f32, length: f32 },
    Cylinder { radius: f32, height: f32 },
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
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
