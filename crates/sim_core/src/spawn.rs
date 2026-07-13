use std::collections::HashMap;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_transform_interpolation::prelude::TransformInterpolation;

use crate::creature::{
    BodyDesc, BodyShape, CreatureDesc, CreatureInstance, JointDesc, JointKind,
};
use crate::env::{
    env_collision_layers, env_origin, EnvId, EnvIsolationConfig, EnvRoot, SimBody, SimJoint,
};

/// Spawns an empty env root marker at the isolated world origin.
pub fn spawn_env_root(
    commands: &mut Commands,
    env_id: EnvId,
    isolation: &EnvIsolationConfig,
) -> Entity {
    let origin = env_origin(env_id, isolation);
    commands
        .spawn((
            Name::new(format!("env_{}", env_id.index())),
            EnvRoot { env_id },
            Transform::from_translation(origin),
        ))
        .id()
}

/// Spawns a creature at the env world origin using absolute body poses.
///
/// Bodies are **not** parented under a hierarchy for physics safety: Avian
/// owns their transforms, and we only tag them with [`SimBody`] / [`EnvId`].
pub fn spawn_creature(
    commands: &mut Commands,
    env_id: EnvId,
    world_origin: Vec3,
    creature: &CreatureDesc,
    interpolate: bool,
) -> CreatureInstance {
    let layers = env_collision_layers(env_id);

    let creature_root = commands
        .spawn((
            Name::new(format!("{}_{}", creature.name, env_id.index())),
            EnvRoot { env_id },
            Transform::from_translation(world_origin),
        ))
        .id();

    let mut bodies: HashMap<String, Entity> = HashMap::new();

    for body in &creature.bodies {
        let entity = spawn_body(
            commands,
            env_id,
            world_origin,
            body,
            layers,
            interpolate,
        );
        bodies.insert(body.name.clone(), entity);
    }

    let mut joints = Vec::with_capacity(creature.joints.len());
    for joint in &creature.joints {
        let entity = spawn_joint(commands, env_id, &bodies, joint);
        joints.push(entity);
    }

    CreatureInstance {
        root: creature_root,
        bodies,
        joints,
    }
}

fn spawn_body(
    commands: &mut Commands,
    env_id: EnvId,
    world_origin: Vec3,
    body: &BodyDesc,
    layers: CollisionLayers,
    interpolate: bool,
) -> Entity {
    let collider = collider_from_shape(body.shape);
    let translation = world_origin + body.pose.translation;
    let transform = Transform::from_translation(translation).with_rotation(body.pose.rotation);

    let mut entity_commands = commands.spawn((
        Name::new(format!("{}_{}", body.name, env_id.index())),
        SimBody { env_id },
        RigidBody::Dynamic,
        collider,
        ColliderDensity(body.density),
        layers,
        Friction::new(0.8),
        Restitution::new(0.05),
        transform,
    ));

    if interpolate {
        entity_commands.insert(TransformInterpolation);
    }

    entity_commands.id()
}

fn collider_from_shape(shape: BodyShape) -> Collider {
    match shape {
        BodyShape::Capsule { radius, length } => Collider::capsule(radius, length),
        BodyShape::Cylinder { radius, height } => Collider::cylinder(radius, height),
        BodyShape::Cuboid { half_extents } => Collider::cuboid(
            half_extents.x * 2.0,
            half_extents.y * 2.0,
            half_extents.z * 2.0,
        ),
        BodyShape::Sphere { radius } => Collider::sphere(radius),
    }
}

fn spawn_joint(
    commands: &mut Commands,
    env_id: EnvId,
    bodies: &HashMap<String, Entity>,
    joint: &JointDesc,
) -> Entity {
    let body_a = *bodies
        .get(&joint.body_a)
        .unwrap_or_else(|| panic!("joint `{}`: missing body `{}`", joint.name, joint.body_a));
    let body_b = *bodies
        .get(&joint.body_b)
        .unwrap_or_else(|| panic!("joint `{}`: missing body `{}`", joint.name, joint.body_b));

    match &joint.kind {
        JointKind::Revolute { axis, angle_limits } => {
            let mut revolute = RevoluteJoint::new(body_a, body_b)
                .with_local_anchor1(joint.anchor_a)
                .with_local_anchor2(joint.anchor_b)
                .with_hinge_axis(*axis);
            if let Some((min, max)) = angle_limits {
                revolute = revolute.with_angle_limits(*min, *max);
            }
            commands
                .spawn((
                    Name::new(format!("{}_{}", joint.name, env_id.index())),
                    SimJoint { env_id },
                    revolute,
                    JointCollisionDisabled,
                ))
                .id()
        }
        JointKind::Spherical {
            twist_axis,
            swing_limits,
            twist_limits,
        } => {
            let mut spherical = SphericalJoint::new(body_a, body_b)
                .with_local_anchor1(joint.anchor_a)
                .with_local_anchor2(joint.anchor_b)
                .with_twist_axis(*twist_axis);
            if let Some((min, max)) = swing_limits {
                spherical = spherical.with_swing_limits(*min, *max);
            }
            if let Some((min, max)) = twist_limits {
                spherical = spherical.with_twist_limits(*min, *max);
            }
            commands
                .spawn((
                    Name::new(format!("{}_{}", joint.name, env_id.index())),
                    SimJoint { env_id },
                    spherical,
                    JointCollisionDisabled,
                ))
                .id()
        }
        JointKind::Fixed => commands
            .spawn((
                Name::new(format!("{}_{}", joint.name, env_id.index())),
                SimJoint { env_id },
                FixedJoint::new(body_a, body_b)
                    .with_local_anchor1(joint.anchor_a)
                    .with_local_anchor2(joint.anchor_b),
                JointCollisionDisabled,
            ))
            .id(),
    }
}

/// Despawns everything tagged with this env id (bodies, joints, roots).
pub fn despawn_env(
    commands: &mut Commands,
    env_id: EnvId,
    roots: &Query<(Entity, &EnvRoot)>,
    bodies: &Query<(Entity, &SimBody)>,
    joints: &Query<(Entity, &SimJoint)>,
) {
    for (entity, joint) in joints.iter() {
        if joint.env_id == env_id {
            commands.entity(entity).despawn();
        }
    }
    for (entity, body) in bodies.iter() {
        if body.env_id == env_id {
            commands.entity(entity).despawn();
        }
    }
    for (entity, root) in roots.iter() {
        if root.env_id == env_id {
            commands.entity(entity).despawn();
        }
    }
}

/// Convenience: despawn then invoke `respawn` closure.
pub fn reset_env<F>(
    commands: &mut Commands,
    env_id: EnvId,
    roots: &Query<(Entity, &EnvRoot)>,
    bodies: &Query<(Entity, &SimBody)>,
    joints: &Query<(Entity, &SimJoint)>,
    respawn: F,
) where
    F: FnOnce(&mut Commands, EnvId),
{
    despawn_env(commands, env_id, roots, bodies, joints);
    respawn(commands, env_id);
}

/// Mesh-friendly sizes for debug rendering of a body shape.
pub fn debug_mesh_for_shape(shape: BodyShape) -> DebugMeshKind {
    match shape {
        BodyShape::Capsule { radius, length } => DebugMeshKind::Capsule { radius, length },
        BodyShape::Cylinder { radius, height } => DebugMeshKind::Cylinder { radius, height },
        BodyShape::Cuboid { half_extents } => DebugMeshKind::Cuboid { half_extents },
        BodyShape::Sphere { radius } => DebugMeshKind::Sphere { radius },
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DebugMeshKind {
    Capsule { radius: f32, length: f32 },
    Cylinder { radius: f32, height: f32 },
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}
