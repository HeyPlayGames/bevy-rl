use std::collections::HashMap;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_transform_interpolation::prelude::TransformInterpolation;

use crate::control::JointTargetAngle;
use crate::creature::{
    BodyDesc, BodyPoseMap, BodyShape, CreatureDesc, CreatureInstance, JointDesc, JointKind,
};
use crate::env::{
    env_creature_collision_layers, env_origin, EnvId, EnvIsolationConfig, EnvRoot, SimBody,
    SimJoint,
};

/// Stable body name for soft resets (maps a spawned entity back to [`BodyDesc::name`]).
#[derive(Component, Clone, Debug)]
pub struct CreaturePart {
    pub name: String,
}

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

/// Spawns a creature at the env world origin from absolute body poses.
///
/// Bodies are **not** parented under a hierarchy for physics safety: Avian
/// owns their transforms, and we only tag them with [`SimBody`] / [`EnvId`].
///
/// `joint_zero_poses` defines revolute angle 0 via body orientations (typically
/// [`crate::compute_zero_body_poses`]). Entity placement uses `placement_poses`,
/// which may include default-angle / noise FK.
pub fn spawn_creature(
    commands: &mut Commands,
    env_id: EnvId,
    world_origin: Vec3,
    creature: &CreatureDesc,
    placement_poses: &BodyPoseMap,
    joint_zero_poses: &BodyPoseMap,
    interpolate: bool,
) -> CreatureInstance {
    let layers = env_creature_collision_layers(env_id);

    let creature_root = commands
        .spawn((
            Name::new(format!("{}_{}", creature.name, env_id.index())),
            EnvRoot { env_id },
            Transform::from_translation(world_origin),
        ))
        .id();

    let mut bodies: HashMap<String, Entity> = HashMap::new();

    for body in &creature.bodies {
        let pose = placement_poses
            .get(&body.name)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        let entity = spawn_body(commands, env_id, world_origin, body, pose, layers, interpolate);
        bodies.insert(body.name.clone(), entity);
    }

    let mut zero_rotation_by_name: HashMap<&str, Quat> = HashMap::new();
    for (name, pose) in joint_zero_poses {
        zero_rotation_by_name.insert(name.as_str(), pose.rotation);
    }

    let mut joints: HashMap<String, Entity> = HashMap::new();
    for joint in &creature.joints {
        let entity = spawn_joint(commands, env_id, &bodies, &zero_rotation_by_name, joint);
        joints.insert(joint.name.clone(), entity);
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
    pose: Transform,
    layers: CollisionLayers,
    interpolate: bool,
) -> Entity {
    let collider = collider_from_shape(body.shape);
    let transform = Transform {
        translation: world_origin + pose.translation,
        rotation: pose.rotation,
        scale: Vec3::ONE,
    };

    let mut entity_commands = commands.spawn((
        Name::new(format!("{}_{}", body.name, env_id.index())),
        SimBody { env_id },
        CreaturePart {
            name: body.name.clone(),
        },
        RigidBody::Dynamic,
        transform,
        collider,
        ColliderDensity(body.density),
        layers,
        Friction::new(0.8),
        Restitution::new(0.05),
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
    zero_rotation_by_name: &HashMap<&str, Quat>,
    joint: &JointDesc,
) -> Entity {
    let body_a = *bodies
        .get(&joint.body_a)
        .unwrap_or_else(|| panic!("joint `{}`: missing body `{}`", joint.name, joint.body_a));
    let body_b = *bodies
        .get(&joint.body_b)
        .unwrap_or_else(|| panic!("joint `{}`: missing body `{}`", joint.name, joint.body_b));

    match &joint.kind {
        JointKind::Revolute {
            axis,
            angle_limits,
            ..
        } => {
            // Avian angle 0 is when the joint frames align. Bake the morphology
            // zero pose into frame2 so authored relative orientation (e.g. hip
            // capsule horizontal at angle 0) is angle 0.
            let rotation_a = zero_rotation_by_name
                .get(joint.body_a.as_str())
                .copied()
                .unwrap_or(Quat::IDENTITY);
            let rotation_b = zero_rotation_by_name
                .get(joint.body_b.as_str())
                .copied()
                .unwrap_or(Quat::IDENTITY);
            let basis2 = (rotation_b.inverse() * rotation_a).normalize();

            let mut revolute = RevoluteJoint::new(body_a, body_b)
                .with_local_anchor1(joint.anchor_a)
                .with_local_anchor2(joint.anchor_b)
                .with_local_basis1(Quat::IDENTITY)
                .with_local_basis2(basis2)
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

/// Cheap reset: teleport creature bodies to `poses` and zero velocities.
///
/// Keeps entities, colliders, and joints. Skips ground / non-[`CreaturePart`] bodies.
pub fn soft_reset_creature(
    commands: &mut Commands,
    env_id: EnvId,
    world_origin: Vec3,
    poses: &BodyPoseMap,
    bodies: &mut Query<(
        Entity,
        &SimBody,
        &CreaturePart,
        &mut Transform,
        &mut Position,
        &mut Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
    joint_targets: &mut Query<(&SimJoint, &mut JointTargetAngle)>,
) {
    for (
        entity,
        sim_body,
        part,
        mut transform,
        mut position,
        mut rotation,
        mut linear_velocity,
        mut angular_velocity,
    ) in bodies.iter_mut()
    {
        if sim_body.env_id != env_id {
            continue;
        }
        let Some(pose) = poses.get(part.name.as_str()) else {
            continue;
        };

        let translation = world_origin + pose.translation;
        let orientation = pose.rotation.normalize();
        transform.translation = translation;
        transform.rotation = orientation;
        position.0 = translation;
        *rotation = Rotation::from(orientation);
        linear_velocity.0 = Vec3::ZERO;
        angular_velocity.0 = Vec3::ZERO;
        commands.entity(entity).remove::<Sleeping>();
    }

    for (sim_joint, mut target) in joint_targets.iter_mut() {
        if sim_joint.env_id == env_id {
            target.0 = 0.0;
        }
    }
}
