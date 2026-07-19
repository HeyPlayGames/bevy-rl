//! Play / Reset: physics preview holding each revolute's default angle.

use avian3d::prelude::*;
use bevy::{
    gizmos::transform_gizmo::TransformGizmoFocus,
    picking::Pickable,
    prelude::*,
    ui_widgets::Activate,
};
use sim_core::{
    attach_default_revolute_actuation, compute_default_body_poses, compute_zero_body_poses,
    despawn_env, env_world_collision_layers, spawn_creature, CreaturePart, EnvRoot, SimBody,
    SimJoint,
};

use crate::edit::bake_focused_scale_into_document;
use crate::scene::{mesh_from_shape, spawn_body_entity};
use crate::state::{
    MorphologyDocument, SimDebugMeshAttached, StudioBody, StudioEditGround, StudioMeshes,
    StudioMode, StudioSelection, GROUND_COLLIDER_THICKNESS, GROUND_SIZE, STUDIO_ENV_ID,
};

pub(crate) fn attach_sim_debug_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    document: Res<MorphologyDocument>,
    studio_meshes: Res<StudioMeshes>,
    bodies: Query<(Entity, &CreaturePart), (With<SimBody>, Without<SimDebugMeshAttached>)>,
) {
    for (entity, part) in &bodies {
        let Some(body) = document
            .creature
            .bodies
            .iter()
            .find(|body| body.name == part.name)
        else {
            continue;
        };
        commands.entity(entity).insert((
            Mesh3d(meshes.add(mesh_from_shape(body.shape))),
            MeshMaterial3d(studio_meshes.body_material.clone()),
            Pickable::IGNORE,
            SimDebugMeshAttached,
        ));
    }
}

pub(crate) fn on_play_activated(
    _activate: On<Activate>,
    mut commands: Commands,
    mut mode: ResMut<StudioMode>,
    mut document: ResMut<MorphologyDocument>,
    mut selection: ResMut<StudioSelection>,
    mut focused: Query<(Entity, &StudioBody, &mut Transform), With<TransformGizmoFocus>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_query: Query<&mut Mesh3d, With<StudioBody>>,
    edit_bodies: Query<Entity, With<StudioBody>>,
    focused_entities: Query<Entity, With<TransformGizmoFocus>>,
    edit_ground: Query<Entity, With<StudioEditGround>>,
) {
    if *mode != StudioMode::Edit {
        return;
    }
    bake_focused_scale_into_document(&mut document, &mut focused, &mut meshes, &mut mesh_query);
    *selection = StudioSelection::None;
    enter_simulation(
        &mut commands,
        &mut mode,
        &mut document,
        &edit_bodies,
        &focused_entities,
        &edit_ground,
    );
}

pub(crate) fn on_reset_activated(
    _activate: On<Activate>,
    mut commands: Commands,
    mut mode: ResMut<StudioMode>,
    mut document: ResMut<MorphologyDocument>,
    mut meshes: ResMut<Assets<Mesh>>,
    studio_meshes: Res<StudioMeshes>,
    roots: Query<(Entity, &EnvRoot)>,
    bodies: Query<(Entity, &SimBody)>,
    joints: Query<(Entity, &SimJoint)>,
    edit_ground: Query<Entity, With<StudioEditGround>>,
) {
    leave_simulation(
        &mut commands,
        &mut mode,
        &mut document,
        &mut meshes,
        &studio_meshes,
        &roots,
        &bodies,
        &joints,
        &edit_ground,
    );
}

fn clear_gizmo_focus(commands: &mut Commands, focused: &Query<Entity, With<TransformGizmoFocus>>) {
    for entity in focused.iter() {
        commands.entity(entity).remove::<TransformGizmoFocus>();
    }
}

fn enter_simulation(
    commands: &mut Commands,
    mode: &mut StudioMode,
    document: &mut MorphologyDocument,
    edit_bodies: &Query<Entity, With<StudioBody>>,
    focused: &Query<Entity, With<TransformGizmoFocus>>,
    edit_ground: &Query<Entity, With<StudioEditGround>>,
) {
    if *mode != StudioMode::Edit {
        return;
    }

    clear_gizmo_focus(commands, focused);
    for entity in edit_bodies.iter() {
        commands.entity(entity).despawn();
    }

    // Same scene: enable physics on the existing ground, spawn creature in place.
    for ground_entity in edit_ground.iter() {
        enable_edit_ground_physics(commands, ground_entity);
    }
    let placement = compute_default_body_poses(&document.creature);
    let joint_zero = compute_zero_body_poses(&document.creature);
    let instance = spawn_creature(
        commands,
        STUDIO_ENV_ID,
        Vec3::ZERO,
        &document.creature,
        &placement,
        &joint_zero,
        true,
    );
    attach_default_revolute_actuation(commands, &instance, &document.creature);

    *mode = StudioMode::Simulating;
    document.status = "simulating (default joint targets)".to_string();
    info!("entered physics preview");
}

fn leave_simulation(
    commands: &mut Commands,
    mode: &mut StudioMode,
    document: &mut MorphologyDocument,
    meshes: &mut Assets<Mesh>,
    studio_meshes: &StudioMeshes,
    roots: &Query<(Entity, &EnvRoot)>,
    bodies: &Query<(Entity, &SimBody)>,
    joints: &Query<(Entity, &SimJoint)>,
    edit_ground: &Query<Entity, With<StudioEditGround>>,
) {
    if *mode != StudioMode::Simulating {
        return;
    }

    despawn_env(commands, STUDIO_ENV_ID, roots, bodies, joints);
    for ground_entity in edit_ground.iter() {
        disable_edit_ground_physics(commands, ground_entity);
    }
    let poses = compute_zero_body_poses(&document.creature);
    for body in &document.creature.bodies {
        let pose = poses
            .get(&body.name)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        spawn_body_entity(commands, meshes, studio_meshes, body, pose);
    }

    *mode = StudioMode::Edit;
    document.status = "edit mode".to_string();
    info!("returned to edit mode");
}

fn enable_edit_ground_physics(commands: &mut Commands, ground_entity: Entity) {
    commands.entity(ground_entity).insert((
        RigidBody::Static,
        Collider::cuboid(GROUND_SIZE, GROUND_COLLIDER_THICKNESS, GROUND_SIZE),
        env_world_collision_layers(STUDIO_ENV_ID),
        Friction::new(0.9),
    ));
}

fn disable_edit_ground_physics(commands: &mut Commands, ground_entity: Entity) {
    commands
        .entity(ground_entity)
        .remove::<(RigidBody, Collider, CollisionLayers, Friction)>();
}
