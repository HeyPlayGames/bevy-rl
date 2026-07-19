//! Gizmo editing: sync poses/dimensions from focused bodies into the document.

use bevy::{
    gizmos::transform_gizmo::{
        TransformGizmoFocus, TransformGizmoMode, TransformGizmoSettings, TransformGizmoSpace,
        TransformGizmoState,
    },
    prelude::*,
};
use sim_core::{compute_zero_body_poses, set_body_pose_at_zero, BodyShape};

use crate::document::save_document;
use crate::hints::instructions_text;
use crate::scene::mesh_from_shape;
use crate::state::{
    InstructionsText, MorphologyDocument, StudioBody, StudioMeshes, StudioMode, StudioSelection,
};

pub(crate) fn gizmo_mode_keys(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<TransformGizmoSettings>,
    mut document: ResMut<MorphologyDocument>,
    mut instructions: Query<&mut Text, With<InstructionsText>>,
    mut focused: Query<(Entity, &StudioBody, &mut Transform), With<TransformGizmoFocus>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_query: Query<&mut Mesh3d, With<StudioBody>>,
    mode: Res<StudioMode>,
) {
    let mut mode_changed = false;
    if keyboard.just_pressed(KeyCode::Digit1) {
        settings.mode = TransformGizmoMode::Translate;
        mode_changed = true;
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        settings.mode = TransformGizmoMode::Rotate;
        mode_changed = true;
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        settings.mode = TransformGizmoMode::Scale;
        mode_changed = true;
    }
    if keyboard.just_pressed(KeyCode::KeyX) {
        settings.space = match settings.space {
            TransformGizmoSpace::World => TransformGizmoSpace::Local,
            TransformGizmoSpace::Local => TransformGizmoSpace::World,
        };
        mode_changed = true;
    }
    if keyboard.just_pressed(KeyCode::KeyS)
        && (keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight))
    {
        bake_focused_scale_into_document(&mut document, &mut focused, &mut meshes, &mut mesh_query);
        save_document(&mut document);
    }

    if mode_changed {
        bake_focused_scale_into_document(&mut document, &mut focused, &mut meshes, &mut mesh_query);
        for mut text in &mut instructions {
            *text = Text::new(instructions_text(*mode, settings.mode, settings.space));
        }
    }
}

pub(crate) fn sync_body_selection_materials(
    studio_meshes: Res<StudioMeshes>,
    selection: Res<StudioSelection>,
    document: Res<MorphologyDocument>,
    focused: Query<Entity, With<TransformGizmoFocus>>,
    mut bodies: Query<(Entity, &StudioBody, &mut MeshMaterial3d<StandardMaterial>)>,
) {
    let focused_entity = focused.iter().next();
    let joint_bodies = match selection.as_ref() {
        StudioSelection::Joint(name) => document
            .creature
            .joints
            .iter()
            .find(|joint| joint.name == *name)
            .map(|joint| (joint.body_a.as_str(), joint.body_b.as_str())),
        _ => None,
    };

    for (entity, studio_body, mut material) in &mut bodies {
        let wanted = if Some(entity) == focused_entity {
            studio_meshes.selected_material.clone()
        } else if let Some((body_a, body_b)) = joint_bodies {
            if studio_body.name == body_a {
                studio_meshes.joint_body_a_material.clone()
            } else if studio_body.name == body_b {
                studio_meshes.joint_body_b_material.clone()
            } else {
                studio_meshes.body_material.clone()
            }
        } else {
            studio_meshes.body_material.clone()
        };
        if material.0 != wanted {
            material.0 = wanted;
        }
    }
}

pub(crate) fn sync_focused_body_to_document(
    gizmo_state: Res<TransformGizmoState>,
    mut was_active: Local<bool>,
    mut document: ResMut<MorphologyDocument>,
    mut focused: Query<(Entity, &StudioBody, &mut Transform), With<TransformGizmoFocus>>,
    mut other_bodies: Query<(&StudioBody, &mut Transform), Without<TransformGizmoFocus>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_query: Query<&mut Mesh3d, With<StudioBody>>,
) {
    let drag_ended = *was_active && !gizmo_state.active;
    *was_active = gizmo_state.active;

    let Ok((entity, studio_body, mut transform)) = focused.single_mut() else {
        return;
    };

    let body_name = studio_body.name.clone();
    let Some(body_index) = document
        .creature
        .bodies
        .iter()
        .position(|body| body.name == body_name)
    else {
        return;
    };

    let authored_pose = Transform {
        translation: transform.translation,
        rotation: transform.rotation.normalize(),
        scale: Vec3::ONE,
    };
    let current_poses = compute_zero_body_poses(&document.creature);
    let pose_changed = current_poses
        .get(&body_name)
        .is_none_or(|pose| {
            pose.translation != authored_pose.translation || pose.rotation != authored_pose.rotation
        });
    if pose_changed {
        set_body_pose_at_zero(&mut document.creature, &body_name, authored_pose);
        let poses = compute_zero_body_poses(&document.creature);
        // Keep the gizmo-owned transform while dragging; snap after release.
        if !gizmo_state.active {
            if let Some(pose) = poses.get(&body_name) {
                transform.translation = pose.translation;
                transform.rotation = pose.rotation;
            }
        }
        for (other_body, mut other_transform) in &mut other_bodies {
            if let Some(pose) = poses.get(&other_body.name) {
                other_transform.translation = pose.translation;
                other_transform.rotation = pose.rotation;
            }
        }
        document.dirty = true;
        document.status.clear();
    }

    if (drag_ended || !gizmo_state.active) && transform.scale != Vec3::ONE {
        bake_scale_into_shape(
            &mut document.creature.bodies[body_index].shape,
            transform.scale,
        );
        let shape = document.creature.bodies[body_index].shape;
        transform.scale = Vec3::ONE;
        if let Ok(mut mesh3d) = mesh_query.get_mut(entity) {
            mesh3d.0 = meshes.add(mesh_from_shape(shape));
        }
        document.dirty = true;
        document.status.clear();
    }
}

pub(crate) fn bake_focused_scale_into_document(
    document: &mut MorphologyDocument,
    focused: &mut Query<(Entity, &StudioBody, &mut Transform), With<TransformGizmoFocus>>,
    meshes: &mut Assets<Mesh>,
    mesh_query: &mut Query<&mut Mesh3d, With<StudioBody>>,
) {
    let Ok((entity, studio_body, mut transform)) = focused.single_mut() else {
        return;
    };
    if transform.scale == Vec3::ONE {
        return;
    }
    let Some(body) = document
        .creature
        .bodies
        .iter_mut()
        .find(|body| body.name == studio_body.name)
    else {
        return;
    };
    bake_scale_into_shape(&mut body.shape, transform.scale);
    transform.scale = Vec3::ONE;
    if let Ok(mut mesh3d) = mesh_query.get_mut(entity) {
        mesh3d.0 = meshes.add(mesh_from_shape(body.shape));
    }
    document.dirty = true;
    document.status.clear();
}

fn bake_scale_into_shape(shape: &mut BodyShape, scale: Vec3) {
    let scale = scale.abs();
    match shape {
        BodyShape::Capsule { radius, length } => {
            *radius = (*radius * (scale.x + scale.z) * 0.5).max(0.001);
            *length = (*length * scale.y).max(0.001);
        }
        BodyShape::Cylinder { radius, height } => {
            *radius = (*radius * (scale.x + scale.z) * 0.5).max(0.001);
            *height = (*height * scale.y).max(0.001);
        }
        BodyShape::Cuboid { half_extents } => {
            half_extents.x = (half_extents.x * scale.x).max(0.001);
            half_extents.y = (half_extents.y * scale.y).max(0.001);
            half_extents.z = (half_extents.z * scale.z).max(0.001);
        }
        BodyShape::Sphere { radius } => {
            let mean = (scale.x + scale.y + scale.z) / 3.0;
            *radius = (*radius * mean).max(0.001);
        }
    }
}
