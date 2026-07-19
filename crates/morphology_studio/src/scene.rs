//! Scene setup: ground, lights, camera, and edit-mode body meshes.

use bevy::{
    gizmos::transform_gizmo::TransformGizmoCamera,
    picking::Pickable,
    prelude::*,
};
use bevy_panorbit_camera::PanOrbitCamera;
use sim_core::{compute_zero_body_poses, BodyDesc, BodyShape};

use crate::selection::on_click_select_body;
use crate::state::{
    MorphologyDocument, StudioBody, StudioColors, StudioEditGround, StudioMeshes, StudioWorldCamera,
    ANCHOR_A_COLOR, ANCHOR_B_COLOR, GROUND_COLLIDER_THICKNESS, GROUND_SIZE,
};

pub(crate) fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut studio_meshes: ResMut<StudioMeshes>,
    studio_colors: Res<StudioColors>,
) {
    studio_meshes.body_material = materials.add(StandardMaterial {
        base_color: studio_colors.creature_color,
        perceptual_roughness: 0.85,
        ..default()
    });
    studio_meshes.selected_material = materials.add(StandardMaterial {
        base_color: studio_colors.selected_color,
        perceptual_roughness: 0.7,
        ..default()
    });
    studio_meshes.joint_body_a_material = materials.add(StandardMaterial {
        base_color: tint_color(studio_colors.creature_color, ANCHOR_A_COLOR, 0.5),
        perceptual_roughness: 0.75,
        ..default()
    });
    studio_meshes.joint_body_b_material = materials.add(StandardMaterial {
        base_color: tint_color(studio_colors.creature_color, ANCHOR_B_COLOR, 0.5),
        perceptual_roughness: 0.75,
        ..default()
    });
    studio_meshes.ground_material = materials.add(StandardMaterial {
        base_color: studio_colors.ground_color,
        perceptual_roughness: 0.95,
        ..default()
    });

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(Vec3::new(
            GROUND_SIZE,
            GROUND_COLLIDER_THICKNESS,
            GROUND_SIZE,
        )))),
        MeshMaterial3d(studio_meshes.ground_material.clone()),
        // Center below the floor so the top face sits at y = 0.
        Transform::from_xyz(0.0, -GROUND_COLLIDER_THICKNESS * 0.5, 0.0),
        Pickable::IGNORE,
        StudioEditGround,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.5, 0.0)),
    ));

    // Leave yaw/pitch/radius unset so PanOrbitCamera derives them from Transform + focus
    // (its pitch convention is not Bevy looking_at euler — negative pitch puts the eye under the ground).
    let focus = Vec3::new(0.0, 0.55, 0.0);
    let eye = focus + Vec3::new(3.2, 2.4, 3.2);
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            ..default()
        },
        TransformGizmoCamera,
        StudioWorldCamera,
        Transform::from_translation(eye),
        PanOrbitCamera {
            focus,
            target_focus: focus,
            ..default()
        },
    ));
}

fn tint_color(base: Color, tint: Color, amount: f32) -> Color {
    let base = base.to_srgba();
    let tint = tint.to_srgba();
    Color::srgb(
        base.red + (tint.red - base.red) * amount,
        base.green + (tint.green - base.green) * amount,
        base.blue + (tint.blue - base.blue) * amount,
    )
}

pub(crate) fn spawn_creature_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    document: Res<MorphologyDocument>,
    studio_meshes: Res<StudioMeshes>,
) {
    let poses = compute_zero_body_poses(&document.creature);
    for body in &document.creature.bodies {
        let pose = poses
            .get(&body.name)
            .copied()
            .unwrap_or(Transform::IDENTITY);
        spawn_body_entity(&mut commands, &mut meshes, &studio_meshes, body, pose);
    }
}

pub(crate) fn spawn_body_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    studio_meshes: &StudioMeshes,
    body: &BodyDesc,
    pose: Transform,
) {
    commands
        .spawn((
            Mesh3d(meshes.add(mesh_from_shape(body.shape))),
            MeshMaterial3d(studio_meshes.body_material.clone()),
            pose.with_scale(Vec3::ONE),
            StudioBody {
                name: body.name.clone(),
            },
        ))
        .observe(on_click_select_body);
}

pub(crate) fn mesh_from_shape(shape: BodyShape) -> Mesh {
    match shape {
        BodyShape::Capsule { radius, length } => Capsule3d::new(radius, length).into(),
        BodyShape::Cylinder { radius, height } => Cylinder::new(radius, height).into(),
        BodyShape::Cuboid { half_extents } => Cuboid::from_size(half_extents * 2.0).into(),
        BodyShape::Sphere { radius } => Sphere::new(radius).into(),
    }
}
