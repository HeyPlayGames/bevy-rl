//! Visual morphology studio: load/save [`CreatureDesc`] as RON and edit body
//! poses / dimensions with Bevy's transform gizmo. Play runs a physics preview
//! that holds each revolute's `default_angle` (no policy); Reset returns to edit
//! mode.
//!
//! Creature-agnostic: pass an already-loaded [`CreatureDesc`] and save path via
//! [`MorphologyStudioConfig`].

mod document;
mod edit;
mod hints;
mod scene;
mod selection;
mod simulate;
mod state;
mod ui;

use std::path::PathBuf;

use avian3d::prelude::*;
use bevy::{
    gizmos::transform_gizmo::{
        TransformGizmoPlugin, TransformGizmoState, TransformGizmoSystems,
    },
    picking::hover::Hovered,
    prelude::*,
    ui::UiSystems,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, PanOrbitCameraSystemSet};
use sim_core::{apply_joint_targets, CreatureDesc};

use edit::{gizmo_mode_keys, sync_body_selection_materials, sync_focused_body_to_document};
use scene::{setup_scene, spawn_creature_bodies};
use selection::{
    apply_studio_selection, clear_selection_on_mode_change, configure_anchor_gizmo_depth,
    draw_joint_anchors,
};
use simulate::attach_sim_debug_meshes;
use state::{
    in_edit_mode, in_sim_mode, LeftPanel, MorphologyDocument, StudioColors, StudioMeshes,
    StudioMode, StudioSelection,
};
use ui::{
    feathers_plugins, feathers_theme, setup_ui, sync_mode_ui, sync_status_label,
    sync_studio_viewport,
};

/// Configuration for [`run_morphology_studio`].
#[derive(Clone, Debug)]
pub struct MorphologyStudioConfig {
    /// Initial morphology to edit (already loaded by the caller).
    pub creature: CreatureDesc,
    /// Default path used for Save (and shown in the status bar).
    pub morph_path: PathBuf,
    pub creature_color: Color,
    pub selected_color: Color,
    pub ground_color: Color,
}

impl MorphologyStudioConfig {
    pub fn new(creature: CreatureDesc, morph_path: PathBuf) -> Self {
        Self {
            creature,
            morph_path,
            creature_color: Color::srgb(0.75, 0.55, 0.35),
            selected_color: Color::srgb(0.95, 0.75, 0.35),
            ground_color: Color::srgb(0.22, 0.23, 0.26),
        }
    }
}

/// Run the morphology studio window until closed.
pub fn run_morphology_studio(config: MorphologyStudioConfig) {
    let MorphologyStudioConfig {
        creature,
        morph_path,
        creature_color,
        selected_color,
        ground_color,
    } = config;

    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "bevy-rl morphology studio".into(),
                    ..default()
                }),
                ..default()
            }),
            MeshPickingPlugin,
            TransformGizmoPlugin,
            PhysicsPlugins::default(),
            PanOrbitCameraPlugin,
            feathers_plugins(),
        ))
        .insert_resource(ClearColor(Color::srgb(0.08, 0.09, 0.11)))
        .insert_resource(Gravity(Vec3::NEG_Y * 9.81))
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .insert_resource(feathers_theme())
        .insert_resource(MorphologyDocument {
            path: morph_path,
            creature,
            dirty: false,
            status: String::new(),
        })
        .insert_resource(StudioColors {
            creature_color,
            selected_color,
            ground_color,
        })
        .insert_resource(StudioMeshes::default())
        .insert_resource(StudioMode::Edit)
        .insert_resource(StudioSelection::None)
        .add_systems(FixedUpdate, apply_joint_targets)
        .add_systems(
            Startup,
            (
                setup_scene,
                setup_ui,
                spawn_creature_bodies,
                configure_anchor_gizmo_depth,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                gizmo_mode_keys.run_if(in_edit_mode),
                sync_status_label,
                sync_body_selection_materials.run_if(in_edit_mode),
                sync_mode_ui,
                clear_selection_on_mode_change,
                apply_studio_selection,
                draw_joint_anchors.run_if(in_edit_mode),
                attach_sim_debug_meshes.run_if(in_sim_mode),
            ),
        )
        .add_systems(
            PostUpdate,
            (
                sync_pan_orbit_blocking.before(PanOrbitCameraSystemSet),
                sync_studio_viewport.after(UiSystems::Layout),
                sync_focused_body_to_document
                    .run_if(in_edit_mode)
                    .after(TransformGizmoSystems),
            ),
        )
        .run();
}

fn sync_pan_orbit_blocking(
    panel: Query<Entity, With<LeftPanel>>,
    hovered: Query<(Entity, &Hovered)>,
    parents: Query<&ChildOf>,
    gizmo_state: Res<TransformGizmoState>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    let Ok(panel_entity) = panel.single() else {
        return;
    };
    let ui_blocking = hovered.iter().any(|(entity, hovered)| {
        if !hovered.0 {
            return false;
        }
        entity == panel_entity
            || parents
                .iter_ancestors(entity)
                .any(|ancestor| ancestor == panel_entity)
    });
    // Hover/active come from the previous frame's gizmo PostUpdate pass; disable
    // orbit before PanOrbitCamera reads mouse so gizmo drags don't also orbit.
    let gizmo_blocking = gizmo_state.active || gizmo_state.hovered_axis.is_some();
    for mut camera in &mut cameras {
        camera.enabled = !ui_blocking && !gizmo_blocking;
    }
}
