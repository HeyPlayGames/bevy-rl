//! Shared studio resources, components, and constants.

use std::path::PathBuf;

use bevy::prelude::*;
use sim_core::{CreatureDesc, EnvId};

pub(crate) const LEFT_PANEL_WIDTH: f32 = 340.0;
pub(crate) const STUDIO_ENV_ID: EnvId = EnvId::new(0);
/// Studio ground: a cuboid floor with top surface at y = 0.
pub(crate) const GROUND_SIZE: f32 = 12.0;
pub(crate) const GROUND_COLLIDER_THICKNESS: f32 = 0.5;
pub(crate) const ANCHOR_SPHERE_RADIUS: f32 = 0.045;
pub(crate) const ANCHOR_A_COLOR: Color = Color::srgb(0.95, 0.25, 0.22);
pub(crate) const ANCHOR_B_COLOR: Color = Color::srgb(0.22, 0.45, 0.95);

#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum StudioMode {
    #[default]
    Edit,
    Simulating,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum StudioSelection {
    #[default]
    None,
    Body(String),
    Joint(String),
}

#[derive(Resource)]
pub(crate) struct MorphologyDocument {
    pub path: PathBuf,
    pub creature: CreatureDesc,
    pub dirty: bool,
    pub status: String,
}

#[derive(Resource, Clone, Debug)]
pub(crate) struct StudioColors {
    pub creature_color: Color,
    pub selected_color: Color,
    pub ground_color: Color,
}

#[derive(Resource, Default)]
pub(crate) struct StudioMeshes {
    pub body_material: Handle<StandardMaterial>,
    pub selected_material: Handle<StandardMaterial>,
    pub ground_material: Handle<StandardMaterial>,
    pub joint_body_a_material: Handle<StandardMaterial>,
    pub joint_body_b_material: Handle<StandardMaterial>,
}

#[derive(Component)]
pub(crate) struct StudioBody {
    pub name: String,
}

#[derive(Component)]
pub(crate) struct StudioEditGround;

#[derive(Component, Clone, Default)]
pub(crate) struct SaveButton;

#[derive(Component, Clone, Default)]
pub(crate) struct LoadButton;

#[derive(Component, Clone, Default)]
pub(crate) struct PlayButton;

#[derive(Component, Clone, Default)]
pub(crate) struct ResetButton;

#[derive(Component, Clone, Default)]
pub(crate) struct StatusLabel;

#[derive(Component, Clone, Default)]
pub(crate) struct InstructionsText;

#[derive(Component, Clone, Default)]
pub(crate) struct LeftPanel;

#[derive(Component, Clone, Default)]
pub(crate) struct BodyListView;

#[derive(Component, Clone, Default)]
pub(crate) struct JointListView;

#[derive(Component, Clone, Default)]
pub(crate) struct BodyListItem {
    pub name: String,
}

#[derive(Component, Clone, Default)]
pub(crate) struct JointListItem {
    pub name: String,
}

/// Marker for the right-hand UI region that owns the 3D camera viewport.
#[derive(Component, Clone, Default)]
pub(crate) struct StudioViewport;

/// Marker for the world [`Camera3d`] (as opposed to the UI camera).
#[derive(Component)]
pub(crate) struct StudioWorldCamera;

#[derive(Component)]
pub(crate) struct SimDebugMeshAttached;

pub(crate) fn in_edit_mode(mode: Res<StudioMode>) -> bool {
    *mode == StudioMode::Edit
}

pub(crate) fn in_sim_mode(mode: Res<StudioMode>) -> bool {
    *mode == StudioMode::Simulating
}
