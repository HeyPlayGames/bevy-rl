//! Left control panel and right viewport layout (Bevy Feathers + BSN).

use bevy::{
    camera::Viewport,
    feathers::{
        containers::{subpane, subpane_body, subpane_header},
        controls::{FeathersButton, FeathersListRow, FeathersListView, FeathersScrollbar},
        dark_theme::create_dark_theme,
        display::label,
        theme::{ThemeBackgroundColor, ThemeBorderColor, ThemedText, UiTheme},
        tokens, FeathersPlugins,
    },
    gizmos::transform_gizmo::{
        TransformGizmoFocus, TransformGizmoMode, TransformGizmoSettings, TransformGizmoSpace,
    },
    input_focus::tab_navigation::TabGroup,
    prelude::*,
    scene::SceneList,
    ui::{ComputedNode, Overflow, PositionType, UiGlobalTransform},
    ui_widgets::{listbox_update_selection, Activate, ControlOrientation, ListItem, ScrollArea},
};
use sim_core::{compute_zero_body_poses, load_ron_config, BodyDesc, CreatureDesc, JointDesc};

use crate::document::save_document;
use crate::edit::bake_focused_scale_into_document;
use crate::hints::instructions_text;
use crate::scene::spawn_body_entity;
use crate::selection::{
    on_body_list_selected, on_joint_list_selected, rebuild_hierarchy_lists,
};
use crate::simulate::{on_play_activated, on_reset_activated};
use crate::state::{
    BodyListItem, BodyListView, InstructionsText, JointListItem, JointListView, LeftPanel,
    LoadButton, MorphologyDocument, PlayButton, ResetButton, SaveButton, StatusLabel, StudioBody,
    StudioMeshes, StudioMode, StudioSelection, StudioViewport, StudioWorldCamera, LEFT_PANEL_WIDTH,
};

pub(crate) fn feathers_plugins() -> FeathersPlugins {
    FeathersPlugins
}

pub(crate) fn feathers_theme() -> UiTheme {
    UiTheme(create_dark_theme())
}

pub(crate) fn setup_ui(mut commands: Commands, document: Res<MorphologyDocument>) {
    // Dedicated UI camera so the world camera can use a restricted viewport without
    // shrinking the Bevy UI layout to the 3D region.
    // Spawned imperatively: `IsDefaultUiCamera` is not BSN-template-friendly.
    commands.spawn((
        Camera2d,
        Camera {
            // Transform gizmo overlay camera uses order 1.
            order: 2,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        IsDefaultUiCamera,
    ));
    commands.spawn_scene(studio_root(&document));
}

fn studio_root(document: &MorphologyDocument) -> impl Scene {
    let body_rows = body_list_rows(&document.creature.bodies);
    let joint_rows = joint_list_rows(&document.creature.joints);
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            flex_direction: FlexDirection::Row,
        }
        Pickable::IGNORE
        TabGroup
        Children [
            left_panel(body_rows, joint_rows),
            (
                Node {
                    flex_grow: 1.0,
                    height: percent(100),
                }
                StudioViewport
                Pickable::IGNORE
            ),
        ]
    }
}

fn left_panel(body_rows: Box<dyn SceneList>, joint_rows: Box<dyn SceneList>) -> impl Scene {
    // Outer frame + inner ScrollArea + FeathersScrollbar mirrors FeathersListView:
    // the panel clips to the window height and scrolls when instructions / lists / buttons overflow.
    bsn! {
        Node {
            width: px(LEFT_PANEL_WIDTH),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            border: UiRect::right(px(1)),
            padding: UiRect {
                right: px(10)
            }
        }
        LeftPanel
        ThemeBackgroundColor(tokens::WINDOW_BG)
        ThemeBorderColor(tokens::SUBPANE_BODY_BORDER)
        Children [
            (
                #panel_scroll
                Node {
                    flex_grow: 1.0,
                    width: percent(100),
                    height: percent(100),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(px(12)),
                    row_gap: px(10),
                    overflow: Overflow::scroll_y(),
                }
                ScrollArea
                Children [
                    (
                        Node {
                            flex_shrink: 0.0,
                        }
                        Text({instructions_text(
                            StudioMode::Edit,
                            TransformGizmoMode::Translate,
                            TransformGizmoSpace::World,
                        )})
                        ThemedText
                        InstructionsText
                    ),
                    (
                        {subpane()}
                        Node {
                            flex_shrink: 0.0,
                            min_height: px(120),
                        }
                        Children [
                            (
                                {subpane_header()}
                                Children [({label("Bodies")})]
                            ),
                            (
                                {subpane_body()}
                                Node {
                                    min_height: px(80),
                                }
                                Children [(
                                    @FeathersListView {
                                        @rows: {body_rows}
                                    }
                                    BodyListView
                                    Node {
                                        height: px(220),
                                        min_height: px(80),
                                    }
                                    on(listbox_update_selection)
                                    on(on_body_list_selected)
                                )]
                            ),
                        ]
                    ),
                    (
                        {subpane()}
                        Node {
                            flex_shrink: 0.0,
                            min_height: px(120),
                        }
                        Children [
                            (
                                {subpane_header()}
                                Children [({label("Joints")})]
                            ),
                            (
                                {subpane_body()}
                                Node {
                                    min_height: px(80),
                                }
                                Children [(
                                    @FeathersListView {
                                        @rows: {joint_rows}
                                    }
                                    JointListView
                                    Node {
                                        height: px(220),
                                        min_height: px(80),
                                    }
                                    on(listbox_update_selection)
                                    on(on_joint_list_selected)
                                )]
                            ),
                        ]
                    ),
                    (
                        Node {
                            flex_direction: FlexDirection::Column,
                            row_gap: px(6),
                            width: percent(100),
                            flex_shrink: 0.0,
                        }
                        Children [
                            (
                                @FeathersButton {
                                    @caption: bsn! { Text("Play") ThemedText }
                                }
                                PlayButton
                                Node { width: percent(100) }
                                on(on_play_activated)
                            ),
                            (
                                @FeathersButton {
                                    @caption: bsn! { Text("Reset") ThemedText }
                                }
                                ResetButton
                                Node { width: percent(100) }
                                on(on_reset_activated)
                            ),
                            (
                                @FeathersButton {
                                    @caption: bsn! { Text("Save") ThemedText }
                                }
                                SaveButton
                                Node { width: percent(100) }
                                on(on_save_activated)
                            ),
                            (
                                @FeathersButton {
                                    @caption: bsn! { Text("Load…") ThemedText }
                                }
                                LoadButton
                                Node { width: percent(100) }
                                on(on_load_activated)
                            ),
                        ]
                    ),
                    (
                        Node {
                            flex_shrink: 0.0,
                        }
                        Text("")
                        ThemedText
                        StatusLabel
                    ),
                ]
            ),
            (
                @FeathersScrollbar {
                    @target: #panel_scroll,
                    @orientation: {ControlOrientation::Vertical}
                }
                Node {
                    position_type: PositionType::Absolute,
                    right: px(0),
                    top: px(0),
                    bottom: px(0),
                    width: px(6),
                }
            ),
        ]
    }
}

pub(crate) fn body_list_rows(bodies: &[BodyDesc]) -> Box<dyn SceneList> {
    let rows: Vec<Box<dyn SceneList>> = bodies
        .iter()
        .map(|body| {
            let name = body.name.clone();
            Box::new(bsn! {
                @FeathersListRow
                BodyListItem { name: {name.clone()} }
                Children [(Text({name}) ThemedText)]
            }) as Box<dyn SceneList>
        })
        .collect();
    Box::new(rows)
}

pub(crate) fn joint_list_rows(joints: &[JointDesc]) -> Box<dyn SceneList> {
    let rows: Vec<Box<dyn SceneList>> = joints
        .iter()
        .map(|joint| {
            let name = joint.name.clone();
            Box::new(bsn! {
                @FeathersListRow
                JointListItem { name: {name.clone()} }
                Children [(Text({name}) ThemedText)]
            }) as Box<dyn SceneList>
        })
        .collect();
    Box::new(rows)
}

/// Keeps the world camera's render viewport aligned with the right-hand UI region.
/// Also mirrors the viewport onto other [`Camera3d`]s (the transform-gizmo overlay)
/// so gizmo meshes stay registered with the same screen rectangle.
pub(crate) fn sync_studio_viewport(
    viewport_nodes: Query<(&ComputedNode, &UiGlobalTransform), With<StudioViewport>>,
    mut world_cameras: Query<&mut Camera, With<StudioWorldCamera>>,
    mut overlay_cameras: Query<&mut Camera, (With<Camera3d>, Without<StudioWorldCamera>)>,
) {
    let Ok((computed, transform)) = viewport_nodes.single() else {
        return;
    };
    if computed.is_empty() {
        return;
    }

    let size = computed.size();
    let (_, _, translation) = transform.to_scale_angle_translation();
    let minimum = translation - size * 0.5;
    let physical_position = UVec2::new(
        minimum.x.max(0.0).round() as u32,
        minimum.y.max(0.0).round() as u32,
    );
    let physical_size = UVec2::new(size.x.round() as u32, size.y.round() as u32).max(UVec2::ONE);
    let viewport = Viewport {
        physical_position,
        physical_size,
        ..default()
    };

    for mut camera in &mut world_cameras {
        camera.viewport = Some(viewport.clone());
    }
    for mut camera in &mut overlay_cameras {
        camera.viewport = Some(viewport.clone());
    }
}

pub(crate) fn sync_status_label(
    document: Res<MorphologyDocument>,
    mode: Res<StudioMode>,
    mut labels: Query<&mut Text, With<StatusLabel>>,
) {
    if !document.is_changed() && !mode.is_changed() {
        return;
    }
    let dirty_marker = if document.dirty { " *" } else { "" };
    let path_name = document
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("morphology.ron");
    let mode_label = match *mode {
        StudioMode::Edit => "edit",
        StudioMode::Simulating => "sim",
    };
    let status = if document.status.is_empty() {
        format!("{path_name}{dirty_marker} [{mode_label}]")
    } else {
        format!("{path_name}{dirty_marker} [{mode_label}] — {}", document.status)
    };
    for mut text in &mut labels {
        *text = Text::new(status.clone());
    }
}

pub(crate) fn sync_mode_ui(
    mode: Res<StudioMode>,
    settings: Res<TransformGizmoSettings>,
    mut instructions: Query<&mut Text, With<InstructionsText>>,
) {
    if !mode.is_changed() {
        return;
    }
    for mut text in &mut instructions {
        *text = Text::new(instructions_text(*mode, settings.mode, settings.space));
    }
}

fn on_save_activated(
    _activate: On<Activate>,
    mode: Res<StudioMode>,
    mut document: ResMut<MorphologyDocument>,
    mut focused: Query<(Entity, &StudioBody, &mut Transform), With<TransformGizmoFocus>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_query: Query<&mut Mesh3d, With<StudioBody>>,
) {
    if *mode != StudioMode::Edit {
        document.status = "save only in edit mode".to_string();
        return;
    }
    bake_focused_scale_into_document(&mut document, &mut focused, &mut meshes, &mut mesh_query);
    save_document(&mut document);
}

fn on_load_activated(
    _activate: On<Activate>,
    mode: Res<StudioMode>,
    mut commands: Commands,
    mut document: ResMut<MorphologyDocument>,
    mut selection: ResMut<StudioSelection>,
    mut meshes: ResMut<Assets<Mesh>>,
    studio_meshes: Res<StudioMeshes>,
    existing_bodies: Query<Entity, With<StudioBody>>,
    focused: Query<Entity, With<TransformGizmoFocus>>,
    body_lists: Query<Entity, With<BodyListView>>,
    joint_lists: Query<Entity, With<JointListView>>,
    children: Query<&Children>,
    list_items: Query<(), With<ListItem>>,
    scroll_areas: Query<(), With<ScrollArea>>,
) {
    if *mode != StudioMode::Edit {
        document.status = "load only in edit mode".to_string();
        return;
    }

    let mut dialog = rfd::FileDialog::new()
        .add_filter("Creature morphology", &["ron"])
        .set_title("Load morphology RON");
    if let Some(parent) = document.path.parent() {
        dialog = dialog.set_directory(parent);
    }
    let Some(path) = dialog.pick_file() else {
        return;
    };

    match load_ron_config::<CreatureDesc>(&path) {
        Ok(creature) => {
            for entity in &focused {
                commands.entity(entity).remove::<TransformGizmoFocus>();
            }
            for entity in &existing_bodies {
                commands.entity(entity).despawn();
            }
            document.path = path;
            document.creature = creature;
            document.dirty = false;
            document.status = "loaded".to_string();
            *selection = StudioSelection::None;
            let poses = compute_zero_body_poses(&document.creature);
            for body in &document.creature.bodies {
                let pose = poses
                    .get(&body.name)
                    .copied()
                    .unwrap_or(Transform::IDENTITY);
                spawn_body_entity(&mut commands, &mut meshes, &studio_meshes, body, pose);
            }
            rebuild_hierarchy_lists(
                &mut commands,
                &document,
                &body_lists,
                &joint_lists,
                &children,
                &list_items,
                &scroll_areas,
            );
            info!("loaded morphology from {}", document.path.display());
        }
        Err(error) => {
            document.status = format!("load failed: {error}");
            error!("failed to load morphology: {error}");
        }
    }
}
