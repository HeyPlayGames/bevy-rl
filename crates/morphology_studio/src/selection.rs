//! Body / joint selection from the hierarchy lists and the 3D viewport.

use bevy::{
    gizmos::transform_gizmo::TransformGizmoFocus,
    picking::pointer::PointerButton,
    prelude::*,
    scene::SceneList,
    ui::Selected,
    ui_widgets::{ListBox, ListItem, ScrollArea, ValueChange},
};
use sim_core::{compute_zero_body_poses, JointDesc};

use crate::state::{
    BodyListItem, JointListItem, MorphologyDocument, StudioBody, StudioMode, StudioSelection,
    ANCHOR_A_COLOR, ANCHOR_B_COLOR, ANCHOR_SPHERE_RADIUS,
};

pub(crate) fn on_body_list_selected(
    value_change: On<ValueChange<Entity>>,
    mut selection: ResMut<StudioSelection>,
    items: Query<&BodyListItem>,
    listboxes: Query<(), With<ListBox>>,
    parents: Query<&ChildOf>,
) {
    if !event_targets_list(&value_change, &listboxes, &parents) {
        return;
    }
    let Ok(item) = items.get(value_change.event().value) else {
        return;
    };
    *selection = StudioSelection::Body(item.name.clone());
}

pub(crate) fn on_joint_list_selected(
    value_change: On<ValueChange<Entity>>,
    mut selection: ResMut<StudioSelection>,
    items: Query<&JointListItem>,
    listboxes: Query<(), With<ListBox>>,
    parents: Query<&ChildOf>,
) {
    if !event_targets_list(&value_change, &listboxes, &parents) {
        return;
    }
    let Ok(item) = items.get(value_change.event().value) else {
        return;
    };
    *selection = StudioSelection::Joint(item.name.clone());
}

fn event_targets_list(
    value_change: &On<ValueChange<Entity>>,
    listboxes: &Query<(), With<ListBox>>,
    parents: &Query<&ChildOf>,
) -> bool {
    let change = value_change.event();
    if listboxes.contains(change.source) {
        return true;
    }
    parents
        .iter_ancestors(change.value)
        .any(|ancestor| listboxes.contains(ancestor))
}

pub(crate) fn apply_studio_selection(
    mut commands: Commands,
    selection: Res<StudioSelection>,
    mode: Res<StudioMode>,
    bodies: Query<(Entity, &StudioBody)>,
    focused: Query<Entity, With<TransformGizmoFocus>>,
    body_rows: Query<(Entity, &BodyListItem, Has<Selected>)>,
    joint_rows: Query<(Entity, &JointListItem, Has<Selected>)>,
) {
    if !selection.is_changed() && !mode.is_changed() {
        return;
    }

    for entity in &focused {
        commands.entity(entity).remove::<TransformGizmoFocus>();
    }

    if *mode != StudioMode::Edit {
        set_named_row_selection(&mut commands, &body_rows, None);
        set_named_row_selection(&mut commands, &joint_rows, None);
        return;
    }

    match selection.as_ref() {
        StudioSelection::None => {
            set_named_row_selection(&mut commands, &body_rows, None);
            set_named_row_selection(&mut commands, &joint_rows, None);
        }
        StudioSelection::Body(name) => {
            set_named_row_selection(&mut commands, &joint_rows, None);
            set_named_row_selection(&mut commands, &body_rows, Some(name.as_str()));
            if let Some((entity, _)) = bodies.iter().find(|(_, body)| body.name == *name) {
                commands.entity(entity).insert(TransformGizmoFocus);
            }
        }
        StudioSelection::Joint(name) => {
            set_named_row_selection(&mut commands, &body_rows, None);
            set_named_row_selection(&mut commands, &joint_rows, Some(name.as_str()));
        }
    }
}

fn set_named_row_selection<T: Component + NameMatch>(
    commands: &mut Commands,
    rows: &Query<(Entity, &T, Has<Selected>)>,
    selected_name: Option<&str>,
) {
    for (entity, item, selected) in rows.iter() {
        let should_select = selected_name.is_some_and(|name| item.matches(name));
        if should_select && !selected {
            commands.entity(entity).insert(Selected);
        } else if !should_select && selected {
            commands.entity(entity).remove::<Selected>();
        }
    }
}

trait NameMatch {
    fn matches(&self, name: &str) -> bool;
}

impl NameMatch for BodyListItem {
    fn matches(&self, name: &str) -> bool {
        self.name == name
    }
}

impl NameMatch for JointListItem {
    fn matches(&self, name: &str) -> bool {
        self.name == name
    }
}

/// Draw selected joint anchors as gizmos with depth bias so they stay visible through bodies.
pub(crate) fn draw_joint_anchors(
    selection: Res<StudioSelection>,
    document: Res<MorphologyDocument>,
    mode: Res<StudioMode>,
    mut gizmos: Gizmos,
) {
    if *mode != StudioMode::Edit {
        return;
    }
    let StudioSelection::Joint(name) = selection.as_ref() else {
        return;
    };
    let Some(joint) = document
        .creature
        .joints
        .iter()
        .find(|joint| joint.name == *name)
    else {
        return;
    };
    let Some((anchor_a, anchor_b)) = joint_anchor_world_positions(&document, joint) else {
        return;
    };
    gizmos.sphere(anchor_a, ANCHOR_SPHERE_RADIUS, ANCHOR_A_COLOR);
    gizmos.sphere(anchor_b, ANCHOR_SPHERE_RADIUS, ANCHOR_B_COLOR);
}

pub(crate) fn configure_anchor_gizmo_depth(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    // -1 draws gizmos in front of all scene geometry.
    config.depth_bias = -1.0;
}

fn joint_anchor_world_positions(
    document: &MorphologyDocument,
    joint: &JointDesc,
) -> Option<(Vec3, Vec3)> {
    let poses = compute_zero_body_poses(&document.creature);
    let pose_a = poses.get(&joint.body_a)?;
    let pose_b = poses.get(&joint.body_b)?;
    let world_a = pose_a.translation + pose_a.rotation * joint.anchor_a;
    let world_b = pose_b.translation + pose_b.rotation * joint.anchor_b;
    Some((world_a, world_b))
}

pub(crate) fn on_click_select_body(
    click: On<Pointer<Click>>,
    mode: Res<StudioMode>,
    mut selection: ResMut<StudioSelection>,
    bodies: Query<&StudioBody>,
) {
    if *mode != StudioMode::Edit || click.button != PointerButton::Primary {
        return;
    }
    let Ok(body) = bodies.get(click.entity) else {
        return;
    };
    *selection = StudioSelection::Body(body.name.clone());
}

pub(crate) fn clear_selection_on_mode_change(
    mode: Res<StudioMode>,
    mut selection: ResMut<StudioSelection>,
) {
    if mode.is_changed() && *mode != StudioMode::Edit {
        *selection = StudioSelection::None;
    }
}

/// Rebuild list rows after a morphology load. Despawns existing [`ListItem`]s under each
/// list view and spawns fresh Feathers rows for the current document.
pub(crate) fn rebuild_hierarchy_lists(
    commands: &mut Commands,
    document: &MorphologyDocument,
    body_lists: &Query<Entity, With<crate::state::BodyListView>>,
    joint_lists: &Query<Entity, With<crate::state::JointListView>>,
    children: &Query<&Children>,
    list_items: &Query<(), With<ListItem>>,
    scroll_areas: &Query<(), With<ScrollArea>>,
) {
    for list_entity in body_lists.iter() {
        rebuild_list_rows(
            commands,
            list_entity,
            children,
            list_items,
            scroll_areas,
            crate::ui::body_list_rows(&document.creature.bodies),
        );
    }
    for list_entity in joint_lists.iter() {
        rebuild_list_rows(
            commands,
            list_entity,
            children,
            list_items,
            scroll_areas,
            crate::ui::joint_list_rows(&document.creature.joints),
        );
    }
}

fn rebuild_list_rows(
    commands: &mut Commands,
    list_entity: Entity,
    children: &Query<&Children>,
    list_items: &Query<(), With<ListItem>>,
    scroll_areas: &Query<(), With<ScrollArea>>,
    rows: Box<dyn SceneList>,
) {
    let existing: Vec<Entity> = children
        .iter_descendants(list_entity)
        .filter(|entity| list_items.contains(*entity))
        .collect();
    for entity in existing {
        commands.entity(entity).despawn();
    }
    let Some(scroll_entity) = children
        .iter_descendants(list_entity)
        .find(|entity| scroll_areas.contains(*entity))
    else {
        return;
    };
    commands
        .entity(scroll_entity)
        .queue_spawn_related_scenes::<Children>(rows);
}
