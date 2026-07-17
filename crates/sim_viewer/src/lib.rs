//! Multi-view client: watch parallel envs with transform interpolation.
//!
//! Creature-agnostic: compose with a creature pack plugin, [`CreatureSpec`],
//! [`SpawnEnvBatch`], and [`ViewerCreatureVisuals`].

mod policy_control;

use std::f32::consts::FRAC_PI_2;

use avian3d::prelude::*;
use bevy::camera::Viewport;
use bevy::ecs::message::MessageWriter;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::IsDefaultUiCamera;
use bevy::ui_widgets::{
    observe, slider_self_update, Button, Slider, SliderDragState, SliderRange, SliderThumb,
    SliderValue, TrackClick, ValueChange,
};
use bevy::window::WindowResolution;
use policy_control::{configure_policy_control, spawn_policy_controls, VIEWER_EPISODE_HORIZON};
use sim_core::prelude::*;

const MAX_VIEW_COUNT: usize = 64;
const DEFAULT_ENV_COUNT: usize = 4;

const SLIDER_TRACK: Color = Color::srgb(0.12, 0.13, 0.16);
const SLIDER_THUMB: Color = Color::srgb(0.35, 0.65, 0.42);

const ORBIT_YAW_SPEED: f32 = 0.005;
const ORBIT_PITCH_SPEED: f32 = 0.004;
const ORBIT_PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01;
const ORBIT_EYE_OFFSET: Vec3 = Vec3::new(2.4, 1.6, 2.4);
const ORBIT_TARGET_HEIGHT: f32 = 0.4;

/// Morphology used to build debug meshes for dynamic bodies.
#[derive(Resource, Clone, Debug)]
pub struct ViewerCreatureVisuals {
    pub creature: CreatureDesc,
    pub creature_color: Color,
    pub ground_color: Color,
}

impl Default for ViewerCreatureVisuals {
    fn default() -> Self {
        Self {
            creature: CreatureDesc::new("creature"),
            creature_color: Color::srgb(0.75, 0.55, 0.35),
            ground_color: Color::srgb(0.62, 0.62, 0.66),
        }
    }
}

pub struct SimViewerPlugin {
    pub env_count: u32,
}

impl Default for SimViewerPlugin {
    fn default() -> Self {
        Self {
            env_count: DEFAULT_ENV_COUNT as u32,
        }
    }
}

impl Plugin for SimViewerPlugin {
    fn build(&self, app: &mut App) {
        let env_count = (self.env_count as usize).clamp(1, MAX_VIEW_COUNT);
        app.insert_resource(ViewerState {
            env_count,
            spawned_count: env_count,
        })
        .insert_resource(SpawnEnvBatch {
            count: env_count as u32,
            interpolate: true,
        })
        .insert_resource(Gravity(Vec3::NEG_Y * 9.81))
        .insert_resource(ClearColor(Color::srgb(0.08, 0.09, 0.11)))
        .add_plugins(PhysicsPlugins::default())
        .add_plugins(SimCorePlugin {
            fixed_hz: 60.0,
            isolation: EnvIsolationConfig {
                spacing: 40.0,
                grid_columns: 16,
            },
            interpolate_transforms: true,
        });
        configure_policy_control(app);
        app.init_resource::<OrbitDrag>()
            .add_systems(
                Startup,
                (
                    setup_lights,
                    setup_cameras,
                    setup_viewer_ui,
                    setup_debug_meshes,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    sync_camera_viewports,
                    sync_env_labels,
                    sync_slider_visuals,
                    orbit_env_cameras,
                    attach_debug_meshes,
                ),
            );
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct ViewerState {
    /// Active viewports / watched env count (1..=64).
    env_count: usize,
    /// How many envs are currently spawned in the world.
    spawned_count: usize,
}

#[derive(Component)]
struct EnvCamera {
    slot: usize,
    yaw: f32,
    pitch: f32,
    radius: f32,
    center: Vec3,
}

#[derive(Resource, Default)]
struct OrbitDrag {
    /// Slot locked for the current drag; kept even if the cursor leaves that viewport.
    active_slot: Option<usize>,
}

#[derive(Component)]
struct EnvLabel {
    slot: usize,
}

#[derive(Component)]
struct EnvCountSlider;

#[derive(Component)]
struct EnvCountSliderThumb;

#[derive(Component)]
struct EnvCountLabel;

fn setup_lights(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 4_500.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.6, 0.0)),
        CascadeShadowConfigBuilder {
            maximum_distance: 25.0,
            ..default()
        }
        .build(),
    ));
}

fn setup_cameras(mut commands: Commands, isolation: Res<EnvIsolationConfig>) {
    // Dedicated UI camera: full-window, transparent clear, above all 3D views.
    commands.spawn((
        Camera2d,
        Camera {
            order: MAX_VIEW_COUNT as isize + 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        IsDefaultUiCamera,
    ));

    for slot in 0..MAX_VIEW_COUNT {
        let env_id = EnvId::new(slot as u32);
        let center = env_origin(env_id, &isolation) + Vec3::new(0.0, ORBIT_TARGET_HEIGHT, 0.0);
        let eye = center + ORBIT_EYE_OFFSET;
        let transform = Transform::from_translation(eye).looking_at(center, Vec3::Y);
        let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
        commands.spawn((
            Camera3d::default(),
            Camera {
                order: slot as isize,
                is_active: false,
                clear_color: if slot == 0 {
                    ClearColorConfig::Default
                } else {
                    ClearColorConfig::None
                },
                ..default()
            },
            EnvCamera {
                slot,
                yaw,
                pitch,
                radius: ORBIT_EYE_OFFSET.length(),
                center,
            },
            transform,
        ));
    }
}

fn setup_viewer_ui(mut commands: Commands, state: Res<ViewerState>) {
    commands
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                position_type: PositionType::Absolute,
                ..default()
            },
            GlobalZIndex(10),
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            for slot in 0..MAX_VIEW_COUNT {
                root.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(8),
                        top: px(8),
                        padding: UiRect::axes(px(8), px(4)),
                        border_radius: BorderRadius::all(px(4)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
                    EnvLabel { slot },
                    Visibility::Hidden,
                    Pickable::IGNORE,
                ))
                .with_children(|label| {
                    label.spawn((
                        Text::new(format!("Env {slot}")),
                        TextFont::from_font_size(16.0),
                        TextColor(Color::srgb(0.95, 0.95, 0.95)),
                    ));
                });
            }

            root.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    right: px(12),
                    top: px(12),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                    padding: UiRect::axes(px(12), px(8)),
                    border_radius: BorderRadius::all(px(6)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.05, 0.06, 0.08, 0.85)),
            ))
            .with_children(|controls| {
                controls.spawn((
                    Text::new("Envs"),
                    TextFont::from_font_size(14.0),
                    TextColor(Color::srgb(0.85, 0.85, 0.85)),
                ));

                controls.spawn((
                    Text::new(state.env_count.to_string()),
                    TextFont::from_font_size(14.0),
                    TextColor(Color::srgb(0.95, 0.95, 0.95)),
                    EnvCountLabel,
                ));

                controls.spawn(env_count_slider(state.env_count as f32));
                spawn_policy_controls(controls);
            });
        });
}

fn env_count_slider(initial: f32) -> impl Bundle {
    (
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Stretch,
            height: px(16),
            width: px(160),
            ..default()
        },
        EnvCountSlider,
        Hovered::default(),
        Slider {
            track_click: TrackClick::Snap,
            ..default()
        },
        SliderValue(initial),
        SliderRange::new(1.0, MAX_VIEW_COUNT as f32),
        Children::spawn((
            Spawn((
                Node {
                    height: px(6),
                    border_radius: BorderRadius::all(px(3)),
                    ..default()
                },
                BackgroundColor(SLIDER_TRACK),
            )),
            Spawn((
                Node {
                    display: Display::Flex,
                    position_type: PositionType::Absolute,
                    left: px(0),
                    right: px(12),
                    top: px(0),
                    bottom: px(0),
                    ..default()
                },
                children![(
                    EnvCountSliderThumb,
                    SliderThumb,
                    Node {
                        display: Display::Flex,
                        width: px(12),
                        height: px(12),
                        position_type: PositionType::Absolute,
                        left: percent(0),
                        border_radius: BorderRadius::MAX,
                        ..default()
                    },
                    BackgroundColor(SLIDER_THUMB),
                )],
            )),
        )),
        observe(slider_self_update),
        observe(on_env_count_changed),
    )
}

fn on_env_count_changed(
    value_change: On<ValueChange<f32>>,
    mut state: ResMut<ViewerState>,
    mut count_labels: Query<&mut Text, With<EnvCountLabel>>,
    mut buffers: ResMut<RlBuffers>,
    spec: Res<CreatureSpec>,
    mut respawn: MessageWriter<RespawnAllEnvs>,
) {
    let count = value_change.value.round().clamp(1.0, MAX_VIEW_COUNT as f32) as usize;
    state.env_count = count;

    for mut text in &mut count_labels {
        *text = Text::new(count.to_string());
    }

    if !value_change.is_final || state.spawned_count == count {
        return;
    }

    respawn.write(RespawnAllEnvs {
        count: count as u32,
        interpolate: true,
    });
    state.spawned_count = count;
    buffers.resize(
        count,
        spec.observation_dim,
        spec.action_dim,
        VIEWER_EPISODE_HORIZON,
    );
}

fn viewport_grid(view_count: usize) -> (u32, u32) {
    let columns = (view_count as f32).sqrt().ceil().max(1.0) as u32;
    let rows = ((view_count as f32) / columns as f32).ceil().max(1.0) as u32;
    (columns, rows)
}

fn viewport_layout(
    view_count: usize,
    width: u32,
    height: u32,
    slot: usize,
) -> Option<(UVec2, UVec2)> {
    if slot >= view_count || view_count == 0 {
        return None;
    }
    let (columns, rows) = viewport_grid(view_count);
    let cell_width = width / columns;
    let cell_height = height / rows;
    let column = slot as u32 % columns;
    let row = slot as u32 / columns;
    Some((
        UVec2::new(column * cell_width, row * cell_height),
        UVec2::new(cell_width, cell_height),
    ))
}

fn viewport_slot_at(
    view_count: usize,
    width: u32,
    height: u32,
    physical_cursor: Vec2,
) -> Option<usize> {
    for slot in 0..view_count {
        let Some((position, size)) = viewport_layout(view_count, width, height, slot) else {
            continue;
        };
        let min_x = position.x as f32;
        let min_y = position.y as f32;
        let max_x = min_x + size.x as f32;
        let max_y = min_y + size.y as f32;
        if physical_cursor.x >= min_x
            && physical_cursor.x < max_x
            && physical_cursor.y >= min_y
            && physical_cursor.y < max_y
        {
            return Some(slot);
        }
    }
    None
}

fn apply_orbit_transform(transform: &mut Transform, camera: &EnvCamera) {
    transform.rotation = Quat::from_euler(EulerRot::YXZ, camera.yaw, camera.pitch, 0.0);
    transform.translation = camera.center - *transform.forward() * camera.radius;
}

fn orbit_env_cameras(
    mut drag: ResMut<OrbitDrag>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    state: Res<ViewerState>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    ui_hovered: Query<&Hovered, Or<(With<Slider>, With<Button>)>>,
    mut cameras: Query<(&mut EnvCamera, &mut Transform)>,
) {
    if mouse_buttons.just_released(MouseButton::Left) {
        drag.active_slot = None;
    } else if !mouse_buttons.pressed(MouseButton::Left) {
        drag.active_slot = None;
    } else if mouse_buttons.just_pressed(MouseButton::Left) {
        let ui_blocking = ui_hovered.iter().any(|hovered| hovered.0);
        if !ui_blocking {
            let Ok(window) = windows.single() else {
                return;
            };
            let Some(cursor) = window.cursor_position() else {
                return;
            };
            let scale = window.scale_factor().max(0.0001);
            let physical_cursor = cursor * scale;
            drag.active_slot = viewport_slot_at(
                state.env_count,
                window.physical_width(),
                window.physical_height(),
                physical_cursor,
            );
        }
    }

    let Some(active_slot) = drag.active_slot else {
        return;
    };

    if !mouse_buttons.pressed(MouseButton::Left) {
        return;
    }

    let delta = mouse_motion.delta;
    if delta == Vec2::ZERO {
        return;
    }

    for (mut env_camera, mut transform) in &mut cameras {
        if env_camera.slot != active_slot {
            continue;
        }
        env_camera.yaw -= delta.x * ORBIT_YAW_SPEED;
        env_camera.pitch = (env_camera.pitch - delta.y * ORBIT_PITCH_SPEED)
            .clamp(-ORBIT_PITCH_LIMIT, ORBIT_PITCH_LIMIT);
        apply_orbit_transform(&mut transform, &env_camera);
        break;
    }
}

fn sync_camera_viewports(
    state: Res<ViewerState>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut cameras: Query<(&EnvCamera, &mut Camera)>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let width = window.physical_width();
    let height = window.physical_height();
    if width == 0 || height == 0 {
        return;
    }

    for (env_camera, mut camera) in &mut cameras {
        match viewport_layout(state.env_count, width, height, env_camera.slot) {
            Some((physical_position, physical_size)) => {
                camera.is_active = true;
                camera.viewport = Some(Viewport {
                    physical_position,
                    physical_size,
                    ..default()
                });
            }
            None => {
                camera.is_active = false;
                camera.viewport = None;
            }
        }
    }
}

fn sync_env_labels(
    state: Res<ViewerState>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut labels: Query<(&EnvLabel, &mut Node, &mut Visibility)>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let width = window.physical_width();
    let height = window.physical_height();
    if width == 0 || height == 0 {
        return;
    }
    let scale = window.scale_factor().max(0.0001);

    for (label, mut node, mut visibility) in &mut labels {
        match viewport_layout(state.env_count, width, height, label.slot) {
            Some((physical_position, _)) => {
                *visibility = Visibility::Visible;
                node.left = px(physical_position.x as f32 / scale + 8.0);
                node.top = px(physical_position.y as f32 / scale + 8.0);
            }
            None => {
                *visibility = Visibility::Hidden;
            }
        }
    }
}

fn sync_slider_visuals(
    sliders: Query<
        (
            Entity,
            &SliderValue,
            &SliderRange,
            &Hovered,
            &SliderDragState,
        ),
        (
            Or<(
                Changed<SliderValue>,
                Changed<Hovered>,
                Changed<SliderDragState>,
            )>,
            With<EnvCountSlider>,
        ),
    >,
    children: Query<&Children>,
    mut thumbs: Query<
        (&mut Node, &mut BackgroundColor, Has<EnvCountSliderThumb>),
        Without<EnvCountSlider>,
    >,
) {
    for (slider_entity, value, range, hovered, drag_state) in &sliders {
        for child in children.iter_descendants(slider_entity) {
            let Ok((mut thumb_node, mut thumb_background, is_thumb)) = thumbs.get_mut(child) else {
                continue;
            };
            if !is_thumb {
                continue;
            }
            thumb_node.left = percent(range.thumb_position(value.0) * 100.0);
            thumb_background.0 = if hovered.0 || drag_state.dragging {
                SLIDER_THUMB.lighter(0.3)
            } else {
                SLIDER_THUMB
            };
        }
    }
}

#[derive(Component)]
struct DebugMeshAttached;

#[derive(Resource)]
struct DebugMeshAssets {
    capsule_meshes: std::collections::HashMap<(u32, u32), Handle<Mesh>>,
    cuboid_meshes: std::collections::HashMap<(u32, u32, u32), Handle<Mesh>>,
    ground_mesh: Handle<Mesh>,
    creature_material: Handle<StandardMaterial>,
    ground_material: Handle<StandardMaterial>,
}

fn setup_debug_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    isolation: Res<EnvIsolationConfig>,
    visuals: Res<ViewerCreatureVisuals>,
) {
    let creature_material = materials.add(StandardMaterial {
        base_color: visuals.creature_color,
        perceptual_roughness: 0.85,
        ..default()
    });
    let ground_material = materials.add(StandardMaterial {
        base_color: visuals.ground_color,
        perceptual_roughness: 0.95,
        ..default()
    });

    let mut capsule_meshes = std::collections::HashMap::new();
    let mut cuboid_meshes = std::collections::HashMap::new();

    for body in &visuals.creature.bodies {
        match body.shape {
            BodyShape::Capsule { radius, length } => {
                let key = ((radius * 1000.0) as u32, (length * 1000.0) as u32);
                capsule_meshes
                    .entry(key)
                    .or_insert_with(|| meshes.add(Capsule3d::new(radius, length)));
            }
            BodyShape::Cuboid { half_extents } => {
                let key = (
                    (half_extents.x * 1000.0) as u32,
                    (half_extents.y * 1000.0) as u32,
                    (half_extents.z * 1000.0) as u32,
                );
                cuboid_meshes
                    .entry(key)
                    .or_insert_with(|| meshes.add(Cuboid::from_size(half_extents * 2.0)));
            }
            BodyShape::Cylinder { radius, height } => {
                let key = ((radius * 1000.0) as u32, (height * 1000.0) as u32);
                capsule_meshes
                    .entry(key)
                    .or_insert_with(|| meshes.add(Cylinder::new(radius, height)));
            }
            BodyShape::Sphere { radius } => {
                let key = ((radius * 1000.0) as u32, 0);
                capsule_meshes
                    .entry(key)
                    .or_insert_with(|| meshes.add(Sphere::new(radius)));
            }
        }
    }

    let ground_half = ground_half_extents(&isolation);
    let ground_mesh = meshes.add(Cuboid::from_size(ground_half * 2.0));

    commands.insert_resource(DebugMeshAssets {
        capsule_meshes,
        cuboid_meshes,
        ground_mesh,
        creature_material,
        ground_material,
    });
}

fn attach_debug_meshes(
    mut commands: Commands,
    assets: Res<DebugMeshAssets>,
    visuals: Res<ViewerCreatureVisuals>,
    bodies: Query<
        (
            Entity,
            &SimBody,
            &RigidBody,
            Option<&Name>,
            Option<&FlatGround>,
        ),
        Without<DebugMeshAttached>,
    >,
) {
    for (entity, sim_body, rigid_body, name, ground) in &bodies {
        if ground.is_some() {
            commands.entity(entity).insert((
                Mesh3d(assets.ground_mesh.clone()),
                MeshMaterial3d(assets.ground_material.clone()),
                DebugMeshAttached,
            ));
            let _ = (sim_body, rigid_body, name);
            continue;
        }

        let Some(name) = name else {
            continue;
        };
        let body_name = name
            .as_str()
            .rsplit_once('_')
            .map(|(prefix, _)| prefix)
            .unwrap_or(name.as_str());

        let Some(body) = visuals
            .creature
            .bodies
            .iter()
            .find(|body| body.name == body_name)
        else {
            continue;
        };

        let mesh_handle = match body.shape {
            BodyShape::Capsule { radius, length } => {
                let key = ((radius * 1000.0) as u32, (length * 1000.0) as u32);
                assets.capsule_meshes.get(&key).cloned()
            }
            BodyShape::Cuboid { half_extents } => {
                let key = (
                    (half_extents.x * 1000.0) as u32,
                    (half_extents.y * 1000.0) as u32,
                    (half_extents.z * 1000.0) as u32,
                );
                assets.cuboid_meshes.get(&key).cloned()
            }
            BodyShape::Cylinder { radius, height } => {
                let key = ((radius * 1000.0) as u32, (height * 1000.0) as u32);
                assets.capsule_meshes.get(&key).cloned()
            }
            BodyShape::Sphere { radius } => {
                let key = ((radius * 1000.0) as u32, 0);
                assets.capsule_meshes.get(&key).cloned()
            }
        };

        let Some(mesh) = mesh_handle else {
            continue;
        };

        commands.entity(entity).insert((
            Mesh3d(mesh),
            MeshMaterial3d(assets.creature_material.clone()),
            DebugMeshAttached,
        ));
    }
}

/// Build a windowed viewer app shell. Callers add a creature pack plugin and visuals.
pub fn build_viewer_app(env_count: u32) -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "bevy-rl viewer".into(),
            resolution: WindowResolution::new(1280, 720),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(SimViewerPlugin { env_count });
    app
}

/// Run the multi-view client with a creature pack and debug-mesh visuals.
pub fn run_viewer(
    env_count: u32,
    visuals: ViewerCreatureVisuals,
    add_creature_plugins: impl FnOnce(&mut App),
) {
    let mut app = build_viewer_app(env_count);
    app.insert_resource(visuals);
    add_creature_plugins(&mut app);
    app.run();
}
