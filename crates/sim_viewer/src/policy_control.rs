//! Load a trained policy and drive envs with deterministic mean actions.

use std::path::Path;

use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui_widgets::{observe, Activate, Button};
use burn::{
    backend::Wgpu,
    tensor::{Device, Tensor},
};
use policy::{
    creature_checkpoint_dir, load_policy_checkpoint, resolve_checkpoint_stem, ActorCritic,
    ActorCriticConfig,
};
use sim_core::prelude::*;

const POLICY_BUTTON_BG: Color = Color::srgb(0.18, 0.22, 0.28);
const POLICY_BUTTON_BG_HOVER: Color = Color::srgb(0.28, 0.34, 0.42);

type InferenceBackend = Wgpu;

/// Long horizon so viewing is not interrupted by frequent episode resets.
pub const VIEWER_EPISODE_HORIZON: u32 = 3_600;

#[derive(Resource)]
pub struct ViewerPolicy {
    device: Device<InferenceBackend>,
    model: Option<ActorCritic<InferenceBackend>>,
    pub display_name: String,
}

impl Default for ViewerPolicy {
    fn default() -> Self {
        Self {
            device: Default::default(),
            model: None,
            display_name: "none".to_string(),
        }
    }
}

impl ViewerPolicy {
    fn clear(&mut self) {
        self.model = None;
        self.display_name = "none".to_string();
    }

    fn load_from_path(
        &mut self,
        path: &Path,
        creature_id: &str,
        observation_dim: usize,
        action_dim: usize,
    ) -> Result<(), String> {
        let config = ActorCriticConfig::from_arch_file(observation_dim, action_dim, None)
            .map_err(|error| format!("failed to load actor-critic config: {error}"))?;
        let (model, meta) =
            load_policy_checkpoint::<InferenceBackend>(&self.device, path, creature_id, &config)
                .map_err(|error| {
                    format!("failed to load policy from {}: {error}", path.display())
                })?;

        self.model = Some(model);
        self.display_name = display_name_for_path(path);
        bevy::log::info!(
            "loaded policy '{}' (update_index={}, mean_rewards={}, mean_episode_lengths={})",
            self.display_name,
            meta.update_index,
            meta.mean_rewards.len(),
            meta.mean_episode_lengths.len()
        );
        Ok(())
    }
}

#[derive(Component)]
struct LoadPolicyButton;

#[derive(Component)]
struct ClearPolicyButton;

#[derive(Component)]
struct PolicyNameLabel;

pub fn spawn_policy_controls(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(8),
            ..default()
        },))
        .with_children(|row| {
            row.spawn((
                Button,
                LoadPolicyButton,
                Hovered::default(),
                Node {
                    padding: UiRect::axes(px(10), px(4)),
                    border_radius: BorderRadius::all(px(4)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(POLICY_BUTTON_BG),
                observe(on_load_policy_activated),
            ))
            .with_children(|button| {
                button.spawn((
                    Text::new("Choose Policy"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.93, 0.95)),
                ));
            });

            row.spawn((
                Button,
                ClearPolicyButton,
                Hovered::default(),
                Node {
                    padding: UiRect::axes(px(10), px(4)),
                    border_radius: BorderRadius::all(px(4)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(POLICY_BUTTON_BG),
                observe(on_clear_policy_activated),
            ))
            .with_children(|button| {
                button.spawn((
                    Text::new("Clear"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.92, 0.93, 0.95)),
                ));
            });

            row.spawn((
                Text::new("none"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.75, 0.8, 0.85)),
                PolicyNameLabel,
            ));
        });
}

fn init_rl_buffers(
    mut buffers: ResMut<RlBuffers>,
    state: Res<super::ViewerState>,
    spec: Res<CreatureSpec>,
) {
    buffers.resize(
        state.env_count,
        spec.observation_dim,
        spec.action_dim,
        VIEWER_EPISODE_HORIZON,
    );
}

fn apply_viewer_policy_mean_actions(
    policy: Res<ViewerPolicy>,
    spec: Res<CreatureSpec>,
    mut buffers: ResMut<RlBuffers>,
) {
    let env_count = buffers.observations.len();
    if env_count == 0 {
        return;
    }

    let Some(model) = policy.model.as_ref() else {
        for actions in &mut buffers.actions {
            actions.fill(0.0);
        }
        return;
    };

    let observation_dim = spec.observation_dim;
    let action_dim = spec.action_dim;

    let mut flat_observations = Vec::with_capacity(env_count * observation_dim);
    for observation in &buffers.observations {
        flat_observations.extend_from_slice(observation);
    }

    let observations_tensor =
        Tensor::<InferenceBackend, 1>::from_floats(flat_observations.as_slice(), &policy.device)
            .reshape([env_count, observation_dim]);

    let output = model.forward(observations_tensor);
    let mean_values = output.mean.to_data().to_vec::<f32>().unwrap_or_default();

    for env_index in 0..env_count {
        let Some(actions) = buffers.actions.get_mut(env_index) else {
            continue;
        };
        let offset = env_index * action_dim;
        if offset + action_dim > mean_values.len() {
            actions.fill(0.0);
            continue;
        }
        actions.copy_from_slice(&mean_values[offset..offset + action_dim]);
        for value in actions.iter_mut() {
            *value = value.clamp(-1.0, 1.0);
        }
    }
}

fn sync_policy_name_label(
    policy: Res<ViewerPolicy>,
    mut labels: Query<&mut Text, With<PolicyNameLabel>>,
) {
    if !policy.is_changed() {
        return;
    }
    for mut text in &mut labels {
        *text = Text::new(policy.display_name.clone());
    }
}

fn tint_policy_buttons(
    mut buttons: Query<
        (&Hovered, &mut BackgroundColor),
        Or<(With<LoadPolicyButton>, With<ClearPolicyButton>)>,
    >,
) {
    for (hovered, mut background) in &mut buttons {
        background.0 = if hovered.0 {
            POLICY_BUTTON_BG_HOVER
        } else {
            POLICY_BUTTON_BG
        };
    }
}

fn on_load_policy_activated(
    _activate: On<Activate>,
    mut policy: ResMut<ViewerPolicy>,
    mut buffers: ResMut<RlBuffers>,
    spec: Res<CreatureSpec>,
) {
    let mut dialog = rfd::FileDialog::new()
        .add_filter("Policy checkpoint", &["mpk", "json"])
        .set_title(format!("Load {} policy checkpoint", spec.id));

    match creature_checkpoint_dir(spec.id) {
        Ok(directory) => {
            if let Err(error) = std::fs::create_dir_all(&directory) {
                bevy::log::warn!(
                    "could not create checkpoint directory {}: {error}",
                    directory.display()
                );
            }
            dialog = dialog.set_directory(directory);
        }
        Err(error) => {
            bevy::log::warn!("checkpoint directory unavailable: {error}");
        }
    }

    let Some(path) = dialog.pick_file() else {
        return;
    };

    if let Err(message) =
        policy.load_from_path(&path, spec.id, spec.observation_dim, spec.action_dim)
    {
        bevy::log::error!("{message}");
        policy.clear();
        for actions in &mut buffers.actions {
            actions.fill(0.0);
        }
    }
}

fn on_clear_policy_activated(
    _activate: On<Activate>,
    mut policy: ResMut<ViewerPolicy>,
    mut buffers: ResMut<RlBuffers>,
) {
    policy.clear();
    for actions in &mut buffers.actions {
        actions.fill(0.0);
    }
    bevy::log::info!("cleared viewer policy (mid-range joint targets)");
}

fn display_name_for_path(path: &Path) -> String {
    let stem = resolve_checkpoint_stem(path);
    stem.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| stem.display().to_string())
}

/// Register policy UI + inference systems on the viewer app.
pub fn configure_policy_control(app: &mut App) {
    app.init_resource::<ViewerPolicy>()
        .add_systems(Startup, init_rl_buffers)
        .add_systems(
            FixedUpdate,
            apply_viewer_policy_mean_actions.before(ControlSystems::ApplyActions),
        )
        .add_systems(Update, (sync_policy_name_label, tint_policy_buttons));
}
