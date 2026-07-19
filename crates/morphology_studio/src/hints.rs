//! On-screen instruction copy for edit vs simulate modes.

use bevy::gizmos::transform_gizmo::{TransformGizmoMode, TransformGizmoSpace};

use crate::state::StudioMode;

pub(crate) fn instructions_text(
    studio_mode: StudioMode,
    gizmo_mode: TransformGizmoMode,
    space: TransformGizmoSpace,
) -> String {
    match studio_mode {
        StudioMode::Edit => {
            let mode_str = match gizmo_mode {
                TransformGizmoMode::Translate => "Translate",
                TransformGizmoMode::Rotate => "Rotate",
                TransformGizmoMode::Scale => "Scale (dimensions)",
            };
            let space_str = match space {
                TransformGizmoSpace::World => "World",
                TransformGizmoSpace::Local => "Local",
            };
            format!(
                "Morphology studio\n\
                 Mode: edit\n\
                 \n\
                 Bodies / Joints lists\n\
                 Click body: gizmo edit\n\
                 Click joint: anchors\n\
                 (A red, B blue)\n\
                 \n\
                 1 Translate\n\
                 2 Rotate\n\
                 3 Scale (dimensions)\n\
                 X World / Local\n\
                 \n\
                 Left-drag orbit\n\
                 Right-drag pan\n\
                 Scroll zoom\n\
                 Ctrl+S Save\n\
                 \n\
                 Gizmo: {mode_str}\n\
                 Space: {space_str}"
            )
        }
        StudioMode::Simulating => {
            "Morphology studio\n\
             Mode: simulating (no policy)\n\
             \n\
             Passive physics preview\n\
             under gravity\n\
             \n\
             Left-drag orbit\n\
             Right-drag pan\n\
             Scroll zoom\n\
             \n\
             Reset returns to edit"
                .to_string()
        }
    }
}
