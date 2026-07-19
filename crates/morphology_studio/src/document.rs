//! Morphology document save helper.

use bevy::prelude::*;
use sim_core::save_ron_config;

use crate::state::MorphologyDocument;

pub(crate) fn save_document(document: &mut MorphologyDocument) {
    match save_ron_config(&document.path, &document.creature) {
        Ok(()) => {
            document.dirty = false;
            document.status = format!("saved {}", document.path.display());
            info!("saved morphology to {}", document.path.display());
        }
        Err(error) => {
            document.status = format!("save failed: {error}");
            error!("failed to save morphology: {error}");
        }
    }
}
