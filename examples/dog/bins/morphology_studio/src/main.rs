//! Dog example: morphology studio entry point.

use dog::{dog_morphology_path, load_dog_morphology};
use morphology_studio::{run_morphology_studio, MorphologyStudioConfig};

fn main() {
    let morph_path = dog_morphology_path();
    let creature = load_dog_morphology().unwrap_or_else(|error| {
        panic!(
            "failed to load dog morphology from {}: {error}",
            morph_path.display()
        );
    });

    run_morphology_studio(MorphologyStudioConfig::new(creature, morph_path));
}
