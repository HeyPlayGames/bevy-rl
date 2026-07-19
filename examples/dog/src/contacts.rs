//! Ground-contact sensing for the dog balance task.

use avian3d::prelude::*;
use bevy::prelude::*;
use sim_core::prelude::*;

/// Lower-leg body names in observation / reward order: FL, FR, BL, BR.
pub const FOOT_PART_NAMES: [&str; 4] = ["fl_lower", "fr_lower", "bl_lower", "br_lower"];

/// Per-env ground contact flags used by observations and rewards.
#[derive(Clone, Copy, Debug, Default)]
pub struct DogGroundContacts {
    /// True when each lower leg is touching that env's ground.
    pub feet: [bool; 4],
}

impl DogGroundContacts {
    pub fn foot_contact_count(self) -> u32 {
        self.feet.iter().filter(|&&touching| touching).count() as u32
    }

    pub fn write_observation(self, observation: &mut [f32], offset: usize) {
        if offset + 4 > observation.len() {
            return;
        }
        for (index, &touching) in self.feet.iter().enumerate() {
            observation[offset + index] = if touching { 1.0 } else { 0.0 };
        }
    }
}

/// Latest ground contacts, one entry per env index.
#[derive(Resource, Default)]
pub struct DogContactBuffer {
    pub envs: Vec<DogGroundContacts>,
}

/// Refresh [`DogContactBuffer`] from Avian touching contacts vs each env's ground.
pub fn update_dog_contacts(
    mut buffer: ResMut<DogContactBuffer>,
    rl_buffers: Res<RlBuffers>,
    parts: Query<(Entity, &SimBody, &CreaturePart)>,
    grounds: Query<(Entity, &FlatGround)>,
    collisions: Collisions,
) {
    let env_count = rl_buffers.observations.len();
    if env_count == 0 {
        buffer.envs.clear();
        return;
    }

    buffer.envs.resize(env_count, DogGroundContacts::default());
    for contacts in &mut buffer.envs {
        *contacts = DogGroundContacts::default();
    }

    let mut ground_by_env = vec![None; env_count];
    for (ground_entity, ground) in &grounds {
        let env_index = ground.env_id.index() as usize;
        if env_index < env_count {
            ground_by_env[env_index] = Some(ground_entity);
        }
    }

    for (part_entity, body, part) in &parts {
        let env_index = body.env_id.index() as usize;
        if env_index >= env_count {
            continue;
        }
        let Some(ground_entity) = ground_by_env[env_index] else {
            continue;
        };
        if !entities_touching(&collisions, part_entity, ground_entity) {
            continue;
        }

        let contacts = &mut buffer.envs[env_index];
        if let Some(foot_index) = FOOT_PART_NAMES
            .iter()
            .position(|name| *name == part.name.as_str())
        {
            contacts.feet[foot_index] = true;
        }
    }
}

fn entities_touching(collisions: &Collisions, entity_a: Entity, entity_b: Entity) -> bool {
    collisions
        .get(entity_a, entity_b)
        .is_some_and(ContactPair::is_touching)
}
