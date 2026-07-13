use avian3d::prelude::*;
use bevy::prelude::*;
use sim_core::prelude::*;

/// Flat ground + simple dog-like quadruped.
pub struct DogGroundPlugin;

impl Plugin for DogGroundPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_requested_batch)
            .add_systems(FixedUpdate, apply_limb_flail_torques);
    }
}

/// How many dog-ground envs to spawn at startup.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SpawnDogGroundBatch {
    pub count: u32,
    pub interpolate: bool,
}

impl Default for SpawnDogGroundBatch {
    fn default() -> Self {
        Self {
            count: 8,
            interpolate: false,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct DogGroundEnv {
    pub env_id: EnvId,
}

/// Limb that receives random torque flailing each fixed tick.
#[derive(Component, Clone, Copy, Debug)]
pub struct FlailLimb;

/// Medium flail torque magnitude (N⋅m).
const FLAIL_TORQUE_STRENGTH: f32 = 3.5;

fn apply_limb_flail_torques(
    tick: Res<SimTick>,
    mut limbs: Query<(Entity, &mut ConstantTorque), With<FlailLimb>>,
) {
    for (entity, mut torque) in &mut limbs {
        let seed = tick
            .0
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(entity.to_bits());
        let direction = random_unit_vector(seed);
        *torque = ConstantTorque(direction * FLAIL_TORQUE_STRENGTH);
    }
}

fn random_unit_vector(seed: u64) -> Vec3 {
    let x = hash_signed(seed);
    let y = hash_signed(seed.wrapping_mul(0x85EB_CA6B));
    let z = hash_signed(seed.wrapping_mul(0xC2B2_AE35));
    Vec3::new(x, y, z).normalize_or_zero()
}

fn hash_signed(seed: u64) -> f32 {
    let mut value = seed;
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
}

fn spawn_requested_batch(
    mut commands: Commands,
    batch: Option<Res<SpawnDogGroundBatch>>,
    isolation: Res<EnvIsolationConfig>,
) {
    let Some(batch) = batch else {
        return;
    };
    for index in 0..batch.count {
        spawn_dog_ground_env(
            &mut commands,
            EnvId::new(index),
            &isolation,
            batch.interpolate,
        );
    }
}

/// Spawns one isolated flat-ground + dog environment.
pub fn spawn_dog_ground_env(
    commands: &mut Commands,
    env_id: EnvId,
    isolation: &EnvIsolationConfig,
    interpolate: bool,
) {
    let origin = env_origin(env_id, isolation);
    let layers = env_collision_layers(env_id);

    let _root = spawn_env_root(commands, env_id, isolation);

    let ground_size = isolation.spacing * 0.9;
    let ground_mesh = generate_bumpy_ground_mesh(env_id.index(), GROUND_RESOLUTION, ground_size);
    let collider = Collider::trimesh(ground_mesh.vertices.clone(), ground_mesh.indices.clone());
    commands.spawn((
        Name::new(format!("ground_{}", env_id.index())),
        DogGroundEnv { env_id },
        SimBody { env_id },
        ground_mesh,
        RigidBody::Static,
        collider,
        layers,
        Friction::new(0.9),
        Transform::from_translation(origin),
    ));

    let dog = dog_quadruped_desc();
    let instance = spawn_creature(commands, env_id, origin, &dog, interpolate);

    for (body_name, body_entity) in &instance.bodies {
        let is_limb = body_name.ends_with("_upper") || body_name.ends_with("_lower");
        if is_limb {
            commands
                .entity(*body_entity)
                .insert((FlailLimb, ConstantTorque::default()));
        }
    }
}

/// Shared triangle mesh for bumpy ground (same data for physics collider and debug render).
#[derive(Component, Clone, Debug)]
pub struct GroundMeshData {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
}

const GROUND_RESOLUTION: usize = 32;
const GROUND_BUMP_AMPLITUDE: f32 = 0.4;

fn generate_bumpy_ground_mesh(seed: u32, resolution: usize, size: f32) -> GroundMeshData {
    let mut vertices = Vec::with_capacity(resolution * resolution);
    for row in 0..resolution {
        for column in 0..resolution {
            let u = if resolution > 1 {
                row as f32 / (resolution - 1) as f32
            } else {
                0.0
            };
            let v = if resolution > 1 {
                column as f32 / (resolution - 1) as f32
            } else {
                0.0
            };
            let sample_seed = (seed as u64)
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((row as u64) << 16)
                .wrapping_add(column as u64);
            let edge_fade = edge_fade_factor(row, column, resolution);
            let bump = (hash_unit(sample_seed) * 2.0 - 1.0) * GROUND_BUMP_AMPLITUDE * edge_fade;
            vertices.push(Vec3::new((u - 0.5) * size, bump, (v - 0.5) * size));
        }
    }

    let mut indices = Vec::with_capacity((resolution - 1) * (resolution - 1) * 2);
    for row in 0..(resolution - 1) {
        for column in 0..(resolution - 1) {
            let top_left = (row * resolution + column) as u32;
            let top_right = top_left + 1;
            let bottom_left = ((row + 1) * resolution + column) as u32;
            let bottom_right = bottom_left + 1;
            // Counter-clockwise when viewed from above (+Y).
            indices.push([top_left, top_right, bottom_left]);
            indices.push([top_right, bottom_right, bottom_left]);
        }
    }

    GroundMeshData { vertices, indices }
}

fn edge_fade_factor(row: usize, column: usize, resolution: usize) -> f32 {
    let max_index = (resolution - 1) as f32;
    let u = row as f32 / max_index;
    let v = column as f32 / max_index;
    let fade_u = (u.min(1.0 - u) * 4.0).clamp(0.0, 1.0);
    let fade_v = (v.min(1.0 - v) * 4.0).clamp(0.0, 1.0);
    fade_u.min(fade_v)
}

fn hash_unit(seed: u64) -> f32 {
    let mut value = seed;
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as f64 / u64::MAX as f64) as f32
}

/// Dog-like quadruped: torso + head + four two-segment legs (capsules + revolute joints).
pub fn dog_quadruped_desc() -> CreatureDesc {
    let torso_half = Vec3::new(0.35, 0.12, 0.18);
    let upper_leg_length = 0.28;
    let lower_leg_length = 0.28;
    let leg_radius = 0.05;
    let torso_y = 1.4;
    let hip_forward = 0.22;
    let hip_side = 0.16;
    let hip_down = 0.12;

    let legs = [
        ("fl", hip_forward, hip_side),
        ("fr", hip_forward, -hip_side),
        ("bl", -hip_forward, hip_side),
        ("br", -hip_forward, -hip_side),
    ];

    let mut dog = CreatureDesc::new("dog")
        .body(
            BodyDesc::new(
                "torso",
                BodyShape::Cuboid {
                    half_extents: torso_half,
                },
            )
            .density(250.0)
            .at(Vec3::new(0.0, torso_y, 0.0)),
        )
        .body(
            BodyDesc::new(
                "head",
                BodyShape::Cuboid {
                    half_extents: Vec3::new(0.12, 0.09, 0.10),
                },
            )
            .density(180.0)
            .at(Vec3::new(torso_half.x + 0.14, torso_y + 0.04, 0.0)),
        );

    let mut neck = JointDesc::spherical(
        "neck",
        "torso",
        "head",
        Vec3::new(torso_half.x, 0.02, 0.0),
        Vec3::new(-0.12, -0.02, 0.0),
    );
    if let JointKind::Spherical {
        ref mut swing_limits,
        ref mut twist_limits,
        ..
    } = neck.kind
    {
        *swing_limits = Some((-0.6, 0.6));
        *twist_limits = Some((-0.4, 0.4));
    }
    dog = dog.joint(neck);

    for (label, hip_x, hip_z) in legs {
        let upper_name = format!("{label}_upper");
        let lower_name = format!("{label}_lower");

        let upper_center = Vec3::new(hip_x, torso_y - hip_down - upper_leg_length * 0.5, hip_z);
        let lower_center = Vec3::new(
            hip_x,
            torso_y - hip_down - upper_leg_length - lower_leg_length * 0.5,
            hip_z,
        );

        dog = dog
            .body(
                BodyDesc::new(
                    upper_name.clone(),
                    BodyShape::Capsule {
                        radius: leg_radius,
                        length: upper_leg_length,
                    },
                )
                .density(220.0)
                .at(upper_center),
            )
            .body(
                BodyDesc::new(
                    lower_name.clone(),
                    BodyShape::Capsule {
                        radius: leg_radius * 0.9,
                        length: lower_leg_length,
                    },
                )
                .density(200.0)
                .at(lower_center),
            )
            .joint(
                JointDesc::revolute(
                    format!("{label}_hip"),
                    "torso",
                    upper_name.clone(),
                    Vec3::new(hip_x, -hip_down, hip_z),
                    Vec3::new(0.0, upper_leg_length * 0.5, 0.0),
                    Vec3::Z,
                )
                .with_angle_limits(-1.2, 0.8),
            )
            .joint(
                JointDesc::revolute(
                    format!("{label}_knee"),
                    upper_name,
                    lower_name,
                    Vec3::new(0.0, -upper_leg_length * 0.5, 0.0),
                    Vec3::new(0.0, lower_leg_length * 0.5, 0.0),
                    Vec3::Z,
                )
                .with_angle_limits(-0.1, 2.0),
            );
    }

    dog
}
