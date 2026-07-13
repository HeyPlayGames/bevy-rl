use avian3d::prelude::*;
use bevy::prelude::*;
use sim_core::prelude::*;

/// Flat ground + simple dog-like quadruped.
pub struct DogGroundPlugin;

impl Plugin for DogGroundPlugin {
    fn build(&self, app: &mut App) {
        // Default is 6; more substeps reduces XPBD foot–ground microslip.
        app.insert_resource(SubstepCount(6))
            .add_systems(Startup, spawn_requested_batch)
            .add_systems(
                FixedUpdate,
                apply_joint_flail_targets.run_if(flail_retarget_due),
            );
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

/// Revolute joint whose motor target angle is randomized at [`FLAIL_RETARGET_HZ`].
///
/// Avian's joint motor drives toward the target; this marker only selects which joints flail.
#[derive(Component, Clone, Copy, Debug)]
pub struct FlailJoint;

/// How often new motor set-positions are sampled (Hz).
const FLAIL_RETARGET_HZ: f64 = 3.0;

/// Spring-damper frequency for the revolute motors (Hz). Higher than retarget rate so
/// joints can settle toward each set-position between samples.
const FLAIL_MOTOR_FREQUENCY: f32 = 8.0;

/// Cap on joint motor torque while flailing (N⋅m).
/// off for now to test floor jitter
const FLAIL_MOTOR_MAX_TORQUE: f32 = 0.0000001;

fn flail_motor() -> AngularMotor {
    AngularMotor::new(MotorModel::SpringDamper {
        frequency: FLAIL_MOTOR_FREQUENCY,
        damping_ratio: 1.0,
    })
    .with_max_torque(FLAIL_MOTOR_MAX_TORQUE)
}

fn flail_retarget_due(tick: Res<SimTick>, time: Res<Time<Fixed>>) -> bool {
    let fixed_hz = 1.0 / time.delta_secs_f64();
    let ticks_per_retarget = (fixed_hz / FLAIL_RETARGET_HZ).round().max(1.0) as u64;
    tick.0.is_multiple_of(ticks_per_retarget)
}

/// Samples a new `target_position` per flail joint. Joint motors (Avian) do the tracking.
fn apply_joint_flail_targets(
    tick: Res<SimTick>,
    mut joints: Query<(Entity, &mut RevoluteJoint), With<FlailJoint>>,
) {
    for (entity, mut joint) in &mut joints {
        let (min_angle, max_angle) = joint
            .angle_limit
            .map(|limit| (limit.min, limit.max))
            .unwrap_or((-1.0, 1.0));

        let seed = tick
            .0
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(entity.to_bits());
        let target = min_angle + hash_unit(seed) * (max_angle - min_angle);

        joint.motor.target_position = target;
        joint.motor.target_velocity = 0.0;
    }
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
    let layers = env_world_collision_layers(env_id);

    let _root = spawn_env_root(commands, env_id, isolation);

    let half_extents = ground_half_extents(isolation);
    let collider = Collider::cuboid(
        half_extents.x * 2.0,
        half_extents.y * 2.0,
        half_extents.z * 2.0,
    );
    // Center so the top face sits at the env origin height.
    let ground_translation = origin - Vec3::Y * half_extents.y;
    commands.spawn((
        Name::new(format!("ground_{}", env_id.index())),
        DogGroundEnv { env_id },
        SimBody { env_id },
        RigidBody::Static,
        collider,
        layers,
        Friction::new(0.9),
        Transform::from_translation(ground_translation),
    ));

    let dog = dog_quadruped_desc();
    let instance = spawn_creature(commands, env_id, origin, &dog, interpolate);

    for (joint_desc, joint_entity) in dog.joints.iter().zip(instance.joints.iter()) {
        if matches!(joint_desc.kind, JointKind::Revolute { .. }) {
            let joint_entity = *joint_entity;
            commands.entity(joint_entity).insert(FlailJoint);
            commands.queue(move |world: &mut World| {
                let Some(mut joint) = world.get_mut::<RevoluteJoint>(joint_entity) else {
                    return;
                };
                joint.motor = flail_motor();
            });
        }
    }
}

/// Half-thickness of the flat ground cuboid.
pub const GROUND_HALF_THICKNESS: f32 = 0.25;

/// Half-extents for the flat ground cuboid in an env.
pub fn ground_half_extents(isolation: &EnvIsolationConfig) -> Vec3 {
    let half_size = isolation.spacing * 0.9 * 0.5;
    Vec3::new(half_size, GROUND_HALF_THICKNESS, half_size)
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

/// Dog-like quadruped: torso + head + four legs with abduct + flex hips and knees.
pub fn dog_quadruped_desc() -> CreatureDesc {
    let torso_half = Vec3::new(0.35, 0.12, 0.18);
    let upper_leg_length = 0.28;
    let lower_leg_length = 0.28;
    let leg_radius = 0.05;
    let hip_radius = 0.04;
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
        let hip_name = format!("{label}_hip");
        let upper_name = format!("{label}_upper");
        let lower_name = format!("{label}_lower");

        let hip_center = Vec3::new(hip_x, torso_y - hip_down, hip_z);
        let upper_center = Vec3::new(hip_x, torso_y - hip_down - upper_leg_length * 0.5, hip_z);
        let lower_center = Vec3::new(
            hip_x,
            torso_y - hip_down - upper_leg_length - lower_leg_length * 0.5,
            hip_z,
        );

        // Flip abduct axis on the right so positive angle is outward on both sides.
        let abduct_axis = if hip_z >= 0.0 { Vec3::X } else { -Vec3::X };

        dog = dog
            .body(
                BodyDesc::new(hip_name.clone(), BodyShape::Sphere { radius: hip_radius })
                    .density(180.0)
                    .at(hip_center),
            )
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
            // Abduction / adduction (out-in), about the body's forward axis.
            .joint(
                JointDesc::revolute(
                    format!("{label}_hip_abduct"),
                    "torso",
                    hip_name.clone(),
                    Vec3::new(hip_x, -hip_down, hip_z),
                    Vec3::ZERO,
                    abduct_axis,
                )
                .with_angle_limits(-0.35, 0.7),
            )
            // Flexion / extension (forward-back), about the lateral axis.
            .joint(
                JointDesc::revolute(
                    format!("{label}_hip_flex"),
                    hip_name,
                    upper_name.clone(),
                    Vec3::ZERO,
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
