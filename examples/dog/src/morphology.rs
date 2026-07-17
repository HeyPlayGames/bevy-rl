use bevy::prelude::*;
use sim_core::prelude::*;

/// Observation: projected gravity (3) + root lin/ang vel (6) + joint angles (12) + joint ang vels (12) + torso height (1) + previous actions (12).
pub const DOG_OBS_DIM: usize = 46;
/// One normalized target-angle command per revolute leg joint.
pub const DOG_ACTION_DIM: usize = 12;

/// Stable action order for dog revolute joints.
pub fn actuated_joint_names() -> [&'static str; DOG_ACTION_DIM] {
    [
        "fl_hip_abduct",
        "fl_hip_flex",
        "fl_knee",
        "fr_hip_abduct",
        "fr_hip_flex",
        "fr_knee",
        "bl_hip_abduct",
        "bl_hip_flex",
        "bl_knee",
        "br_hip_abduct",
        "br_hip_flex",
        "br_knee",
    ]
}

/// Dog-like quadruped: torso + four legs with abduct + flex hips and knees.
pub fn dog_quadruped_desc() -> CreatureDesc {
    let torso_half = Vec3::new(0.35, 0.12, 0.18);
    let upper_leg_length = 0.28;
    let lower_leg_length = 0.28;
    let leg_radius = 0.05;
    let hip_radius = 0.04;
    // Feet near ground: lower capsule bottom ≈ torso_y - hip_down - upper - lower - radius.
    let torso_y = 0.73;
    let hip_forward = 0.22;
    let hip_side = 0.16;
    let hip_down = 0.12;

    let legs = [
        ("fl", hip_forward, hip_side),
        ("fr", hip_forward, -hip_side),
        ("bl", -hip_forward, hip_side),
        ("br", -hip_forward, -hip_side),
    ];

    let mut dog = CreatureDesc::new("dog").body(
        BodyDesc::new(
            "torso",
            BodyShape::Cuboid {
                half_extents: torso_half,
            },
        )
        .density(250.0)
        .at(Vec3::new(0.0, torso_y, 0.0)),
    );

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
            .joint(
                JointDesc::revolute(
                    format!("{label}_hip_abduct"),
                    "torso",
                    hip_name.clone(),
                    Vec3::new(hip_x, -hip_down, hip_z),
                    Vec3::ZERO,
                    abduct_axis,
                )
                .with_angle_limits(-0.1, 0.3),
            )
            .joint(
                JointDesc::revolute(
                    format!("{label}_hip_flex"),
                    hip_name,
                    upper_name.clone(),
                    Vec3::ZERO,
                    Vec3::new(0.0, upper_leg_length * 0.5, 0.0),
                    Vec3::Z,
                )
                .with_angle_limits(-0.3, 0.5),
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
                .with_angle_limits(0.0, 1.0),
            );
    }

    dog
}
