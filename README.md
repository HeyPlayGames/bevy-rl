# sim_batch

Parallel headless physics simulation environments on Bevy 0.19 + Avian 3D.

Many env instances share one Avian world, isolated by spatial separation and collision layers. A viewer can watch ~4 envs with transform interpolation between fixed ticks.

## Crates

| Crate | Role |
| --- | --- |
| `sim_core` | Fixed timestep, `EnvId`, isolation layers, creature articulation format + spawner |
| `sim_envs` | Concrete envs (flat ground + dog-like quadruped) |
| `sim_viewer` | Windowed multi-view client |
| `sim_headless` | Throughput-first batch runner |
| `sim_viewer` (bin) | Opens the viewer |

## Run

```bash
# 16 envs, 600 fixed ticks, as fast as possible
cargo run -p sim_headless -- 16 600

# Watch envs 0–3 (8 total simulated)
cargo run -p sim_viewer_bin
```

## Design notes

- Sims tick in `FixedPostUpdate` (Avian default) at 60 Hz.
- Headless uses `TimeUpdateStrategy::ManualDuration` so wall clock does not throttle throughput.
- Isolation: env origins on a grid (`spacing` default 40) + `CollisionLayers` from `env_id % 31`.
- Creatures are physics-first articulations (capsules/cuboids + revolute/spherical joints). Visuals are debug meshes matching colliders.
- Viewer: 2×2 cameras with viewports + `TransformInterpolation` on dynamic bodies (render rate ≠ sim rate).
