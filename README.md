# bevy-rl

Infrastructure for training RL agents in parallel Bevy + Avian physics sims (Burn PPO).

Many env instances share one Avian world, isolated by spatial separation and collision layers. Policies are trained with PPO (Burn wgpu) and can run in-sim via the viewer.

Bevy-rl ships **crates** (sim, policy, training, viewer, morphology studio). Entry points for train / view / morphology edit live in example packs — start with [`examples/dog`](examples/dog).

## Workspace layout

| Crate | Role |
| --- | --- |
| `sim_core` | Physics core: env isolation, creature articulations, joint target actuation, shared `RlBuffers` / `CreatureSpec` contract |
| `policy` | Burn actor-critic + checkpoints (keyed by creature id) |
| `training` | PPO / GAE / rollout / train dashboard + `run_ppo` helper |
| `sim_viewer` | Generic multi-view client (`run_viewer`) — compose with a creature pack |
| `morphology_studio` | Visual morphology editor (`run_morphology_studio`) — gizmo edit, RON save/load, physics preview |
| `examples/dog` | **Example pack** — quadruped balance + `dog_train` / `dog_view` / `dog_morphology_studio` binaries |

### Creature pack contract

A pack plugs into the helpers by providing:

1. **`CreatureSpec`** — id, observation dim, action dim  
2. **Spawn** — honor `SpawnEnvBatch` / `RespawnAllEnvs` (flat ground helper: `spawn_flat_ground`)  
3. **Step systems** — fill `RlBuffers` observations & rewards; mark `ActuatedRevolute` joints  
4. **Optional** — `ViewerCreatureVisuals` for debug meshes  

Action → joint target is handled by `sim_core` (`ControlSystems::ApplyActions`).

## Run (dog example)

```bash
# Watch envs (pick a policy in the UI to drive them)
cargo run -p dog_view

# Edit dog morphology (gizmos + RON save/load + physics preview)
cargo run -p dog_morphology_studio

# PPO training — Burn TUI shows loss / mean reward
cargo run -p dog_train -- 16 50

# Resume from latest AppData checkpoint
cargo run -p dog_train -- 16 50 --load

# Resume from an explicit checkpoint stem or directory
cargo run -p dog_train -- 16 50 --load "%APPDATA%/bevy-rl/checkpoints/dog/latest"
```

### Viewer policy

In the viewer controls panel:

1. Click **Policy…** and choose a checkpoint (`.mpk` or `.json`). The dialog opens in the creature’s checkpoint folder when it exists (or can be created).
2. Creatures are driven each physics tick with the policy’s deterministic mean actions (shared weights across all visible envs).
3. Click **Clear** to unload the policy and return to zero torque.

Dim / creature mismatches log loudly and leave the policy unloaded.

## Checkpoints

Trainer saves once at the end of the run (or after an early TUI stop) to the OS app data directory (not the repo):

- Windows: `%APPDATA%\bevy-rl\checkpoints\dog\`
- macOS: `~/Library/Application Support/bevy-rl/checkpoints/dog/`
- Linux: `~/.local/share/bevy-rl/checkpoints/dog/`

Each save writes inference-ready weights (`.mpk`) plus JSON metadata:

- `latest.mpk` / `latest.json`
- `step_000042.mpk` / `step_000042.json` (final update index)

Metadata includes creature id, obs/action/hidden dims, update index, and mean reward. Load fails if dims do not match.

During training, Burn's terminal UI shows live **Loss**, **Mean Reward**, **Mean Episode Return**, and **Update Time** (seconds per PPO update) graphs (arrow keys switch metrics / plot types; `q` then `s` stops cleanly and still saves).

## Design notes

- Sims tick at 60 Hz physics; the policy runs at 20 Hz (actions held for 3 physics steps).
- Dog (example) observations: projected gravity, root lin/ang vel, joint angles + ang vels, height, foot contacts, previous actions (50-D).
- Dog actions: 12 normalized joint target angles.
- Episodes are fixed-horizon; balance reward is upright + height + foot stance, with a fall penalty on terminal steps.
