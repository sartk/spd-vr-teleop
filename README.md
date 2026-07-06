# mujoco-vr-teleop

MuJoCo VR teleoperation with scene-builder based task construction, WebXR streaming, trajectory recording, and open-loop replay.

For operator setup and data collection instructions, see [Getting Started: VR Data Collection](docs/getting-started.md).

## TL;DR — collector session

```bash
# 1. one-time setup
./setup.sh all
npm install --prefix frontend
npm run build --prefix frontend

# 2. forward port 8012 from the Quest over USB
adb reverse tcp:8012 tcp:8012

# 3. launch the TUI in collector mode and pick a scene (jenga / spell_and_stow / dishrack / ...)
uv run python -m mujoco_vr_teleop.tui --mode collector
```

Open `http://localhost:8012/` on the headset's browser. Pedals: A = save
progress, B = pause/resume, C tap = revert to last save, C hold = skip task.

## Installation

Requires `uv`, Node, and npm.

```bash
./setup.sh all
npm install --prefix frontend
```

Use `./setup.sh ik` (default, same as `all`) for the VR streamer plus mink-based IK, or `./setup.sh none` for a base install with no extras.

## Scene Building

Save an inspectable XML with the scene-builder CLI:

```bash
uv run python -m mujoco_vr_teleop.build_scene \
  --scene-type jenga \
  --arm-type yam_ultra
```

This writes the XML and config under `examples/generated/` and prints the XML path. Scene type and arm type are explicit; the builder defaults to Sharpa hands, IK wrist control, and finger IK. Normal streaming builds the scene directly from the same nested builder config, so you do not need to pass an XML path.

Use builder overrides such as `--scene-type dishrack` or `--wrist-control hybrid_weld` when you want a different scene or wrist mode.

Domain randomization is enabled by default in the streamer: each scene's task reset owns scaling and placement (variant pools, scatter resets). Randomization is applied on startup and hard reset; each episode saves a full model snapshot in the zarr so replay reconstructs the exact recorded scene.

## VR Streaming

Click the notification that pops up to allow developer USB access on Quest, and run `adb reverse tcp:8012 tcp:8012`.

Build the frontend, then start the supervisor TUI:

```bash
npm run build --prefix frontend && \
uv run mujoco-vr-tui --builder.arm-type yam_ultra
```

The TUI shows a scene picker on first launch (jenga, spell_and_stow, dishrack, hang_mugs, bottles_in_bin, cleanup, cup_stack, cup_ball, cups_and_mugs, tea_time, shapes, nut_and_bolt). Press `W` mid-session to switch scenes. The supervisor restarts the streamer subprocess with the chosen scene; the frontend reloads its assets automatically. To skip the picker, pass the scene up front:

```bash
uv run mujoco-vr-tui --builder.scene-type jenga --builder.arm-type yam_ultra
```

If you want to run the streamer without the supervisor:

```bash
uv run mujoco-vr-stream --builder.scene-type jenga --builder.arm-type yam_ultra
```

Defaults: `sharpa` hands, mink IK for wrists and fingers, foot-pedal control. To use hybrid wrist welds instead of pure IK wrists:

```bash
uv run mujoco-vr-tui --builder.wrist-control hybrid_weld --builder.arm-type yam_ultra
```

Hybrid wrist mode enters weld when current wrist error is below `--weld-enter-thresh` (default 0.03 m); in hybrid wrist mode the recorded `ctrl` arm slots are overwritten with delayed future IK targets.

Open `http://<machine-ip>:8012` in the headset browser and tap "Enter VR" or "Enter XR". The session boots in **playground mode**: tracking is paused, no recording. Tap `B` to start tracking (still no recording — practice freely). Hold `B` for 1.5s to leave playground and start recording. While in playground, hold `A` to lower the table height and `C` to raise it; the new height persists across runs in `~/.vr_streamer/table_height.json`. Once recording: `A` saves a checkpoint (ignored if a hand is touching an object), `B` toggles pause, `C` discards back to the last checkpoint, hold `C` 2s to hard-reset.

## Replay

Try the checked-in example trajectory first:

```bash
uv run python -m mujoco_vr_teleop.replay_trajectory_open_loop \
  examples/trajectories/20260518_0653/ep_00000/ \
  --mp4-out examples/trajectories/20260518_0653/ep_00000/replay_top.mp4 \
  --camera top
```

For a newly recorded session, replace the trajectory path with the saved episode folder, for example:

```bash
uv run python -m mujoco_vr_teleop.replay_trajectory_open_loop \
  data/default/20260518_0653/ep_00000/ \
  --mp4-out data/default/20260518_0653/ep_00000/replay_top.mp4 \
  --camera top
```

### Kinematic replay

By default, replay restores the start frame once and then steps physics
forward using the recorded actuator commands ("open-loop action rollout").
This is useful for evaluating dynamics, but small numerical differences mean
the replayed contact/dynamics state drifts from the original recording.

For a deterministic playback that reproduces the recording exactly, pass
`--kinematic`. At every frame the replay sets `data.qpos` / `data.qvel` from
the recording and calls `mj_forward` — no physics integration, no actuator
dynamics, no contact-driven drift. Use this for visualization, dataset
inspection, or as a baseline against open-loop replay.

```bash
uv run python -m mujoco_vr_teleop.replay_trajectory_open_loop \
  examples/trajectories/20260518_0653/ep_00000/ \
  --kinematic --no-viser \
  --mp4-out examples/trajectories/20260518_0653/ep_00000/replay_kinematic.mp4 \
  --camera top
```
