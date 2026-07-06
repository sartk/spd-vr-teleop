"""Open-loop replay for recorded zarr trajectories.

The replay restores the recorded start state once, then advances MuJoCo by
feeding recorded actions. It does not overwrite qpos after the selected start
frame.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import mujoco
import numpy as np

from mujoco_vr_teleop.replay_scene import (
    load_model_from_recording,
    resolve_episode_zarr_path,
)
from mujoco_vr_teleop.trajectory_filters import (
    HandObjectContactFilterResult,
    compute_hand_object_contact_filter,
    contact_filter_to_cache_dict,
    load_contact_filter_cache,
    save_contact_filter_cache,
)

try:
    import zarr
except ImportError:
    raise SystemExit("Missing zarr. Install with: pip install zarr")


ARM_SIDE_ORDER = ("right", "left")
ARM_JOINT_COUNT = 6
HAND_JOINT_COUNT = 22
HAND_SKELETON_EDGES = (
    ("wrist", "thumb-phalanx-proximal"),
    ("thumb-phalanx-proximal", "thumb-phalanx-distal"),
    ("thumb-phalanx-distal", "thumb-tip"),
    ("wrist", "index-finger-phalanx-proximal"),
    ("index-finger-phalanx-proximal", "index-finger-phalanx-intermediate"),
    ("index-finger-phalanx-intermediate", "index-finger-phalanx-distal"),
    ("index-finger-phalanx-distal", "index-finger-tip"),
    ("wrist", "middle-finger-phalanx-proximal"),
    ("middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate"),
    ("middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal"),
    ("middle-finger-phalanx-distal", "middle-finger-tip"),
    ("wrist", "ring-finger-phalanx-proximal"),
    ("ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate"),
    ("ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal"),
    ("ring-finger-phalanx-distal", "ring-finger-tip"),
    ("wrist", "pinky-finger-phalanx-proximal"),
    ("pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate"),
    ("pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal"),
    ("pinky-finger-phalanx-distal", "pinky-finger-tip"),
)


def _load_array(group, name: str) -> Any | None:
    return group[name] if name in group else None


def _array_len(arr: Any | None) -> int | None:
    return None if arr is None else int(arr.shape[0])


def _frame_dt(sim_time: Any, fps_override: float | None) -> float:
    if fps_override is not None:
        if fps_override <= 0.0:
            raise SystemExit("--fps must be > 0")
        return 1.0 / fps_override
    diffs = np.diff(np.asarray(sim_time, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    assert len(diffs)
    return float(np.median(diffs))


def _load_arrays(traj) -> dict[str, Any | None]:
    arrays = {
        "qpos": traj["qpos"],
        "qvel": traj["qvel"],
        "ctrl": traj["ctrl"],
        "mocap_pos": _load_array(traj, "mocap_pos"),
        "mocap_quat": _load_array(traj, "mocap_quat"),
        "arm_ik_target_pos": _load_array(traj, "arm_ik_target_pos"),
        "arm_ik_target_quat_wxyz": _load_array(traj, "arm_ik_target_quat_wxyz"),
        "arm_ik_solution_qpos": _load_array(traj, "arm_ik_solution_qpos"),
        "hand_ik_solution_qpos": _load_array(traj, "hand_ik_solution_qpos"),
        "arm_joint_qpos": _load_array(traj, "arm_joint_qpos"),
        "arm_joint_qvel": _load_array(traj, "arm_joint_qvel"),
        "arm_ctrl": _load_array(traj, "arm_ctrl"),
        "hand_joint_qpos": _load_array(traj, "hand_joint_qpos"),
        "hand_joint_qvel": _load_array(traj, "hand_joint_qvel"),
        "hand_ctrl": _load_array(traj, "hand_ctrl"),
        "sim_time": np.asarray(traj["sim_time"], dtype=float),
        "physics_step": np.asarray(traj["physics_step"], dtype=np.int64),
    }
    lengths = [
        length
        for length in (_array_len(arr) for arr in arrays.values())
        if length is not None
    ]
    if not lengths:
        raise SystemExit("Trajectory contains no replayable arrays")
    frame_count = min(lengths)
    arrays["frame_count"] = frame_count
    return arrays


def _validate_shapes(model: mujoco.MjModel, arrays: dict[str, Any | None]) -> None:
    qpos = arrays.get("qpos")
    if qpos is None:
        raise SystemExit("Open-loop replay requires qpos to initialize the start frame")
    if int(qpos.shape[1]) != model.nq:
        raise SystemExit(f"qpos width {qpos.shape[1]} does not match model.nq={model.nq}")

    qvel = arrays.get("qvel")
    if qvel is not None and int(qvel.shape[1]) != model.nv:
        raise SystemExit(f"qvel width {qvel.shape[1]} does not match model.nv={model.nv}")

    ctrl = arrays.get("ctrl")
    if ctrl is not None and int(ctrl.shape[1]) != model.nu:
        raise SystemExit(f"ctrl width {ctrl.shape[1]} does not match model.nu={model.nu}")

    sim_time = arrays["sim_time"]
    if int(sim_time.shape[0]) < 2:
        raise SystemExit("Open-loop replay requires at least two sim_time samples")

    physics_step = arrays["physics_step"]
    if int(physics_step.shape[0]) < 2:
        raise SystemExit("Open-loop replay requires at least two physics_step samples")
    step_diffs = np.diff(np.asarray(physics_step, dtype=np.int64))
    if np.any(step_diffs <= 0):
        raise SystemExit("physics_step must be strictly increasing")

    mocap_pos = arrays.get("mocap_pos")
    if mocap_pos is not None and tuple(mocap_pos.shape[1:]) != (model.nmocap, 3):
        raise SystemExit(f"mocap_pos shape {mocap_pos.shape} does not match model.nmocap={model.nmocap}")

    mocap_quat = arrays.get("mocap_quat")
    if mocap_quat is not None and tuple(mocap_quat.shape[1:]) != (model.nmocap, 4):
        raise SystemExit(f"mocap_quat shape {mocap_quat.shape} does not match model.nmocap={model.nmocap}")

    for name, shape in {
        "arm_ik_target_pos": (len(ARM_SIDE_ORDER), 3),
        "arm_ik_target_quat_wxyz": (len(ARM_SIDE_ORDER), 4),
        "arm_ik_solution_qpos": (len(ARM_SIDE_ORDER), ARM_JOINT_COUNT),
        "hand_ik_solution_qpos": (len(ARM_SIDE_ORDER), HAND_JOINT_COUNT),
        "arm_joint_qpos": (len(ARM_SIDE_ORDER), ARM_JOINT_COUNT),
        "arm_joint_qvel": (len(ARM_SIDE_ORDER), ARM_JOINT_COUNT),
        "arm_ctrl": (len(ARM_SIDE_ORDER), ARM_JOINT_COUNT),
        "hand_joint_qpos": (len(ARM_SIDE_ORDER), HAND_JOINT_COUNT),
        "hand_joint_qvel": (len(ARM_SIDE_ORDER), HAND_JOINT_COUNT),
        "hand_ctrl": (len(ARM_SIDE_ORDER), HAND_JOINT_COUNT),
    }.items():
        arr = arrays.get(name)
        if arr is not None and tuple(arr.shape[1:]) != shape:
            raise SystemExit(f"{name} shape {arr.shape} does not match expected (*, {shape})")


def _window(frame_count: int, start_frame: int, num_steps: int | None) -> tuple[int, int]:
    start = min(max(0, int(start_frame)), max(0, frame_count - 1))
    count = frame_count - start if num_steps is None else int(num_steps)
    count = min(max(1, count), frame_count - start)
    return start, start + count


def _arm_joint_addresses(model: mujoco.MjModel) -> dict[str, list[tuple[int, int]]]:
    addresses: dict[str, list[tuple[int, int]]] = {}
    for side in ("right", "left"):
        prefix = f"{side}-arm-"
        side_addresses = []
        for idx in range(1, ARM_JOINT_COUNT + 1):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{idx}")
            if jid < 0:
                side_addresses = []
                break
            side_addresses.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
        if side_addresses:
            addresses[side] = side_addresses
    return addresses


def _actuator_for_joint(model: mujoco.MjModel, joint_id: int) -> int:
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id][0]) == joint_id:
            return actuator_id
    return -1


def _arm_actuator_ids(model: mujoco.MjModel) -> dict[str, list[int]]:
    actuators: dict[str, list[int]] = {}
    for side in ("right", "left"):
        prefix = f"{side}-arm-"
        side_actuators = []
        for idx in range(1, ARM_JOINT_COUNT + 1):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{idx}")
            actuator_id = _actuator_for_joint(model, joint_id) if joint_id >= 0 else -1
            if actuator_id < 0:
                side_actuators = []
                break
            side_actuators.append(actuator_id)
        if side_actuators:
            actuators[side] = side_actuators
    return actuators


def _hand_joint_addresses(model: mujoco.MjModel) -> dict[str, list[tuple[int, int]]]:
    addresses: dict[str, list[tuple[int, int]]] = {}
    finger_tokens = ("thumb", "index", "middle", "ring", "pinky")
    for side in ARM_SIDE_ORDER:
        prefix = f"{side}_"
        side_addresses = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id][0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if not joint_name.startswith(prefix):
                continue
            if not any(token in joint_name for token in finger_tokens):
                continue
            side_addresses.append((int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])))
        if side_addresses:
            addresses[side] = side_addresses
    return addresses


def _hand_actuator_ids(model: mujoco.MjModel) -> dict[str, list[int]]:
    actuators: dict[str, list[int]] = {}
    finger_tokens = ("thumb", "index", "middle", "ring", "pinky")
    for side in ARM_SIDE_ORDER:
        prefix = f"{side}_"
        side_actuators = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id][0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if joint_name.startswith(prefix) and any(token in joint_name for token in finger_tokens):
                side_actuators.append(actuator_id)
        if side_actuators:
            actuators[side] = side_actuators
    return actuators


def _interp_solution(sim_time: np.ndarray, values: Any, query_time: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] < 2:
        return arr[0].copy()
    if query_time <= sim_time[0]:
        return arr[0].copy()
    elif query_time >= sim_time[-1]:
        lo, hi, alpha = len(sim_time) - 2, len(sim_time) - 1, 1.0
    else:
        hi = int(np.searchsorted(sim_time, query_time, side="right"))
        lo = max(0, hi - 1)
        hi = min(hi, len(sim_time) - 1)
        denom = max(float(sim_time[hi] - sim_time[lo]), 1e-9)
        alpha = float((query_time - sim_time[lo]) / denom)
    denom = max(float(sim_time[hi] - sim_time[lo]), 1e-9)
    out = (1.0 - alpha) * arr[lo] + alpha * arr[hi]
    return np.where(np.isfinite(out), out, arr[lo])


def _scale_actuator_damping(model: mujoco.MjModel, actuator_ids: list[int], mult: float) -> None:
    for actuator_id in actuator_ids:
        model.actuator_biasprm[actuator_id, 2] *= float(mult)


def _scale_actuator_stiffness(model: mujoco.MjModel, actuator_ids: list[int], mult: float) -> None:
    for actuator_id in actuator_ids:
        model.actuator_gainprm[actuator_id, 0] *= float(mult)
        model.actuator_biasprm[actuator_id, 1] *= float(mult)


def _apply_actuator_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    values: np.ndarray,
) -> None:
    command = np.asarray(values, dtype=float).copy()
    for actuator_id, value in zip(actuator_ids, command):
        if not np.isfinite(value):
            continue
        if model.actuator_ctrllimited[actuator_id]:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            value = float(np.clip(value, lo, hi))
        data.ctrl[actuator_id] = value


def _apply_ik_control_override(model: mujoco.MjModel, data: mujoco.MjData, override: dict[str, Any]) -> None:
    current_ctrl = override.get("current_ctrl")
    if current_ctrl is not None:
        data.ctrl[override["actuator_ids"]] = current_ctrl


def _update_ik_control_override(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    override: dict[str, Any] | None,
    replay_time: float,
) -> None:
    if override is None:
        return
    while replay_time + 1e-12 >= override["next_update_time"]:
        target_time = override["next_update_time"] - override["latency_seconds"]
        ctrl = data.ctrl[override["actuator_ids"]].copy()
        arm_values = _interp_solution(override["sim_time"], override["arm_solution"], target_time)
        for side_idx, side in enumerate(ARM_SIDE_ORDER):
            _apply_actuator_values(
                model,
                data,
                override["arm_actuator_ids"].get(side, []),
                arm_values[side_idx],
            )
        hand_solution = override.get("hand_solution")
        if hand_solution is not None:
            hand_values = _interp_solution(override["sim_time"], hand_solution, target_time)
            for side_idx, side in enumerate(ARM_SIDE_ORDER):
                _apply_actuator_values(
                    model,
                    data,
                    override["hand_actuator_ids"].get(side, []),
                    hand_values[side_idx],
                )
        ctrl[:] = data.ctrl[override["actuator_ids"]]
        override["current_ctrl"] = ctrl
        override["next_update_time"] += override["control_dt"]


def _reset_ik_control_override(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    override: dict[str, Any] | None,
    frame_idx: int,
) -> None:
    if override is None:
        return
    sim_time = override["sim_time"]
    frame_time = float(sim_time[frame_idx])
    tick = np.floor((frame_time - sim_time[0]) / override["control_dt"])
    override["next_update_time"] = float(sim_time[0] + max(0.0, tick) * override["control_dt"])
    override["current_ctrl"] = None
    _update_ik_control_override(model, data, override, frame_time)


def _build_ik_control_override(
    model: mujoco.MjModel,
    arrays: dict[str, Any | None],
    arm_actuator_ids: dict[str, list[int]],
    hand_actuator_ids: dict[str, list[int]],
    control_rate: float | None,
    latency_steps: int,
    stiffness_mult: float,
    damping_mult: float,
) -> dict[str, Any] | None:
    if control_rate is None:
        return None
    arm_solution = arrays.get("arm_ik_solution_qpos")
    if arm_solution is None:
        raise SystemExit("--ik-control-rate-override requires trajectory/arm_ik_solution_qpos")
    if not arm_actuator_ids:
        raise SystemExit("--ik-control-rate-override could not find arm actuators in the model")
    if control_rate <= 0.0:
        raise SystemExit("--ik-control-rate-override must be > 0")
    if latency_steps < 0:
        raise SystemExit("--ik-control-latency-override must be >= 0")
    if stiffness_mult < 0.0:
        raise SystemExit("--ik-control-stiffness-mult must be >= 0")
    if damping_mult < 0.0:
        raise SystemExit("--ik-control-damping-mult must be >= 0")
    actuator_ids = sorted(
        {
            actuator_id
            for ids in list(arm_actuator_ids.values()) + list(hand_actuator_ids.values())
            for actuator_id in ids
        }
    )
    _scale_actuator_stiffness(model, actuator_ids, stiffness_mult)
    _scale_actuator_damping(model, actuator_ids, damping_mult)
    return {
        "sim_time": np.asarray(arrays["sim_time"], dtype=float),
        "arm_solution": arm_solution,
        "hand_solution": arrays.get("hand_ik_solution_qpos"),
        "arm_actuator_ids": arm_actuator_ids,
        "hand_actuator_ids": hand_actuator_ids,
        "actuator_ids": np.asarray(actuator_ids, dtype=int),
        "control_dt": 1.0 / float(control_rate),
        "latency_seconds": float(latency_steps) / float(control_rate),
        "stiffness_mult": float(stiffness_mult),
        "damping_mult": float(damping_mult),
        "next_update_time": 0.0,
        "current_ctrl": None,
    }


def _set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    frame_idx: int,
    ik_control_override: dict[str, Any] | None = None,
) -> None:
    data.qpos[:] = np.asarray(arrays["qpos"][frame_idx], dtype=float)
    qvel = arrays.get("qvel")
    data.qvel[:] = np.asarray(qvel[frame_idx], dtype=float) if qvel is not None else 0.0
    _set_action(model, data, arrays, frame_idx, ik_control_override)
    _reset_ik_control_override(model, data, arrays, ik_control_override, frame_idx)
    mujoco.mj_forward(model, data)


def _forward_visual_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def _set_visual_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    frame_idx: int,
    ik_control_override: dict[str, Any] | None = None,
) -> None:
    data.qpos[:] = np.asarray(arrays["qpos"][frame_idx], dtype=float)
    qvel = arrays.get("qvel")
    data.qvel[:] = np.asarray(qvel[frame_idx], dtype=float) if qvel is not None else 0.0
    _set_action(model, data, arrays, frame_idx, ik_control_override)
    _reset_ik_control_override(model, data, arrays, ik_control_override, frame_idx)
    _forward_visual_kinematics(model, data)


def _apply_future_arm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    frame_idx: int,
    future_arm: dict[str, Any],
) -> None:
    """Overwrite arm actuator ctrl with recorded arm_joint_qpos from a future frame."""
    arm_joint_qpos = arrays.get("arm_joint_qpos")
    if arm_joint_qpos is None:
        return
    n_frames = _array_len(arm_joint_qpos) or 0
    if n_frames == 0:
        return
    target_idx = min(frame_idx + future_arm["offset"], n_frames - 1)
    future_values = np.asarray(arm_joint_qpos[target_idx], dtype=float)
    for side_idx, side in enumerate(ARM_SIDE_ORDER):
        _apply_actuator_values(
            model,
            data,
            future_arm["arm_actuator_ids"].get(side, []),
            future_values[side_idx],
        )


def _make_future_arm(
    episode: "ReplayEpisode", offset: int
) -> dict[str, Any] | None:
    """Build a future-arm spec, or None if disabled / unavailable for this episode."""
    if offset <= 0:
        return None
    if not episode.arm_actuator_ids:
        print("--future-arm-offset: no arm actuators in model; ignoring")
        return None
    if episode.arrays.get("arm_joint_qpos") is None:
        print("--future-arm-offset: trajectory has no arm_joint_qpos; ignoring")
        return None
    return {"arm_actuator_ids": episode.arm_actuator_ids, "offset": int(offset)}


def _set_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    frame_idx: int,
    ik_control_override: dict[str, Any] | None = None,
    future_arm: dict[str, Any] | None = None,
) -> None:
    ctrl = arrays.get("ctrl")
    if ctrl is not None:
        data.ctrl[:] = np.asarray(ctrl[frame_idx], dtype=float)
    elif model.nu > 0:
        data.ctrl[:] = 0.0

    mocap_pos = arrays.get("mocap_pos")
    if mocap_pos is not None and model.nmocap > 0:
        data.mocap_pos[:] = np.asarray(mocap_pos[frame_idx], dtype=float)

    mocap_quat = arrays.get("mocap_quat")
    if mocap_quat is not None and model.nmocap > 0:
        data.mocap_quat[:] = np.asarray(mocap_quat[frame_idx], dtype=float)
    if ik_control_override is not None:
        _apply_ik_control_override(model, data, ik_control_override)
    if future_arm is not None:
        _apply_future_arm(model, data, arrays, frame_idx, future_arm)


def _step_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    frame_idx: int,
    ik_control_override: dict[str, Any] | None = None,
    future_arm: dict[str, Any] | None = None,
) -> None:
    _set_action(model, data, arrays, frame_idx, ik_control_override, future_arm)
    physics_steps = np.asarray(arrays["physics_step"], dtype=np.int64)
    n_steps = int(physics_steps[frame_idx] - physics_steps[frame_idx - 1])
    assert n_steps > 0
    sim_time = np.asarray(arrays["sim_time"], dtype=float)
    step_dt = float((sim_time[frame_idx] - sim_time[frame_idx - 1]) / n_steps)
    for step_idx in range(n_steps):
        replay_time = float(sim_time[frame_idx - 1] + step_idx * step_dt)
        _update_ik_control_override(model, data, ik_control_override, replay_time)
        if future_arm is not None:
            _apply_future_arm(model, data, arrays, frame_idx, future_arm)
        mujoco.mj_step(model, data)


def _camera_spec(camera: str | None):
    if camera is None or str(camera).strip() == "":
        return None
    value = str(camera).strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _render_mp4(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    *,
    start: int,
    end: int,
    fps: float,
    out_path: Path,
    width: int,
    height: int,
    camera,
    ik_control_override: dict[str, Any] | None,
    kinematic: bool = False,
    future_arm: dict[str, Any] | None = None,
) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    error: Exception | None = None
    try:
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        if kinematic:
            _set_visual_state(model, data, arrays, start, ik_control_override)
        else:
            _set_state(model, data, arrays, start, ik_control_override)
        total_frames = end - start
        render_t0 = time.perf_counter()
        last_report = render_t0
        for frame_idx in range(start, end):
            if frame_idx > start:
                if kinematic:
                    _set_visual_state(
                        model, data, arrays, frame_idx, ik_control_override
                    )
                else:
                    _step_action(
                        model, data, arrays, frame_idx, ik_control_override, future_arm
                    )
            renderer.update_scene(data, camera=camera)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            # Progress: report at most ~once a second so long renders aren't
            # silent, plus a final 100% line.
            done = frame_idx - start + 1
            now = time.perf_counter()
            if now - last_report >= 1.0 or done == total_frames:
                fps_rate = done / max(now - render_t0, 1e-6)
                eta = (total_frames - done) / max(fps_rate, 1e-6)
                print(
                    f"  render {done}/{total_frames} "
                    f"({100.0 * done / total_frames:.0f}%) "
                    f"{fps_rate:.1f} fps, ETA {eta:.0f}s",
                    flush=True,
                )
                last_report = now
    except Exception as exc:
        error = exc
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
        returncode = proc.wait()
        if hasattr(renderer, "close"):
            renderer.close()

    if error is not None:
        raise error
    if returncode != 0:
        detail = f": {stderr}" if stderr else ""
        raise SystemExit(f"ffmpeg exited with code {returncode}{detail}")


def _mocap_names_by_id(model: mujoco.MjModel) -> dict[int, str]:
    names = {}
    for body_id in range(model.nbody):
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            names[mocap_id] = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"mocap_{mocap_id}"
            )
    return names


def _mocap_skeleton_edges(mocap_names: dict[int, str]) -> list[tuple[int, int]]:
    id_by_name = {name: mocap_id for mocap_id, name in mocap_names.items()}
    edges = []
    for side in ARM_SIDE_ORDER:
        prefix = f"{side}-"
        for a, b in HAND_SKELETON_EDGES:
            ia = id_by_name.get(prefix + a)
            ib = id_by_name.get(prefix + b)
            if ia is not None and ib is not None:
                edges.append((ia, ib))
    return edges


def _draw_mocap_skeleton(
    server,
    model: mujoco.MjModel,
    arrays: dict[str, Any | None],
    frame_idx: int,
    mocap_names: dict[int, str],
    mocap_edges: list[tuple[int, int]],
) -> None:
    mocap_pos = arrays.get("mocap_pos")
    if mocap_pos is None or model.nmocap == 0:
        return
    points = np.asarray(mocap_pos[frame_idx], dtype=np.float32)
    colors = np.full((model.nmocap, 3), 230, dtype=np.uint8)
    for mocap_id, name in mocap_names.items():
        if name.startswith("right-"):
            colors[mocap_id] = np.array([255, 120, 60], dtype=np.uint8)
        elif name.startswith("left-"):
            colors[mocap_id] = np.array([80, 180, 255], dtype=np.uint8)
    server.scene.add_point_cloud(
        "/replay/mocap_points",
        points=points,
        colors=colors,
        point_size=0.008,
        point_shape="circle",
        precision="float32",
    )
    if mocap_edges:
        segments = np.asarray([[points[a], points[b]] for a, b in mocap_edges], dtype=np.float32)
        server.scene.add_line_segments(
            "/replay/mocap_skeleton",
            points=segments,
            colors=(245, 245, 245),
            line_width=2.0,
        )


def _ik_overlay_body(model: mujoco.MjModel, body_id: int) -> bool:
    if int(model.body_mocapid[body_id]) >= 0:
        return False
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return (
        name.startswith("right-arm")
        or name.startswith("left-arm")
        or name.startswith("right_")
        or name.startswith("left_")
        or "sharpa" in name
    )


def _blue_overlay_mesh(model: mujoco.MjModel, geom_ids: list[int]):
    from mjviser.scene import merge_geoms
    from trimesh.visual import ColorVisuals

    mesh = merge_geoms(model, geom_ids)
    mesh.visual = ColorVisuals(
        mesh,
        face_colors=np.tile(
            np.array([55, 150, 255, 85], dtype=np.uint8),
            (len(mesh.faces), 1),
        ),
    )
    return mesh


def _create_blue_body_overlay(
    server,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    namespace: str,
) -> dict[int, list[Any]]:
    body_geoms: dict[int, list[int]] = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if not _ik_overlay_body(model, body_id):
            continue
        if int(model.geom_group[geom_id]) >= 3 or float(model.geom_rgba[geom_id, 3]) == 0.0:
            continue
        body_geoms.setdefault(body_id, []).append(geom_id)

    handles: dict[int, list[Any]] = {}
    for body_id, geom_ids in body_geoms.items():
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        handle = server.scene.add_mesh_trimesh(
            f"{namespace}/{body_name}",
            _blue_overlay_mesh(model, geom_ids),
            position=data.xpos[body_id],
            wxyz=data.xquat[body_id],
            cast_shadow=False,
            receive_shadow=False,
        )
        handles.setdefault(body_id, []).append(handle)
    return handles


def _create_ik_overlay(server, model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, list[Any]]:
    return _create_blue_body_overlay(server, model, data, "/ik_overlay")


def _set_ik_overlay_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arrays: dict[str, Any | None],
    frame_idx: int,
    arm_addresses: dict[str, list[tuple[int, int]]],
    hand_addresses: dict[str, list[tuple[int, int]]],
) -> None:
    solution = arrays.get("arm_ik_solution_qpos")
    if solution is None:
        raise SystemExit("--visualize-ik requires trajectory/arm_ik_solution_qpos")
    data.qpos[:] = np.asarray(arrays["qpos"][frame_idx], dtype=float)
    data.qvel[:] = 0.0
    values = np.asarray(solution[frame_idx], dtype=float)
    for side_idx, side in enumerate(ARM_SIDE_ORDER):
        for joint_idx, (qadr, dadr) in enumerate(arm_addresses.get(side, [])):
            value = values[side_idx, joint_idx]
            if np.isfinite(value):
                data.qpos[qadr] = value
                data.qvel[dadr] = 0.0
    hand_solution = arrays.get("hand_ik_solution_qpos")
    if hand_solution is not None:
        hand_values = np.asarray(hand_solution[frame_idx], dtype=float)
        for side_idx, side in enumerate(ARM_SIDE_ORDER):
            for joint_idx, (qadr, dadr) in enumerate(hand_addresses.get(side, [])):
                if joint_idx >= hand_values.shape[1]:
                    break
                value = hand_values[side_idx, joint_idx]
                if np.isfinite(value):
                    data.qpos[qadr] = value
                    data.qvel[dadr] = 0.0
    _forward_visual_kinematics(model, data)


def _update_blue_overlay(handles: dict[int, list[Any]], data: mujoco.MjData) -> None:
    for body_id, body_handles in handles.items():
        for handle in body_handles:
            handle.position = data.xpos[body_id]
            handle.wxyz = data.xquat[body_id]


WRIST_BODY_NAMES = ("right-wrist", "left-wrist")


def _wrist_body_ids(model: mujoco.MjModel) -> list[int]:
    """Body ids for the wrist end-effectors, skipping any not present."""
    ids = []
    for name in WRIST_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(int(bid))
    return ids


def _quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle (degrees) between two wxyz quaternions."""
    a = np.asarray(q_a, dtype=float)
    b = np.asarray(q_b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = float(np.clip(abs(np.dot(a / na, b / nb)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _wrist_tracking_error(
    live: mujoco.MjData, ref: mujoco.MjData, wrist_body_ids: list[int]
) -> tuple[float, float]:
    """Mean wrist (pos_m, quat_deg) error between a live and reference MjData."""
    if not wrist_body_ids:
        return 0.0, 0.0
    pos_errs = []
    quat_errs = []
    for bid in wrist_body_ids:
        pos_errs.append(float(np.linalg.norm(live.xpos[bid] - ref.xpos[bid])))
        quat_errs.append(_quat_angle_deg(live.xquat[bid], ref.xquat[bid]))
    return float(np.mean(pos_errs)), float(np.mean(quat_errs))


@dataclass
class EpisodeOption:
    index: int
    path: Path
    scene: str
    date: str
    task_id: str
    task_title: str
    task_index: int | None
    archive: str
    session: str
    episode_dir: str
    frame_count: int | None


@dataclass
class ReplayEpisode:
    traj_path: Path
    root: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    arrays: dict[str, Any | None]
    xml_path: Path
    arm_addresses: dict[str, list[tuple[int, int]]]
    hand_addresses: dict[str, list[tuple[int, int]]]
    arm_actuator_ids: dict[str, list[int]]
    ik_control_override: dict[str, Any] | None
    contact_filter: HandObjectContactFilterResult | None
    frame_dt: float
    physics_steps_per_frame: int
    start: int
    end: int


def _try_resolve_episode_zarr_path(path: Path) -> Path | None:
    try:
        resolved = resolve_episode_zarr_path(path)
    except (FileNotFoundError, OSError):
        return None
    return resolved if resolved.exists() else None


def discover_episode_zarr_paths(path: str | Path) -> list[Path]:
    """Return one or more episode zarrs from a zarr, episode dir, or dataset dir."""

    root = Path(path).expanduser().resolve()
    if root.suffix == ".zarr" or not root.is_dir():
        resolved = _try_resolve_episode_zarr_path(root)
        return [] if resolved is None else [resolved]

    resolved = _try_resolve_episode_zarr_path(root)
    if resolved is not None:
        return [resolved]

    found: list[Path] = []
    seen: set[Path] = set()
    replay_configs = sorted(
        Path(dirpath) / "replay_config.json"
        for dirpath, _, filenames in os.walk(root, followlinks=True)
        if "replay_config.json" in filenames
    )
    for config_path in replay_configs:
        zarr_path = _try_resolve_episode_zarr_path(config_path.parent)
        if zarr_path is not None and zarr_path not in seen:
            found.append(zarr_path)
            seen.add(zarr_path)

    if found:
        return found

    # Fallback for directories containing zarrs but no replay_config.json.
    zarr_paths = sorted(
        Path(dirpath) / dirname
        for dirpath, dirnames, _ in os.walk(root, followlinks=True)
        for dirname in dirnames
        if dirname.endswith(".zarr")
    )
    for zarr_path in zarr_paths:
        resolved = zarr_path.resolve()
        if resolved not in seen:
            found.append(resolved)
            seen.add(resolved)
    return found


def _date_from_session_name(session: str) -> str:
    raw = session.split("_", 1)[0]
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return "<unknown date>"


def _episode_option_from_path(index: int, path: Path) -> EpisodeOption:
    scene = "<unknown>"
    task_id = "<missing>"
    task_title = ""
    task_index = None
    frame_count = None
    try:
        root = zarr.open(str(path), mode="r")
        task = root.attrs.get("task") or {}
        if isinstance(task, dict):
            task_id = str(task.get("task_id") or task_id)
            task_title = str(task.get("title") or "")
            raw_task_index = task.get("task_index")
            task_index = int(raw_task_index) if raw_task_index is not None else None
        streamer_config = root.attrs.get("streamer_config") or {}
        config = streamer_config.get("config") if isinstance(streamer_config, dict) else {}
        builder = config.get("builder") if isinstance(config, dict) else {}
        if isinstance(builder, dict):
            scene = str(builder.get("scene_type") or scene)
        raw_frame_count = root.attrs.get("trajectory_length")
        frame_count = int(raw_frame_count) if raw_frame_count is not None else None
    except Exception as exc:
        print(f"Could not read episode metadata for {path}: {exc}")

    episode_dir = path.parent.name
    if "__" in episode_dir:
        parts = episode_dir.split("__", 2)
        if len(parts) == 3:
            archive, session, episode_dir = parts
        else:
            archive = path.parents[4].name if len(path.parents) > 4 else ""
            session = path.parent.parent.name if len(path.parents) > 1 else ""
    else:
        archive = path.parents[4].name if len(path.parents) > 4 else ""
        session = path.parent.parent.name if len(path.parents) > 1 else ""
    date = _date_from_session_name(session)
    return EpisodeOption(
        index=index,
        path=path,
        scene=scene,
        date=date,
        task_id=task_id,
        task_title=task_title,
        task_index=task_index,
        archive=archive,
        session=session,
        episode_dir=episode_dir,
        frame_count=frame_count,
    )


def build_episode_options(episode_paths: Sequence[Path]) -> list[EpisodeOption]:
    return [
        _episode_option_from_path(index, path)
        for index, path in enumerate(episode_paths)
    ]


def _task_browser_label(option: EpisodeOption) -> str:
    if option.task_title:
        return f"{option.task_id} - {option.task_title}"
    return option.task_id


def _short_task_name(task_id: str) -> str:
    return task_id.rsplit("/", 1)[-1]


def _option_hours(option: EpisodeOption) -> float:
    if option.frame_count is None:
        return 0.0
    return float(option.frame_count) / 60.0 / 3600.0


def _episode_browser_label(option: EpisodeOption) -> str:
    duration = (
        f", {option.frame_count / 60.0:.1f}s"
        if option.frame_count is not None else ""
    )
    prefix = f"{option.archive}/" if option.archive else ""
    return (
        f"#{option.index + 1:03d} "
        f"{prefix}{option.session}/{option.episode_dir}{duration}"
    )


def _episode_label(traj_path: Path, root: Any | None = None) -> str:
    episode_dir = traj_path.parent
    session_dir = episode_dir.parent
    task_id = None
    if root is not None:
        task = root.attrs.get("task") or {}
        if isinstance(task, dict):
            task_id = task.get("task_id")
    prefix = f"{session_dir.name}/{episode_dir.name}"
    return f"{prefix} ({task_id})" if task_id else prefix


def _default_contact_filter_cache_dir(trajectory: Path) -> Path:
    path = trajectory.expanduser().resolve()
    base_dir = path if path.is_dir() else path.parent
    for candidate in (
        base_dir / "contact_filter_cache",
        base_dir / ".contact_filter_cache",
        base_dir.parent / "contact_filter_cache",
        base_dir.parent / ".contact_filter_cache",
    ):
        if candidate.exists():
            return candidate
    return base_dir / ".contact_filter_cache"


def _print_episode_summary(episode: ReplayEpisode) -> None:
    print(f"Rebuilt scene XML: {episode.xml_path}")
    print(f"Episode: {episode.traj_path}")
    print(f"Frames: {episode.arrays['frame_count']} selected={episode.start}:{episode.end}")
    print(
        f"Frame dt: {episode.frame_dt:.6f}s ({1.0 / episode.frame_dt:.2f} Hz), "
        f"physics_steps/frame={episode.physics_steps_per_frame}, "
        f"model timestep={episode.model.opt.timestep:.8f}s, "
        f"arena={episode.model.narena / (1024.0 * 1024.0):.1f} MiB"
    )
    if episode.ik_control_override is not None:
        override = episode.ik_control_override
        print(
            "IK control override: "
            f"latency={override['latency_seconds'] * 1000.0:.2f}ms, "
            f"stiffness_mult={override['stiffness_mult']:.3g}, "
            f"damping_mult={override['damping_mult']:.3g}"
        )
    if episode.contact_filter is not None:
        filt = episode.contact_filter
        print(
            "Contact filter: "
            f"hand_geoms={len(filt.hand_geom_ids)}, "
            f"object_geoms={len(filt.object_geom_ids)}, "
            f"pre_buffer={filt.pre_contact_buffer_s:.2f}s, "
            f"post_buffer={filt.post_contact_buffer_s:.2f}s, "
            f"cut_spans={len(filt.cut_spans)}, "
            f"cut_time={filt.cut_duration_s:.1f}s, "
            f"longest_no_contact={filt.longest_no_contact_s:.1f}s"
        )


def _hash_model_array(hasher: "hashlib._Hash", name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.view(np.uint8).tobytes())


def _hash_model_names(
    hasher: "hashlib._Hash",
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    count: int,
) -> None:
    hasher.update(str(int(obj_type)).encode("ascii"))
    for obj_id in range(count):
        name = mujoco.mj_id2name(model, obj_type, obj_id) or ""
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")


def _viewer_scene_signature(model: mujoco.MjModel) -> str:
    """Return a conservative fingerprint for Viser scene-handle reuse."""

    hasher = hashlib.blake2b(digest_size=16)
    for name in (
        "nbody",
        "ngeom",
        "nsite",
        "nmesh",
        "nmat",
        "nmocap",
    ):
        hasher.update(name.encode("ascii"))
        hasher.update(int(getattr(model, name)).to_bytes(8, "little", signed=False))

    for obj_type, count in (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        (mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
        (mujoco.mjtObj.mjOBJ_MATERIAL, model.nmat),
    ):
        _hash_model_names(hasher, model, obj_type, int(count))

    for name in (
        "body_parentid",
        "body_mocapid",
        "body_pos",
        "body_quat",
        "geom_bodyid",
        "geom_type",
        "geom_group",
        "geom_dataid",
        "geom_matid",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_rgba",
        "site_bodyid",
        "site_type",
        "site_group",
        "site_size",
        "site_pos",
        "site_quat",
        "site_rgba",
        "mesh_vertadr",
        "mesh_vertnum",
        "mesh_faceadr",
        "mesh_facenum",
        "mesh_vert",
        "mesh_face",
        "mat_rgba",
    ):
        if hasattr(model, name):
            _hash_model_array(hasher, name, getattr(model, name))
    return hasher.hexdigest()


def load_replay_episode(
    traj_path: Path,
    *,
    start_frame: int,
    num_steps: int | None,
    fps_override: float | None,
    visualize_ik: bool,
    ik_control_rate_override: float | None,
    ik_control_latency_override: int,
    ik_control_stiffness_mult: float,
    ik_control_damping_mult: float,
    contact_filter_enabled: bool,
    no_contact_cut_seconds: float,
    contact_filter_buffer_seconds: float,
    pre_contact_buffer_seconds: float,
    post_contact_buffer_seconds: float,
    contact_filter_cache_dir: Path | None,
    contact_filter_cache_only: bool,
    replay_memory_bytes: int | None,
    scene_overrides: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ReplayEpisode:
    root = zarr.open(str(traj_path), mode="r")
    if "trajectory" not in root:
        raise SystemExit(f"Invalid episode zarr (missing trajectory group): {traj_path}")

    recorded_xml_path = traj_path.resolve().parent.parent / str(root.attrs["scene_xml"])
    model, xml_path = load_model_from_recording(
        root,
        replay_memory_bytes=replay_memory_bytes,
        recorded_xml_path=recorded_xml_path,
        scene_overrides=scene_overrides,
    )
    data = mujoco.MjData(model)
    arm_addresses = _arm_joint_addresses(model)
    hand_addresses = _hand_joint_addresses(model)
    arm_actuator_ids = _arm_actuator_ids(model)
    hand_actuator_ids = _hand_actuator_ids(model)

    arrays = _load_arrays(root["trajectory"])
    _validate_shapes(model, arrays)
    ik_control_override = _build_ik_control_override(
        model,
        arrays,
        arm_actuator_ids,
        hand_actuator_ids,
        ik_control_rate_override,
        ik_control_latency_override,
        ik_control_stiffness_mult,
        ik_control_damping_mult,
    )
    if visualize_ik:
        if arrays.get("arm_ik_solution_qpos") is None:
            raise SystemExit("--visualize-ik requires trajectory/arm_ik_solution_qpos")
        if not arm_addresses:
            raise SystemExit("--visualize-ik could not find YAM arm joints in the model")
        if arrays.get("hand_ik_solution_qpos") is not None and not hand_addresses:
            raise SystemExit("--visualize-ik found hand IK data but could not find Sharpa hand joints in the model")

    frame_dt = _frame_dt(arrays["sim_time"], fps_override)
    physics_step = np.asarray(arrays["physics_step"], dtype=np.int64)
    physics_steps_per_frame = int(np.median(np.diff(physics_step)))
    start, end = _window(int(arrays["frame_count"]), start_frame, num_steps)

    contact_filter: HandObjectContactFilterResult | None = None
    if contact_filter_enabled:
        if contact_filter_cache_dir is not None:
            try:
                contact_filter = load_contact_filter_cache(
                    contact_filter_cache_dir,
                    traj_path,
                    no_contact_cut_seconds,
                    contact_buffer_s=contact_filter_buffer_seconds,
                    pre_contact_buffer_s=pre_contact_buffer_seconds,
                    post_contact_buffer_s=post_contact_buffer_seconds,
                    expected_frame_count=int(arrays["frame_count"]),
                )
            except Exception as exc:
                print(f"Ignoring invalid contact filter cache for {traj_path}: {exc}")
        if contact_filter is None and contact_filter_cache_only:
            print(f"Contact filter cache miss for {traj_path}; overlay disabled")
        if contact_filter is None:
            if not contact_filter_cache_only:
                contact_filter = compute_hand_object_contact_filter(
                    model,
                    arrays["qpos"],
                    arrays["sim_time"],
                    threshold_s=no_contact_cut_seconds,
                    contact_buffer_s=contact_filter_buffer_seconds,
                    pre_contact_buffer_s=pre_contact_buffer_seconds,
                    post_contact_buffer_s=post_contact_buffer_seconds,
                    qvel=arrays.get("qvel"),
                    progress=progress,
                )
                if contact_filter_cache_dir is not None:
                    payload = contact_filter_to_cache_dict(
                        contact_filter,
                        traj_path=traj_path,
                        frame_count=int(arrays["frame_count"]),
                        frame_times=arrays["sim_time"],
                        method="cpu-viewer",
                    )
                    save_contact_filter_cache(
                        contact_filter_cache_dir,
                        traj_path,
                        no_contact_cut_seconds,
                        payload,
                        contact_buffer_s=contact_filter_buffer_seconds,
                        pre_contact_buffer_s=pre_contact_buffer_seconds,
                        post_contact_buffer_s=post_contact_buffer_seconds,
                    )

    _set_state(model, data, arrays, start, ik_control_override)

    return ReplayEpisode(
        traj_path=traj_path,
        root=root,
        model=model,
        data=data,
        arrays=arrays,
        xml_path=xml_path,
        arm_addresses=arm_addresses,
        hand_addresses=hand_addresses,
        arm_actuator_ids=arm_actuator_ids,
        ik_control_override=ik_control_override,
        contact_filter=contact_filter,
        frame_dt=frame_dt,
        physics_steps_per_frame=physics_steps_per_frame,
        start=start,
        end=end,
    )


def _run_viser(
    episode: ReplayEpisode,
    *,
    episode_paths: Sequence[Path],
    episode_options: Sequence[EpisodeOption],
    episode_index: int,
    load_episode: Callable[[int], ReplayEpisode],
    speed: float,
    host: str,
    port: int,
    show_mocap_skeleton: bool,
    visualize_ik: bool,
    kinematic: bool = False,
    future_arm_offset: int = 0,
) -> None:
    import viser
    import viser.uplot as uplot
    import mjviser.conversions as mjviser_conversions
    from mjviser import ViserMujocoScene

    # The replay viewer cares about faithful visuals and fast episode swaps, not
    # minimizing GLB vertex count. On these robot meshes, trimesh vertex
    # deduplication can dominate scene construction time.
    mjviser_conversions._can_merge_vertices = lambda _: False

    class ReplayViserMujocoScene(ViserMujocoScene):
        def _compute_hull_body_meshes(self) -> None:
            self._hull_body_meshes = {}
            self._hull_mesh_bodies = set()

        def _build_hull_handles(self) -> None:
            self._hull_fixed_handles = {}
            self._hull_dynamic_handles = []

    server = viser.ViserServer(host=host, port=port, label="trajectory-replay")
    server.scene.add_grid("/replay/grid", width=2.0, height=2.0, plane="xy", cell_size=0.1, section_size=0.5)
    server.initial_camera.position = np.array([1.0, -1.1, 0.8])
    server.initial_camera.look_at = np.array([0.45, 0.0, 0.25])
    server.initial_camera.up = np.array([0.0, 0.0, 1.0])

    current_episode = episode
    current_index = int(episode_index)
    episode_options_by_index = {option.index: option for option in episode_options}
    model = current_episode.model
    data = current_episode.data
    arrays = current_episode.arrays
    frame_count = int(arrays["frame_count"])
    frame_dt = current_episode.frame_dt
    start = current_episode.start
    end = current_episode.end
    ik_control_override = current_episode.ik_control_override
    contact_filter = current_episode.contact_filter
    arm_addresses = current_episode.arm_addresses
    hand_addresses = current_episode.hand_addresses
    future_arm = _make_future_arm(current_episode, future_arm_offset)
    scene: ViserMujocoScene | None = None
    scene_signature: str | None = None
    ik_data: mujoco.MjData | None = None
    ik_overlay: dict[int, list[Any]] = {}
    ghost_data: mujoco.MjData | None = None
    ghost_overlay: dict[int, list[Any]] = {}
    ghost_checkbox: Any = None

    def ghost_checkbox_value() -> bool:
        return ghost_checkbox is not None and bool(ghost_checkbox.value)

    wrist_body_ids: list[int] = _wrist_body_ids(model)
    error_window = 300  # rolling samples shown on the tracking-error plots
    error_t: deque[float] = deque(maxlen=error_window)
    error_pos: deque[float] = deque(maxlen=error_window)
    error_quat: deque[float] = deque(maxlen=error_window)

    mocap_names: dict[int, str] = {}
    mocap_edges: list[tuple[int, int]] = []
    cut_span_by_frame: np.ndarray | None = None

    filter_marker = server.scene.add_box(
        "/replay/contact_filter_cut_marker",
        color=(255, 30, 30),
        dimensions=(0.12, 0.12, 0.12),
        position=np.array([0.45, -0.62, 0.85]),
        opacity=0.9,
        visible=False,
        cast_shadow=False,
        receive_shadow=False,
    )
    keep_marker = server.scene.add_box(
        "/replay/contact_filter_keep_marker",
        color=(30, 180, 70),
        dimensions=(0.12, 0.12, 0.12),
        position=np.array([0.45, -0.62, 0.85]),
        opacity=0.9,
        visible=False,
        cast_shadow=False,
        receive_shadow=False,
    )

    state = {
        "start": start,
        "end": end,
        "frame": start,
        "playing": False,
        "last": time.perf_counter(),
        "loading": False,
    }

    def refresh_episode_auxiliary_handles(remove_existing: bool = True) -> None:
        nonlocal ik_data, ik_overlay, ghost_data, ghost_overlay
        nonlocal mocap_names, mocap_edges, cut_span_by_frame
        if remove_existing:
            server.scene.remove_by_name("/ik_overlay")
            server.scene.remove_by_name("/kinematic_ghost")
            server.scene.remove_by_name("/replay/mocap_points")
            server.scene.remove_by_name("/replay/mocap_skeleton")
        ik_data = mujoco.MjData(model) if visualize_ik else None
        ik_overlay = _create_ik_overlay(server, model, data) if visualize_ik else {}
        ghost_enabled = not kinematic and ghost_checkbox_value()
        ghost_data = mujoco.MjData(model) if ghost_enabled else None
        ghost_overlay = (
            _create_blue_body_overlay(server, model, data, "/kinematic_ghost")
            if ghost_enabled
            else {}
        )
        mocap_names = _mocap_names_by_id(model)
        mocap_edges = _mocap_skeleton_edges(mocap_names)
        cut_span_by_frame = None
        if contact_filter is not None:
            cut_span_by_frame = np.full(frame_count, -1, dtype=np.int32)
            for span_idx, span in enumerate(contact_filter.cut_spans):
                cut_span_by_frame[span.start : span.end] = span_idx

    def rebuild_scene(remove_existing: bool = True) -> None:
        nonlocal scene, scene_signature
        if remove_existing:
            server.scene.remove_by_name("/bodies")
            server.scene.remove_by_name("/fixed_bodies")
        # mjviser adds dynamic meshes under /bodies/... but does not create a
        # root handle for that subtree. Create one here so scene swaps can
        # remove the whole old dynamic-mesh subtree before rebuilding.
        server.scene.add_frame("/bodies", show_axes=False)
        scene = ReplayViserMujocoScene(server, model, num_envs=1)
        scene.camera_tracking_enabled = False
        scene_signature = _viewer_scene_signature(model)
        refresh_episode_auxiliary_handles(remove_existing=True)
        print(f"Viser scene rebuilt: signature={scene_signature}", flush=True)

    rebuild_scene(remove_existing=False)

    def time_step_s() -> float:
        return max(float(frame_dt), 1e-6)

    def frame_to_time_s(frame_idx: int) -> float:
        return float(frame_idx) * time_step_s()

    def duration_to_frames(duration_s: float) -> int:
        return max(1, int(round(float(duration_s) / time_step_s())))

    def frame_from_time_s(time_s: float) -> int:
        frame_idx = int(round(float(time_s) / time_step_s()))
        return min(max(frame_idx, 0), max(0, frame_count - 1))

    def episode_duration_s() -> float:
        return float(frame_count) * time_step_s()

    def window_duration_s(start_frame: int, end_frame: int) -> float:
        return max(time_step_s(), float(max(1, end_frame - start_frame)) * time_step_s())

    def span_start_time_s(start_frame: int) -> float:
        return frame_to_time_s(start_frame)

    def span_end_time_s(end_frame: int) -> float:
        return frame_to_time_s(end_frame)

    with server.gui.add_folder("Contact Filter", expand_by_default=True):
        filter_banner = server.gui.add_html("")
        filter_summary = server.gui.add_markdown("")
    with server.gui.add_folder("Replay"):
        status = server.gui.add_markdown("")
        start_slider = server.gui.add_slider(
            "Start Time (s)",
            min=0.0,
            max=max(0.0, frame_to_time_s(frame_count - 1)),
            step=time_step_s(),
            initial_value=frame_to_time_s(start),
        )
        steps_slider = server.gui.add_slider(
            "Duration (s)",
            min=time_step_s(),
            max=max(time_step_s(), float(frame_count - start) * time_step_s()),
            step=time_step_s(),
            initial_value=window_duration_s(start, end),
        )
        frame_slider = server.gui.add_slider(
            "Time (s)",
            min=frame_to_time_s(start),
            max=max(frame_to_time_s(start), frame_to_time_s(end - 1)),
            step=time_step_s(),
            initial_value=frame_to_time_s(start),
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.05,
            max=max(4.0, float(speed) * 2.0),
            step=0.05,
            initial_value=float(speed),
        )
        future_arm_slider = server.gui.add_slider(
            "Future Arm Offset (frames)",
            min=1,
            max=15,
            step=1,
            initial_value=int(min(15, max(1, future_arm_offset))),
        )
        future_arm_checkbox = server.gui.add_checkbox(
            "Use Future Arm Offset",
            initial_value=future_arm_offset > 0,
        )
        ghost_checkbox = server.gui.add_checkbox(
            "Show Kinematic Ghost",
            initial_value=False,
            disabled=kinematic,
        )
        play_button = server.gui.add_button("Play")
        pause_button = server.gui.add_button("Pause")
        reset_button = server.gui.add_button("Reset")
    with server.gui.add_folder("Wrist Tracking Error", expand_by_default=True):
        error_summary = server.gui.add_markdown(
            "_Enable **Show Kinematic Ghost** to measure wrist tracking error._"
        )
        # uPlot needs >=2 points to draw a line; seed with a flat zero segment.
        _seed = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        error_pos_plot = server.gui.add_uplot(
            data=_seed,
            series=(
                uplot.Series(label="t (s)"),
                uplot.Series(label="pos err (m)", stroke="#3796ff", width=2.0),
            ),
            aspect=1.6,
        )
        error_quat_plot = server.gui.add_uplot(
            data=_seed,
            series=(
                uplot.Series(label="t (s)"),
                uplot.Series(label="quat err (deg)", stroke="#ff7a3c", width=2.0),
            ),
            aspect=1.6,
        )
    with server.gui.add_folder("Episodes", expand_by_default=True):
        episode_status = server.gui.add_markdown("")
        date_dropdown = server.gui.add_dropdown(
            "Date",
            options=["<loading>"],
            initial_value="<loading>",
        )
        task_dropdown = server.gui.add_dropdown(
            "Task",
            options=["<loading>"],
            initial_value="<loading>",
        )
        episode_dropdown = server.gui.add_dropdown(
            "Episode",
            options=["<loading>"],
            initial_value="<loading>",
        )
        load_selected_episode_button = server.gui.add_button("Load Selected Episode")
        prev_episode_button = server.gui.add_button("Previous Episode")
        next_episode_button = server.gui.add_button("Next Episode")
    browser_state = {"updating": False}

    def option_for_index(index: int) -> EpisodeOption:
        return episode_options_by_index.get(
            index,
            EpisodeOption(
                index=index,
                path=episode_paths[index],
                scene="<unknown>",
                date="<unknown date>",
                task_id="<missing>",
                task_title="",
                task_index=None,
                archive="",
                session=episode_paths[index].parent.parent.name,
                episode_dir=episode_paths[index].parent.name,
                frame_count=None,
            ),
        )

    def date_options() -> list[str]:
        return sorted({option.date for option in episode_options})

    def task_groups_for_date(date: str) -> dict[str, list[EpisodeOption]]:
        task_groups: dict[str, list[EpisodeOption]] = {}
        for option in episode_options:
            if option.date == date:
                task_groups.setdefault(option.task_id, []).append(option)
        return task_groups

    def task_label_for_group(task_id: str, options: Sequence[EpisodeOption]) -> str:
        hours = sum(_option_hours(option) for option in options)
        return f"({hours:.2f}h {len(options)}) {_short_task_name(task_id)}"

    def task_label_map_for_date(date: str) -> dict[str, str]:
        task_groups = task_groups_for_date(date)
        return {
            task_label_for_group(task_id, options): task_id
            for task_id, options in task_groups.items()
        }

    def task_options_for_date(date: str) -> list[str]:
        task_groups = task_groups_for_date(date)
        items: list[tuple[int, str, str]] = []
        for task_id, options in task_groups.items():
            task_index = min(
                (
                    option.task_index
                    for option in options
                    if option.task_index is not None
                ),
                default=999,
            )
            items.append((task_index, task_id.lower(), task_label_for_group(task_id, options)))
        return [label for _, _, label in sorted(items)]

    def episode_options_for_selection(date: str, task_label: str) -> list[EpisodeOption]:
        task_id = task_label_map_for_date(date).get(task_label)
        if task_id is None:
            return []
        options = [
            option
            for option in episode_options
            if option.date == date and option.task_id == task_id
        ]
        return sorted(
            options,
            key=lambda option: (
                option.archive,
                option.session,
                option.episode_dir,
                option.index,
            ),
        )

    def update_episode_dropdown_options(date: str, task_label: str) -> None:
        options = episode_options_for_selection(date, task_label)
        labels = [_episode_browser_label(option) for option in options]
        episode_dropdown.options = labels or ["<none>"]
        if labels and episode_dropdown.value not in labels:
            episode_dropdown.value = labels[0]

    def sync_browser_to_index(index: int) -> None:
        option = option_for_index(index)
        browser_state["updating"] = True
        try:
            dates = date_options()
            date_dropdown.options = dates or ["<none>"]
            date_dropdown.value = option.date if option.date in dates else date_dropdown.options[0]
            tasks = task_options_for_date(date_dropdown.value)
            task_dropdown.options = tasks or ["<none>"]
            matching = [
                label
                for label, task_id in task_label_map_for_date(date_dropdown.value).items()
                if task_id == option.task_id
            ]
            task_label = matching[0] if matching else task_dropdown.options[0]
            task_dropdown.value = task_label if task_label in tasks else task_dropdown.options[0]
            update_episode_dropdown_options(date_dropdown.value, task_dropdown.value)
            option_label = _episode_browser_label(option)
            if option_label in episode_dropdown.options:
                episode_dropdown.value = option_label
        finally:
            browser_state["updating"] = False

    def selected_browser_episode_index() -> int | None:
        for option in episode_options_for_selection(
            str(date_dropdown.value), str(task_dropdown.value)
        ):
            if _episode_browser_label(option) == episode_dropdown.value:
                return option.index
        return None

    def update_browser_selection_status() -> None:
        selected_index = selected_browser_episode_index()
        if selected_index is None:
            return
        option = option_for_index(selected_index)
        episode_status.content = (
            f"`selected={selected_index + 1}/{len(episode_paths)}`  "
            f"`loaded={current_index + 1}/{len(episode_paths)}`  "
            f"`date={option.date}`  "
            f"`task={option.task_id}`"
        )

    def update_episode_status() -> None:
        option = option_for_index(current_index)
        episode_status.content = (
            f"`episode={current_index + 1}/{len(episode_paths)}`  "
            f"`date={option.date}`  "
            f"`scene={option.scene}`  "
            f"`task={option.task_id}`  "
            f"`duration={episode_duration_s():.1f}s`  "
            f"`{_episode_label(current_episode.traj_path, current_episode.root)}`"
        )

    def update_filter_summary() -> None:
        if contact_filter is None:
            filter_summary.content = "No hand/object contact filter overlay is active."
            return
        filter_summary.content = (
            f"`threshold={contact_filter.threshold_s:.2f}s`  "
            f"`pre_buffer={contact_filter.pre_contact_buffer_s:.2f}s`  "
            f"`post_buffer={contact_filter.post_contact_buffer_s:.2f}s`  "
            f"`cut_spans={len(contact_filter.cut_spans)}`  "
            f"`cut_time={contact_filter.cut_duration_s:.1f}s`  "
            f"`longest_no_contact={contact_filter.longest_no_contact_s:.1f}s`"
        )

    def update_status() -> None:
        mode_label = "kinematic" if kinematic else "action rollout"
        status.content = (
            f"`episode={current_index + 1}/{len(episode_paths)}`  "
            f"`time={frame_to_time_s(state['frame']):.2f}s`  "
            f"`start={frame_to_time_s(state['start']):.2f}s`  "
            f"`duration={window_duration_s(state['start'], state['end']):.2f}s`  "
            f"`speed={float(speed_slider.value):.2f}x`  "
            f"`ik_ctrl_override={ik_control_override is not None}`  "
            f"`mode={mode_label}`  "
            f"`ik_overlay={visualize_ik}`"
        )

    def update_filter_indicator(frame_idx: int) -> None:
        if contact_filter is None or cut_span_by_frame is None:
            filter_marker.visible = False
            keep_marker.visible = False
            filter_banner.content = (
                "<div style='padding:8px;border-radius:4px;background:#f2f2f2;"
                "color:#444;font-weight:600'>Contact filter inactive</div>"
            )
            return
        span_idx = int(cut_span_by_frame[frame_idx])
        in_cut = span_idx >= 0
        filter_marker.visible = in_cut
        keep_marker.visible = not in_cut
        if in_cut:
            span = contact_filter.cut_spans[span_idx]
            filter_banner.content = (
                "<div style='padding:8px;border-radius:4px;background:#b00020;"
                "color:white;font-weight:700'>WOULD CUT"
                f"<br/><span style='font-weight:500'>t={frame_to_time_s(frame_idx):.2f}s is in "
                f"no-contact span {span_start_time_s(span.start):.2f}-"
                f"{span_end_time_s(span.end):.2f}s "
                f"({span.duration_s:.1f}s &gt; {contact_filter.threshold_s:.1f}s)</span></div>"
            )
        else:
            filter_banner.content = ""

    def push_error_plots() -> None:
        # uPlot needs >=2 points; pad a single sample into a flat segment.
        t_arr = np.asarray(error_t, dtype=np.float64)
        pos_arr = np.asarray(error_pos, dtype=np.float64)
        quat_arr = np.asarray(error_quat, dtype=np.float64)
        if t_arr.size == 0:
            t_arr = np.array([0.0, 1.0])
            pos_arr = np.array([0.0, 0.0])
            quat_arr = np.array([0.0, 0.0])
        elif t_arr.size == 1:
            t_arr = np.array([t_arr[0], t_arr[0] + 1e-3])
            pos_arr = np.repeat(pos_arr, 2)
            quat_arr = np.repeat(quat_arr, 2)
        error_pos_plot.data = (t_arr, pos_arr)
        error_quat_plot.data = (t_arr, quat_arr)

    def clear_tracking_error() -> None:
        error_t.clear()
        error_pos.clear()
        error_quat.clear()
        push_error_plots()
        if ghost_data is None:
            error_summary.content = (
                "_Enable **Show Kinematic Ghost** to measure wrist tracking error._"
            )
        else:
            error_summary.content = "_Play the rollout to accumulate wrist error._"

    def record_tracking_error(frame_idx: int) -> None:
        assert ghost_data is not None
        pos_err, quat_err = _wrist_tracking_error(data, ghost_data, wrist_body_ids)
        error_t.append(frame_to_time_s(frame_idx))
        error_pos.append(pos_err)
        error_quat.append(quat_err)
        push_error_plots()
        pos_arr = np.asarray(error_pos, dtype=np.float64)
        quat_arr = np.asarray(error_quat, dtype=np.float64)
        error_summary.content = (
            f"**pos** {pos_err * 1000.0:.1f} mm "
            f"(avg {float(pos_arr.mean()) * 1000.0:.1f}) &nbsp; "
            f"**quat** {quat_err:.2f}° (avg {float(quat_arr.mean()):.2f})"
        )

    def update_scene(frame_idx: int) -> None:
        assert scene is not None
        scene.update_from_mjdata(data)
        if visualize_ik:
            assert ik_data is not None
            _set_ik_overlay_state(model, ik_data, arrays, frame_idx, arm_addresses, hand_addresses)
            _update_blue_overlay(ik_overlay, ik_data)
        if ghost_data is not None:
            _set_visual_state(model, ghost_data, arrays, frame_idx)
            _update_blue_overlay(ghost_overlay, ghost_data)
            record_tracking_error(frame_idx)
        if show_mocap_skeleton:
            _draw_mocap_skeleton(server, model, arrays, frame_idx, mocap_names, mocap_edges)
        update_filter_indicator(frame_idx)

    def apply_frame_state(frame_idx: int, *, action_rollout: bool = False) -> bool:
        try:
            if action_rollout:
                _step_action(
                    model, data, arrays, frame_idx, ik_control_override, future_arm
                )
            else:
                _set_visual_state(model, data, arrays, frame_idx, ik_control_override)
        except Exception as exc:
            state["playing"] = False
            message = (
                f"Time {frame_to_time_s(frame_idx):.2f}s failed for "
                f"episode {current_index + 1}: {exc}"
            )
            print(message)
            status.content = f"`{message}`"
            filter_marker.visible = False
            keep_marker.visible = False
            filter_banner.content = (
                "<div style='padding:8px;border-radius:4px;background:#b00020;"
                "color:white;font-weight:700'>Time update failed"
                f"<br/><span style='font-weight:500'>{message}</span></div>"
            )
            return False
        return True

    def reset_to_window() -> None:
        if state["loading"]:
            return
        state["start"], state["end"] = _window(
            frame_count,
            frame_from_time_s(float(start_slider.value)),
            duration_to_frames(float(steps_slider.value)),
        )
        state["frame"] = state["start"]
        steps_slider.max = max(
            time_step_s(), float(frame_count - state["start"]) * time_step_s()
        )
        frame_slider.min = frame_to_time_s(state["start"])
        frame_slider.max = max(
            frame_to_time_s(state["start"]), frame_to_time_s(state["end"] - 1)
        )
        frame_slider.value = frame_to_time_s(state["frame"])
        if not apply_frame_state(state["start"]):
            return
        clear_tracking_error()
        update_scene(state["frame"])
        update_status()

    def apply_loaded_episode(next_episode: ReplayEpisode, next_index: int) -> None:
        nonlocal current_episode, current_index, model, data, arrays, frame_count, frame_dt
        nonlocal start, end, ik_control_override, contact_filter, arm_addresses, hand_addresses
        nonlocal future_arm, wrist_body_ids
        current_episode = next_episode
        current_index = int(next_index)
        model = current_episode.model
        data = current_episode.data
        arrays = current_episode.arrays
        frame_count = int(arrays["frame_count"])
        frame_dt = current_episode.frame_dt
        start = current_episode.start
        end = current_episode.end
        ik_control_override = current_episode.ik_control_override
        contact_filter = current_episode.contact_filter
        arm_addresses = current_episode.arm_addresses
        hand_addresses = current_episode.hand_addresses
        wrist_body_ids = _wrist_body_ids(model)
        refresh_future_arm()

        state["loading"] = True
        state["playing"] = False
        try:
            next_scene_signature = _viewer_scene_signature(model)
            if scene is not None and scene_signature == next_scene_signature:
                refresh_episode_auxiliary_handles(remove_existing=True)
                print(f"Viser scene reused: signature={next_scene_signature}", flush=True)
            else:
                rebuild_scene(remove_existing=True)
            start_slider.min = 0.0
            start_slider.max = max(0.0, frame_to_time_s(frame_count - 1))
            start_slider.step = time_step_s()
            start_slider.value = frame_to_time_s(start)
            steps_slider.min = time_step_s()
            steps_slider.max = max(
                time_step_s(), float(frame_count - start) * time_step_s()
            )
            steps_slider.step = time_step_s()
            steps_slider.value = window_duration_s(start, end)
            frame_slider.min = frame_to_time_s(start)
            frame_slider.max = max(frame_to_time_s(start), frame_to_time_s(end - 1))
            frame_slider.step = time_step_s()
            frame_slider.value = frame_to_time_s(start)
            state["start"] = start
            state["end"] = end
            state["frame"] = start
            state["last"] = time.perf_counter()
            sync_browser_to_index(current_index)
            update_episode_status()
            update_filter_summary()
            clear_tracking_error()
            update_scene(state["frame"])
            update_status()
        finally:
            state["loading"] = False

    def load_episode_index(next_index: int, *, autoplay: bool) -> bool:
        state["playing"] = False
        state["loading"] = True
        option = option_for_index(next_index)
        episode_status.content = (
            f"`loading episode {next_index + 1}/{len(episode_paths)}`  "
            f"`date={option.date}`  "
            f"`scene={option.scene}`  "
            f"`task={option.task_id}`  "
            f"`{episode_paths[next_index]}`"
        )
        filter_banner.content = (
            "<div style='padding:8px;border-radius:4px;background:#f2f2f2;"
            "color:#444;font-weight:600'>Loading contact filter...</div>"
        )
        try:
            next_episode = load_episode(next_index)
            apply_loaded_episode(next_episode, next_index)
        except (Exception, SystemExit) as exc:
            state["loading"] = False
            message = f"failed to load episode {next_index + 1}/{len(episode_paths)}: {exc}"
            print(message)
            episode_status.content = f"`{message}`"
            filter_banner.content = (
                "<div style='padding:8px;border-radius:4px;background:#b00020;"
                "color:white;font-weight:700'>Episode load failed"
                f"<br/><span style='font-weight:500'>{message}</span></div>"
            )
            return False
        state["playing"] = autoplay
        return True

    def switch_episode(delta: int) -> None:
        if len(episode_paths) <= 1:
            return
        first_error: BaseException | None = None
        for step in range(1, len(episode_paths) + 1):
            next_index = (current_index + delta * step) % len(episode_paths)
            if load_episode_index(next_index, autoplay=False):
                return
            first_error = RuntimeError(episode_status.content)
        state["loading"] = False
        episode_status.content = f"`failed to load another episode`  `{first_error}`"

    @date_dropdown.on_update
    def _(_) -> None:
        if state["loading"] or browser_state["updating"]:
            return
        browser_state["updating"] = True
        try:
            tasks = task_options_for_date(str(date_dropdown.value))
            task_dropdown.options = tasks or ["<none>"]
            task_dropdown.value = task_dropdown.options[0]
            update_episode_dropdown_options(str(date_dropdown.value), str(task_dropdown.value))
        finally:
            browser_state["updating"] = False
        update_browser_selection_status()

    @task_dropdown.on_update
    def _(_) -> None:
        if state["loading"] or browser_state["updating"]:
            return
        browser_state["updating"] = True
        try:
            update_episode_dropdown_options(str(date_dropdown.value), str(task_dropdown.value))
        finally:
            browser_state["updating"] = False
        update_browser_selection_status()

    @episode_dropdown.on_update
    def _(_) -> None:
        if state["loading"] or browser_state["updating"]:
            return
        selected_index = selected_browser_episode_index()
        if selected_index is None:
            return
        if selected_index == current_index:
            update_episode_status()
            return
        load_episode_index(selected_index, autoplay=False)

    @load_selected_episode_button.on_click
    def _(_) -> None:
        if state["loading"]:
            return
        selected_index = selected_browser_episode_index()
        if selected_index is not None:
            load_episode_index(selected_index, autoplay=False)

    @start_slider.on_update
    def _(_) -> None:
        reset_to_window()

    @steps_slider.on_update
    def _(_) -> None:
        reset_to_window()

    @frame_slider.on_update
    def _(_) -> None:
        if state["loading"]:
            return
        idx = min(
            max(frame_from_time_s(float(frame_slider.value)), state["start"]),
            state["end"] - 1,
        )
        if idx == state["frame"]:
            return
        state["playing"] = False
        state["frame"] = idx
        if not apply_frame_state(idx):
            return
        clear_tracking_error()
        update_scene(state["frame"])
        update_status()

    @speed_slider.on_update
    def _(_) -> None:
        state["last"] = time.perf_counter()
        update_status()

    def refresh_future_arm() -> None:
        nonlocal future_arm
        offset = int(future_arm_slider.value) if future_arm_checkbox.value else 0
        future_arm = _make_future_arm(current_episode, offset)

    @future_arm_slider.on_update
    def _(_) -> None:
        refresh_future_arm()

    @future_arm_checkbox.on_update
    def _(_) -> None:
        refresh_future_arm()

    @ghost_checkbox.on_update
    def _(_) -> None:
        nonlocal ghost_data, ghost_overlay
        if state["loading"]:
            return
        server.scene.remove_by_name("/kinematic_ghost")
        if not kinematic and ghost_checkbox.value:
            ghost_data = mujoco.MjData(model)
            ghost_overlay = _create_blue_body_overlay(
                server, model, data, "/kinematic_ghost"
            )
        else:
            ghost_data = None
            ghost_overlay = {}
        clear_tracking_error()
        update_scene(state["frame"])

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = True
        state["last"] = time.perf_counter()

    @pause_button.on_click
    def _(_) -> None:
        state["playing"] = False

    @reset_button.on_click
    def _(_) -> None:
        state["playing"] = False
        reset_to_window()

    @prev_episode_button.on_click
    def _(_) -> None:
        switch_episode(-1)

    @next_episode_button.on_click
    def _(_) -> None:
        switch_episode(1)

    sync_browser_to_index(current_index)
    reset_to_window()
    update_episode_status()
    update_filter_summary()
    state["playing"] = False
    state["last"] = time.perf_counter()
    print(f"viser replay running at http://localhost:{server.get_port()}", flush=True)
    try:
        while True:
            now = time.perf_counter()
            wall_dt = frame_dt / max(float(speed_slider.value), 1e-6)
            if state["playing"] and not state["loading"] and now - state["last"] >= wall_dt:
                if state["frame"] + 1 >= state["end"]:
                    state["playing"] = False
                else:
                    state["frame"] += 1
                    if not apply_frame_state(state["frame"], action_rollout=not kinematic):
                        state["last"] = now
                        continue
                    frame_slider.value = frame_to_time_s(state["frame"])
                    update_scene(state["frame"])
                    update_status()
                state["last"] = now
            time.sleep(0.001)
    except KeyboardInterrupt:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to an episode folder/zarr or a directory containing multiple episodes",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Recorded frame to restore before action rollout")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of replay frames to run from start-frame")
    parser.add_argument("--fps", type=float, default=None, help="Override playback/video frame rate")
    parser.add_argument("--speed", type=float, default=1.0, help="Viser playback speed multiplier")
    parser.add_argument(
        "--ik-control-rate-override",
        type=float,
        default=None,
        help="Replay arm/hand IK solution controls at this synthetic control rate instead of using recorded IK ctrl.",
    )
    parser.add_argument(
        "--ik-control-latency-override",
        type=int,
        default=1,
        help="Integer latency in synthetic IK control ticks; one tick is 1 / --ik-control-rate-override seconds.",
    )
    parser.add_argument(
        "--ik-control-stiffness-mult",
        type=float,
        default=1.0,
        help="Multiplier on stiffness terms for arm/hand position actuators used by synthetic IK controls.",
    )
    parser.add_argument(
        "--ik-control-damping-mult",
        type=float,
        default=1.0,
        help="Multiplier on damping terms for arm/hand position actuators used by synthetic IK controls.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser host")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--no-viser", action="store_true", help="Do not start the viser playback UI")
    parser.add_argument("--mp4-out", type=Path, default=None, help="Optional MP4 path for action-rollout render")
    parser.add_argument("--width", type=int, default=1280, help="MP4 width")
    parser.add_argument("--height", type=int, default=720, help="MP4 height")
    parser.add_argument("--camera", type=str, default="top", help="MuJoCo camera name or id for MP4 rendering")
    parser.add_argument(
        "--show-mocap-skeleton",
        action="store_true",
        help="Draw recorded mocap hand skeleton points in the viser replay.",
    )
    parser.add_argument(
        "--visualize-ik",
        action="store_true",
        help="Show transparent blue arm/hand meshes at the recorded IK arm solution in viser.",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help=(
            "Kinematic replay: at each frame, set qpos/qvel from the recording "
            "and update visual kinematics only. Skips physics integration entirely; the "
            "playback is deterministic and matches the recorded trajectory "
            "exactly (no actuator dynamics, no contact-driven drift). Useful "
            "for visualization and policy-eval baselines."
        ),
    )
    parser.add_argument(
        "--future-arm-offset",
        type=int,
        default=0,
        help=(
            "Action-rollout only: instead of commanding the recorded arm ctrl, "
            "command the recorded arm_joint_qpos from N frames in the future. "
            "Hand actuators still use recorded ctrl. The offset is clamped at the "
            "last frame. 0 (default) disables this and uses recorded arm ctrl. "
            "Ignored under --kinematic."
        ),
    )
    parser.add_argument(
        "--no-contact-cut-seconds",
        type=float,
        default=4.0,
        help=(
            "Viser overlay threshold: mark frames inside hand/object no-contact "
            "spans longer than this many seconds as frames that would be cut."
        ),
    )
    parser.add_argument(
        "--contact-filter-buffer-seconds",
        type=float,
        default=0.25,
        help=(
            "Seconds of no-contact context to keep next to contact when marking "
            "cut spans. A long no-contact gap between contacts cuts from "
            "buffer seconds after the previous contact to buffer seconds before "
            "the next contact."
        ),
    )
    parser.add_argument(
        "--pre-contact-filter-buffer-seconds",
        type=float,
        default=None,
        help=(
            "Seconds to keep before a new hand/object contact starts. Defaults "
            "to --contact-filter-buffer-seconds."
        ),
    )
    parser.add_argument(
        "--post-contact-filter-buffer-seconds",
        type=float,
        default=None,
        help=(
            "Seconds to keep after hand/object contact ends. Defaults to "
            "--contact-filter-buffer-seconds."
        ),
    )
    parser.add_argument(
        "--disable-contact-filter-overlay",
        action="store_true",
        help="Disable the Viser hand/object no-contact filter overlay.",
    )
    parser.add_argument(
        "--contact-filter-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing cached hand/object contact filter JSON files. "
            "If omitted, the viewer looks for contact_filter_cache next to the "
            "trajectory dataset and writes misses to .contact_filter_cache."
        ),
    )
    parser.add_argument(
        "--contact-filter-cache-only",
        action="store_true",
        help="Use cached contact filters only; do not compute a missing filter in the viewer.",
    )
    parser.add_argument(
        "--replay-memory-mb",
        type=float,
        default=128.0,
        help=(
            "MuJoCo arena size, in MiB, used when recompiling replay models. "
            "Set to 0 to use the XML/model default."
        ),
    )
    args = parser.parse_args()

    if args.speed <= 0.0:
        raise SystemExit("--speed must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be > 0")
    if args.visualize_ik and args.no_viser:
        raise SystemExit("--visualize-ik requires the viser UI")
    if args.future_arm_offset < 0:
        raise SystemExit("--future-arm-offset must be >= 0")
    if args.future_arm_offset > 0 and args.kinematic:
        print("--future-arm-offset is ignored under --kinematic (no physics rollout)")
    if args.no_contact_cut_seconds <= 0.0:
        raise SystemExit("--no-contact-cut-seconds must be > 0")
    if args.contact_filter_buffer_seconds < 0.0:
        raise SystemExit("--contact-filter-buffer-seconds must be >= 0")
    pre_contact_buffer_seconds = (
        args.contact_filter_buffer_seconds
        if args.pre_contact_filter_buffer_seconds is None
        else args.pre_contact_filter_buffer_seconds
    )
    post_contact_buffer_seconds = (
        args.contact_filter_buffer_seconds
        if args.post_contact_filter_buffer_seconds is None
        else args.post_contact_filter_buffer_seconds
    )
    if pre_contact_buffer_seconds < 0.0:
        raise SystemExit("--pre-contact-filter-buffer-seconds must be >= 0")
    if post_contact_buffer_seconds < 0.0:
        raise SystemExit("--post-contact-filter-buffer-seconds must be >= 0")
    if args.replay_memory_mb < 0.0:
        raise SystemExit("--replay-memory-mb must be >= 0")
    episode_paths = discover_episode_zarr_paths(args.trajectory)
    if not episode_paths:
        raise SystemExit(f"No episode zarrs found under: {args.trajectory}")
    if len(episode_paths) > 1 and args.no_viser:
        raise SystemExit("Directory replay with multiple episodes requires the viser UI")
    if len(episode_paths) > 1 and args.mp4_out is not None:
        raise SystemExit("--mp4-out is only supported with a single episode path")

    episode_options = build_episode_options(episode_paths)
    contact_filter_enabled = not args.no_viser and not args.disable_contact_filter_overlay
    contact_filter_cache_dir = (
        args.contact_filter_cache_dir.expanduser().resolve()
        if args.contact_filter_cache_dir is not None
        else _default_contact_filter_cache_dir(args.trajectory)
    )
    if contact_filter_enabled:
        print(f"Contact filter cache: {contact_filter_cache_dir}")

    scene_overrides = None

    def _load_episode_at(index: int) -> ReplayEpisode:
        traj_path = episode_paths[index]
        print(f"Loading episode {index + 1}/{len(episode_paths)}: {traj_path}")
        if contact_filter_enabled:
            print(
                "Loading hand/object contact filter "
                f"(no-contact cutoff > {args.no_contact_cut_seconds:.2f}s, "
                f"pre_buffer={pre_contact_buffer_seconds:.2f}s, "
                f"post_buffer={post_contact_buffer_seconds:.2f}s)..."
            )

        def _progress(done: int, total: int) -> None:
            if contact_filter_enabled:
                print(f"  contact filter frames {done}/{total}")

        return load_replay_episode(
            traj_path,
            start_frame=args.start_frame,
            num_steps=args.num_steps,
            fps_override=args.fps,
            visualize_ik=args.visualize_ik,
            ik_control_rate_override=args.ik_control_rate_override,
            ik_control_latency_override=args.ik_control_latency_override,
            ik_control_stiffness_mult=args.ik_control_stiffness_mult,
            ik_control_damping_mult=args.ik_control_damping_mult,
            contact_filter_enabled=contact_filter_enabled,
            no_contact_cut_seconds=args.no_contact_cut_seconds,
            contact_filter_buffer_seconds=args.contact_filter_buffer_seconds,
            pre_contact_buffer_seconds=pre_contact_buffer_seconds,
            post_contact_buffer_seconds=post_contact_buffer_seconds,
            contact_filter_cache_dir=contact_filter_cache_dir,
            contact_filter_cache_only=args.contact_filter_cache_only,
            replay_memory_bytes=(
                None
                if args.replay_memory_mb <= 0.0
                else int(args.replay_memory_mb * 1024.0 * 1024.0)
            ),
            scene_overrides=scene_overrides,
            progress=_progress,
        )

    episode: ReplayEpisode | None = None
    episode_index = 0
    first_error: BaseException | None = None
    for candidate_index in range(len(episode_paths)):
        try:
            episode = _load_episode_at(candidate_index)
        except (Exception, SystemExit) as exc:
            if first_error is None:
                first_error = exc
            print(f"Skipping episode {candidate_index + 1}/{len(episode_paths)}: {exc}")
            continue
        episode_index = candidate_index
        break
    if episode is None:
        raise SystemExit(f"No loadable episodes found under {args.trajectory}: {first_error}")
    _print_episode_summary(episode)
    mode_label = "kinematic state replay" if args.kinematic else "open-loop action rollout"
    print(f"Mode: {mode_label} (visualize_ik={args.visualize_ik})")

    if args.mp4_out is not None:
        out_path = args.mp4_out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _render_mp4(
            episode.model,
            episode.data,
            episode.arrays,
            start=episode.start,
            end=episode.end,
            fps=1.0 / episode.frame_dt,
            out_path=out_path,
            width=args.width,
            height=args.height,
            camera=_camera_spec(args.camera),
            ik_control_override=episode.ik_control_override,
            kinematic=args.kinematic,
            future_arm=_make_future_arm(episode, args.future_arm_offset),
        )
        print(f"Wrote MP4: {out_path}")

    if not args.no_viser:
        _run_viser(
            episode,
            episode_paths=episode_paths,
            episode_options=episode_options,
            episode_index=episode_index,
            load_episode=_load_episode_at,
            speed=args.speed,
            host=args.host,
            port=args.port,
            show_mocap_skeleton=args.show_mocap_skeleton,
            visualize_ik=args.visualize_ik,
            kinematic=args.kinematic,
            future_arm_offset=args.future_arm_offset,
        )


if __name__ == "__main__":
    main()
