"""Trajectory recording and session management for MuJoCo VR teleop.

Records per-frame MuJoCo state to zarr files with metadata.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import attrs
import mujoco
import numpy as np
import zarr


# Default MuJoCo data fields to capture per frame
DEFAULT_FRAME_KEYS = [
    "mocap_pos",
    "mocap_quat",
    "qpos",
    "qvel",
    "ctrl",
    "sensordata",
    "site_xpos",
    "site_xmat",
]


def save_keyframe_file(path: str | Path, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Save current MuJoCo state as a reusable keyframe JSON file.

    This file lives next to the scene XML and can be loaded on future runs
    via --keyframe-file to restore the same initial state.
    """
    kf = {
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
    }
    if model.nmocap > 0:
        kf["mocap_pos"] = data.mocap_pos.tolist()
        kf["mocap_quat"] = data.mocap_quat.tolist()
    with open(path, "w") as f:
        json.dump(kf, f, indent=2)
    print(f"Saved keyframe file: {path}")


def load_keyframe_file(path: str | Path, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Load a keyframe JSON file and apply it to the MuJoCo data.

    Restores qpos, qvel, ctrl, and optionally mocap_pos/mocap_quat.
    """
    with open(path) as f:
        kf = json.load(f)
    if "qpos" in kf:
        data.qpos[:] = np.array(kf["qpos"])
    if "qvel" in kf:
        data.qvel[:] = np.array(kf["qvel"])
    if "ctrl" in kf:
        data.ctrl[:] = np.array(kf["ctrl"])
    if "mocap_pos" in kf and model.nmocap > 0:
        data.mocap_pos[:] = np.array(kf["mocap_pos"])
    if "mocap_quat" in kf and model.nmocap > 0:
        data.mocap_quat[:] = np.array(kf["mocap_quat"])
    mujoco.mj_forward(model, data)
    print(f"Loaded keyframe file: {path}")


def get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def capture_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_keys: list[str],
    physics_step: int,
    extra: dict[str, np.ndarray] | None = None,
) -> dict:
    """Capture a single frame of MuJoCo state."""
    frame = {
        "timestamp": time.time(),
        "sim_time": float(data.time),
        "physics_step": int(physics_step),
    }
    for key in frame_keys:
        arr = getattr(data, key, None)
        if arr is not None and arr.size > 0:
            frame[key] = np.array(arr).copy()
    if extra:
        for key, value in extra.items():
            frame[key] = np.asarray(value).copy()
    return frame


def frames_to_arrays(frames: list[dict]) -> dict[str, np.ndarray]:
    """Convert a list of frame dicts to stacked numpy arrays."""
    if not frames:
        return {}
    keys = [k for k in frames[0].keys() if k != "timestamp"]
    arrays = {}
    arrays["timestamps"] = np.array([f["timestamp"] for f in frames], dtype=np.float64)
    for key in keys:
        arrays[key] = np.stack([f[key] for f in frames])
    return arrays


@attrs.define
class _PendingFailure:
    """One discarded suffix waiting to be flushed into the next episode zarr."""
    frames: list[dict]
    fork_frame: int
    fork_marker: int
    fork_timestamp: float
    task_id: str


def write_failures_group(root: "zarr.Group", pending: list[_PendingFailure]) -> int:
    """Write ``pending`` segments into ``root['failures']``.

    Layout: ``failures/trajectory/<col>`` is one long concatenated stream per
    column; ``failures/segments/{start_index, length, fork_frame,
    fork_marker, fork_timestamp, task_id}`` indexes it. Slice segment i with
    ``trajectory[col][start_index[i] : start_index[i] + length[i]]``.

    Returns total failure frames written.
    """
    if not pending:
        return 0
    lengths = np.asarray([len(p.frames) for p in pending], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    total = int(lengths.sum())

    failures = root.create_group("failures")
    traj = failures.create_group("trajectory")
    # Concat each column across segments, then write once.
    per_segment = [frames_to_arrays(p.frames) for p in pending]
    for col in per_segment[0]:
        stacked = np.concatenate([seg[col] for seg in per_segment], axis=0)
        traj.create_dataset(col, data=stacked)

    seg = failures.create_group("segments")
    seg.create_dataset("start_index", data=starts)
    seg.create_dataset("length", data=lengths)
    seg.create_dataset("fork_frame", data=np.asarray([p.fork_frame for p in pending], np.int64))
    seg.create_dataset("fork_marker", data=np.asarray([p.fork_marker for p in pending], np.int64))
    seg.create_dataset("fork_timestamp", data=np.asarray([p.fork_timestamp for p in pending], np.float64))
    seg.create_dataset("task_id", data=np.asarray([p.task_id[:64] for p in pending], dtype="U64"))
    return total


def gs6_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternions (N, 4) xyzw to 6D Gram-Schmidt rotation (N, 6).

    Continuous rotation representation for ML training.
    """
    from scipy.spatial.transform import Rotation
    if quat.ndim == 1:
        quat = quat[np.newaxis]
    mats = Rotation.from_quat(quat).as_matrix()  # (N, 3, 3)
    # First two columns of the rotation matrix
    return np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=-1)  # (N, 6)


class Session:
    """Manages a demo collection session with episode counting and directory structure."""

    def __init__(
        self,
        output_dir: str | Path,
        session_name: str = "default",
        frame_keys: list[str] | None = None,
        head_clip: int = 10,
        command: str = "",
        save_gs6: bool = False,
        xml_path: str | Path | None = None,
        streamer_config: dict | None = None,
    ):
        self.frame_keys = frame_keys or list(DEFAULT_FRAME_KEYS)
        self.head_clip = head_clip
        self.command = command
        self.save_gs6 = save_gs6
        self._xml_path = Path(xml_path) if xml_path else None
        self.xml_name = self._xml_path.name if self._xml_path else None
        self.streamer_config = copy.deepcopy(streamer_config) if streamer_config is not None else None

        # Session directory (created lazily on first save)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.session_dir = Path(output_dir) / session_name / timestamp
        self._dir_created = False
        # Optional one-shot callback fired after the session directory is created.
        # Used by the streamer to attach .log files to the dir.
        self.on_dir_created: list = []

        self.episode_counter = 0
        self.git_commit = get_git_commit()

        # Current recording state
        self._frames: list[dict] = []
        self._recording = False
        self._event_markers: dict[str, list[float]] = {}
        self._marker_frame_indices: list[int] = []
        self._marker_timestamps: list[float] = []
        self._recording_started_at: float | None = None
        self._pause_started_at: float | None = None
        self._paused_seconds = 0.0
        self._last_marker_elapsed: float | None = None
        self._saved_recorded_seconds = 0.0
        self._saved_marker_count = 0
        self._episode_initial_state: dict | None = None
        self._checkpoint_frame_count = 0
        self._checkpoint_marker_count = 0
        self._checkpoint_last_marker_elapsed: float | None = None
        self._checkpoint_recording_elapsed = 0.0

        # Reverted segments accumulate here in memory and get flushed into
        # the next saved episode's zarr under a `failures/` subgroup.
        self._pending_failures: list[_PendingFailure] = []

        print(f"Session dir: {self.session_dir} (created on first save)")
        print(f"Frame keys: {self.frame_keys}")
        print(f"Head clip: {self.head_clip} frames")

    def _ensure_dir(self) -> None:
        """Create session directory and write metadata on first use."""
        if self._dir_created:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._dir_created = True

        # Copy scene XML
        if self._xml_path is not None:
            shutil.copy2(self._xml_path, self.session_dir / self._xml_path.name)
            print(f"Saved scene XML: {self.session_dir / self._xml_path.name}")

        self._write_session_info()
        self._write_streamer_config()
        self._flush_keyframe()
        print(f"Session dir created: {self.session_dir}")
        for cb in self.on_dir_created:
            try:
                cb(self.session_dir)
            except Exception as exc:
                print(f"on_dir_created callback failed: {exc}")

    def _streamer_config_payload(self) -> dict:
        return {
            "git_commit": self.git_commit,
            "config": self.streamer_config,
        }

    def update_variant_pools(self, variant_pools: dict | None) -> None:
        """Refresh the recorded variant-pool selection for the next episode.

        ``streamer_config`` is captured once at session start, but the scene
        is rebuilt per episode with freshly re-rolled variant pools (bottle
        count + per-pool variant ids). The streamer must call this after every
        scene rebuild so each episode's saved ``streamer_config`` carries the
        ``chosen_variants`` of the scene it actually recorded against. Without
        it every episode inherits the first build's stale selection.
        """
        if self.streamer_config is None or not variant_pools:
            return
        # domain_randomization may be a bool (DR disabled) rather than a dict;
        # only the dict form has a variant_pools section to refresh.
        dr = self.streamer_config.get("domain_randomization")
        if not isinstance(dr, dict):
            dr = {}
            self.streamer_config["domain_randomization"] = dr
        dr["variant_pools"] = copy.deepcopy(variant_pools)

    def update_scene_xml(self, xml_path: str | Path | None) -> None:
        """Point the recorder at the current scene XML for the next episode.

        The scene is rebuilt per episode (fresh variant pools), so each episode
        is recorded against a different generated XML. The streamer must call
        this after every rebuild; ``save_episode`` then copies that XML next to
        the episode zarr so replay loads the exact scene it recorded against.
        """
        if xml_path is None:
            return
        self._xml_path = Path(xml_path)
        self.xml_name = self._xml_path.name

    def _episode_replay_config_payload(self, episode_index: int, zarr_name: str, head_clip: int) -> dict:
        return {
            "episode_index": episode_index,
            "zarr": zarr_name,
            "scene_xml": self.xml_name,
            "git_commit": self.git_commit,
            "command": self.command,
            "frame_keys": self.frame_keys,
            "head_clip": head_clip,
            "save_gs6": self.save_gs6,
            "streamer_config": self._streamer_config_payload(),
        }

    def _write_streamer_config(self) -> None:
        if self.streamer_config is None:
            return
        config_path = self.session_dir / "streamer_config.json"
        with open(config_path, "w") as f:
            json.dump(self._streamer_config_payload(), f, indent=2)

    def _write_session_info(self) -> None:
        """Write session metadata to a JSON file."""
        info = {
            "command": self.command,
            "git_commit": self.git_commit,
            "frame_keys": self.frame_keys,
            "head_clip": self.head_clip,
            "save_gs6": self.save_gs6,
            "start_time": datetime.now().isoformat(),
        }
        if self.xml_name:
            info["scene_xml"] = self.xml_name
        if self.streamer_config is not None:
            info["streamer_config_file"] = "streamer_config.json"
        info_path = self.session_dir / "session_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def save_keyframe(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        domain_randomization: dict | None = None,
        body_scale: np.ndarray | None = None,
        body_names: list[str] | None = None,
    ) -> None:
        """Store keyframe state. Written to disk when session dir is created.

        The resolved scene geometry is not snapshotted — replay reconstructs it
        from the domain_randomization sample log via the scene builder.
        """
        self._pending_keyframe = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy()}
        if model.nmocap > 0:
            self._pending_keyframe["mocap_pos"] = data.mocap_pos.copy()
            self._pending_keyframe["mocap_quat"] = data.mocap_quat.copy()
        self._episode_initial_state = {
            key: value.copy()
            for key, value in self._pending_keyframe.items()
        }
        if body_scale is not None:
            self._episode_initial_state["body_scale"] = np.asarray(body_scale, dtype=float).copy()
        if body_names is not None:
            self._episode_initial_state["body_names"] = list(body_names)
        if domain_randomization is not None:
            self._episode_initial_state["domain_randomization"] = copy.deepcopy(domain_randomization)

    def _flush_keyframe(self) -> None:
        """Write pending keyframe to disk if session dir exists."""
        if hasattr(self, "_pending_keyframe") and self._pending_keyframe:
            kf_path = self.session_dir / "keyframe.npz"
            np.savez(str(kf_path), **self._pending_keyframe)
            print(f"Saved keyframe: {kf_path}")

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def marker_count(self) -> int:
        return len(self._marker_frame_indices)

    @property
    def total_marker_count(self) -> int:
        return self._saved_marker_count + len(self._marker_frame_indices)

    @property
    def last_marker_time(self) -> float | None:
        if not self._marker_timestamps:
            return None
        return self._marker_timestamps[-1]

    @property
    def recording_elapsed(self) -> float | None:
        if self._recording_started_at is None:
            return None
        now = self._pause_started_at if self._pause_started_at is not None else time.time()
        return max(0.0, now - self._recording_started_at - self._paused_seconds)

    @property
    def seconds_since_marker(self) -> float | None:
        elapsed = self.recording_elapsed
        if elapsed is None:
            return None
        if self._last_marker_elapsed is None:
            return elapsed
        return max(0.0, elapsed - self._last_marker_elapsed)

    @property
    def total_recorded_seconds(self) -> float:
        return self._saved_recorded_seconds + float(self.recording_elapsed or 0.0)

    def start_recording(self) -> None:
        """Start recording frames."""
        self._frames = []
        self._event_markers = {}
        self._marker_frame_indices = []
        self._marker_timestamps = []
        self._recording_started_at = time.time()
        self._pause_started_at = None
        self._paused_seconds = 0.0
        self._last_marker_elapsed = None
        self._checkpoint_frame_count = 0
        self._checkpoint_marker_count = 0
        self._checkpoint_last_marker_elapsed = None
        self._checkpoint_recording_elapsed = 0.0
        self._recording = True
        print(f"Recording started (episode {self.episode_counter})")

    def stop_recording(self) -> None:
        """Stop recording without saving."""
        self._recording = False
        self._pause_started_at = None
        print(f"Recording stopped ({self.frame_count} frames)")

    def pause_recording(self) -> None:
        """Pause recording without clearing buffered frames."""
        if not self._recording:
            return
        self._recording = False
        self._pause_started_at = time.time()
        print(f"Recording paused ({self.frame_count} frames)")

    def resume_recording(self) -> None:
        """Resume recording into the existing frame buffer."""
        if self._recording:
            return
        if self._recording_started_at is None:
            self.start_recording()
            return
        if self._pause_started_at is not None:
            self._paused_seconds += time.time() - self._pause_started_at
            self._pause_started_at = None
        self._recording = True
        print(f"Recording resumed ({self.frame_count} frames)")

    def record_frame(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        physics_step: int,
        extra: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Capture and store a frame if recording is active."""
        if not self._recording:
            return
        self._frames.append(capture_frame(model, data, self.frame_keys, physics_step, extra))

    def add_frame_extra(self, frame_idx: int, extra: dict[str, np.ndarray]) -> None:
        if not (0 <= frame_idx < len(self._frames)):
            return
        for key, value in extra.items():
            self._frames[frame_idx][key] = np.asarray(value).copy()

    def write_frame_ctrl(self, frame_idx: int, act_ids: list[int], values: np.ndarray) -> None:
        if not (0 <= frame_idx < len(self._frames)):
            return
        ctrl = self._frames[frame_idx].get("ctrl")
        if ctrl is None:
            return
        ctrl = np.asarray(ctrl).copy()
        ids = np.asarray(act_ids, dtype=int)
        vals = np.asarray(values, dtype=float).reshape(-1)
        n = min(ids.size, vals.size)
        if n <= 0:
            return
        ids = ids[:n]
        vals = vals[:n]
        valid = np.isfinite(vals) & (ids >= 0) & (ids < ctrl.shape[0])
        ctrl[ids[valid]] = vals[valid]
        self._frames[frame_idx]["ctrl"] = ctrl

    def mark(self) -> int | None:
        """Mark the latest recorded frame as a checkpoint."""
        if not self._recording or not self._frames:
            return None
        frame_idx = len(self._frames) - 1
        if self._marker_frame_indices and self._marker_frame_indices[-1] == frame_idx:
            return frame_idx
        self._marker_frame_indices.append(frame_idx)
        self._marker_timestamps.append(time.time())
        self._last_marker_elapsed = self.recording_elapsed
        print(f"Marked checkpoint at frame {frame_idx}")
        return frame_idx

    def _trim_to_last_marker(self) -> None:
        if not self._marker_frame_indices:
            return
        keep_count = self._marker_frame_indices[-1] + 1
        dropped = max(0, len(self._frames) - keep_count)
        if dropped:
            self._frames = self._frames[:keep_count]
            print(f"Discarded {dropped} frames after last checkpoint")

    def discard_to_checkpoint(self, *, task_info: dict | None = None) -> None:
        # Buffer dropped frames for the next save_episode to flush into
        # `failures/`. Then truncate live state back to the last checkpoint.
        dropped_frames = self._frames[self._checkpoint_frame_count:]
        if dropped_frames:
            self._pending_failures.append(_PendingFailure(
                frames=list(dropped_frames),
                fork_frame=self._checkpoint_frame_count,
                fork_marker=self._checkpoint_marker_count,
                fork_timestamp=time.time(),
                task_id=str((task_info or {}).get("task_id", "") or ""),
            ))
        self._frames = self._frames[:self._checkpoint_frame_count]
        self._marker_frame_indices = self._marker_frame_indices[:self._checkpoint_marker_count]
        self._marker_timestamps = self._marker_timestamps[:self._checkpoint_marker_count]
        self._last_marker_elapsed = self._checkpoint_last_marker_elapsed
        now = time.time()
        self._recording_started_at = now - self._checkpoint_recording_elapsed
        self._pause_started_at = now
        self._paused_seconds = 0.0
        self._recording = False
        print(f"Discarded {len(dropped_frames)} frames; buffered for next failures/")

    def add_event_marker(self, name: str) -> None:
        """Add a timestamped event marker (e.g., button press)."""
        if name not in self._event_markers:
            self._event_markers[name] = []
        self._event_markers[name].append(time.time())

    def save_episode(
        self,
        task_info: dict | None = None,
        *,
        trim_to_last_marker: bool = False,
        clip_head: bool = True,
        finalize: bool = True,
    ) -> Optional[Path]:
        """Save the current recording as a zarr episode. Returns path or None if empty."""
        was_recording = self._recording
        if finalize:
            self._recording = False
        if trim_to_last_marker:
            self._trim_to_last_marker()
        self._ensure_dir()

        # Clip head frames
        head_clip = self.head_clip if clip_head else 0
        frames = self._frames[head_clip:]
        if not frames:
            print(f"Episode too short ({len(self._frames)} frames, {head_clip} clipped). Discarded.")
            if finalize:
                self._frames = []
                self._marker_frame_indices = []
                self._marker_timestamps = []
                self._recording_started_at = None
                self._pause_started_at = None
                self._paused_seconds = 0.0
                self._last_marker_elapsed = None
            else:
                self._recording = was_recording
            return None

        arrays = frames_to_arrays(frames)
        if trim_to_last_marker and self._last_marker_elapsed is not None:
            saved_duration = self._last_marker_elapsed
        else:
            saved_duration = float(self.recording_elapsed or 0.0)
        marker_frames = np.zeros(len(frames), dtype=np.bool_)
        for frame_idx in self._marker_frame_indices:
            clipped_idx = frame_idx - head_clip
            if 0 <= clipped_idx < len(marker_frames):
                marker_frames[clipped_idx] = True
        arrays["marker_frames"] = marker_frames

        # Add GS6 rotation representations if requested
        if self.save_gs6:
            for key in list(arrays.keys()):
                if "quat" in key and arrays[key].ndim >= 2:
                    arr = arrays[key]
                    orig_shape = arr.shape
                    flat = arr.reshape(-1, 4)
                    gs6 = gs6_from_quat(flat)
                    arrays[key + "_gs6"] = gs6.reshape(orig_shape[:-1] + (6,))

        # Save to zarr inside a per-episode directory. Videos for this episode
        # are written next to the zarr by the background renderer.
        ep_name = f"ep_{self.episode_counter:05d}"
        ep_dir = self.session_dir / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_path = ep_dir / f"{ep_name}.zarr"
        replay_config_path = ep_dir / "replay_config.json"
        with open(replay_config_path, "w") as f:
            json.dump(
                self._episode_replay_config_payload(self.episode_counter, ep_path.name, head_clip),
                f,
                indent=2,
            )
        # Copy the scene XML this episode recorded against into the episode
        # dir. The scene is rebuilt per episode, so each episode keeps its own
        # XML rather than sharing the session's first-build copy.
        if self._xml_path is not None and self._xml_path.exists():
            shutil.copy2(self._xml_path, ep_dir / self._xml_path.name)

        root = zarr.open(str(ep_path), mode="w")

        # Metadata
        root.attrs["command"] = self.command
        root.attrs["git_commit"] = self.git_commit
        if self.xml_name:
            root.attrs["scene_xml"] = self.xml_name
        if self.streamer_config is not None:
            root.attrs["streamer_config"] = self._streamer_config_payload()
        root.attrs["replay_config"] = "replay_config.json"
        root.attrs["trajectory_length"] = len(frames)
        root.attrs["frame_keys"] = self.frame_keys
        root.attrs["head_clip"] = head_clip
        root.attrs["episode_index"] = self.episode_counter
        root.attrs["save_time"] = datetime.now().isoformat()

        if task_info:
            root.attrs["task"] = task_info
            root.attrs["task_prompt"] = str(task_info.get("instruction", ""))

        if self._episode_initial_state:
            init = root.create_group("initial_state")
            for key, value in self._episode_initial_state.items():
                if key in {"body_names", "domain_randomization"}:
                    continue
                arr = np.asarray(value)
                if arr.size == 0:
                    continue
                init.create_dataset(key, data=arr)
            if "body_names" in self._episode_initial_state:
                init.attrs["body_names"] = self._episode_initial_state["body_names"]
            if "domain_randomization" in self._episode_initial_state:
                dr_meta = self._episode_initial_state["domain_randomization"]
                init.attrs["domain_randomization"] = dr_meta
                # Provenance: surface the exact mesh path that each variant
                # pool resolved to this episode. Pulled from variant_select
                # samples so replay can verify the asset is still on disk.
                variant_paths: dict[str, str] = {}
                for sample in (dr_meta.get("samples") or []) if isinstance(dr_meta, dict) else []:
                    if sample.get("op") != "variant_select":
                        continue
                    value = sample.get("value") or {}
                    pool = value.get("pool")
                    path = value.get("mesh_path")
                    if pool and path:
                        variant_paths[str(pool)] = str(path)
                if variant_paths:
                    init.attrs["selected_variant_paths"] = variant_paths

        # Trajectory data
        traj = root.create_group("trajectory")
        if "arm_ik_target_pos" in arrays:
            traj.attrs["arm_ik_side_order"] = ["right", "left"]
            traj.attrs["arm_ik_joint_order"] = [f"joint{i}" for i in range(1, 7)]
        if "wrist_hybrid_weld_mask" in arrays:
            traj.attrs["wrist_hybrid_side_order"] = ["right", "left"]
        if "hand_ik_solution_qpos" in arrays:
            traj.attrs["hand_ik_side_order"] = ["right", "left"]
            traj.attrs["hand_ik_joint_count"] = int(arrays["hand_ik_solution_qpos"].shape[2])
        chunk_size = min(500, len(frames))
        for key, arr in arrays.items():
            if arr.size == 0:
                continue
            chunks = (chunk_size,) + arr.shape[1:]
            traj.create_dataset(key, data=arr, chunks=chunks)

        # Event markers
        if self._event_markers:
            events = root.create_group("events")
            for name, timestamps in self._event_markers.items():
                events.create_dataset(name, data=np.array(timestamps, dtype=np.float64))

        # Flush any reverted segments alongside the successful trajectory.
        failure_frames = write_failures_group(root, self._pending_failures)
        if failure_frames:
            root.attrs["failure_segments"] = len(self._pending_failures)
            root.attrs["failure_frames"] = failure_frames
            self._pending_failures = []

        action = "Saved" if finalize else "Checkpointed"
        suffix = f"; +{failure_frames} failures" if failure_frames else ""
        print(f"{action} episode {self.episode_counter}: {ep_path.name} ({len(frames)} frames{suffix})")
        if finalize:
            self._saved_recorded_seconds += saved_duration
            self._saved_marker_count += int(np.count_nonzero(marker_frames))
            self.episode_counter += 1
            self._frames = []
            self._event_markers = {}
            self._marker_frame_indices = []
            self._marker_timestamps = []
            self._recording_started_at = None
            self._pause_started_at = None
            self._paused_seconds = 0.0
            self._last_marker_elapsed = None
            self._checkpoint_frame_count = 0
            self._checkpoint_marker_count = 0
            self._checkpoint_last_marker_elapsed = None
            self._checkpoint_recording_elapsed = 0.0
        else:
            self._checkpoint_frame_count = len(self._frames)
            self._checkpoint_marker_count = len(self._marker_frame_indices)
            self._checkpoint_last_marker_elapsed = self._last_marker_elapsed
            self._checkpoint_recording_elapsed = float(self.recording_elapsed or 0.0)
            self._recording = was_recording
        return ep_path

    def save_checkpoint(self, task_info: dict | None = None) -> Optional[Path]:
        return self.save_episode(
            task_info=task_info,
            trim_to_last_marker=True,
            clip_head=False,
            finalize=False,
        )

    def finalize_checkpoint_episode(self) -> None:
        if self._checkpoint_frame_count <= 0:
            self.discard_episode()
            return
        saved_markers = self._checkpoint_marker_count
        self._saved_recorded_seconds += self._checkpoint_recording_elapsed
        self._saved_marker_count += saved_markers
        print(
            f"Finalized episode {self.episode_counter}: "
            f"{self._checkpoint_frame_count} frames, {saved_markers} checkpoints"
        )
        self.episode_counter += 1
        self._frames = []
        self._event_markers = {}
        self._marker_frame_indices = []
        self._marker_timestamps = []
        self._recording_started_at = None
        self._pause_started_at = None
        self._paused_seconds = 0.0
        self._last_marker_elapsed = None
        self._checkpoint_frame_count = 0
        self._checkpoint_marker_count = 0
        self._checkpoint_last_marker_elapsed = None
        self._checkpoint_recording_elapsed = 0.0
        # See discard_episode: drop orphaned reverted segments instead of
        # letting them accumulate or mis-attach to a later episode.
        self._pending_failures = []
        self._recording = False

    def discard_episode(self) -> None:
        """Discard the current recording without saving."""
        n = len(self._frames)
        self._frames = []
        self._event_markers = {}
        self._marker_frame_indices = []
        self._marker_timestamps = []
        self._recording_started_at = None
        self._pause_started_at = None
        self._paused_seconds = 0.0
        self._last_marker_elapsed = None
        # Reverted segments only make sense flushed into the save_episode that
        # follows their reverts. Discarding the recording orphans them, so drop
        # them too — otherwise they accumulate in RAM across the whole session.
        self._pending_failures = []
        self._recording = False
        print(f"Discarded recording ({n} frames)")


def task_allowed_start_modes(task: dict) -> list[str]:
    """Return the list of DR start modes a task accepts.

    Schema (in priority order):
    - ``"start_modes"``: explicit list. Each entry is a string. The sentinel
      value ``"continue"`` means "don't randomize — keep the current sim state".
    - ``"start_mode"`` (legacy): a single string or a list of strings.
    - absent: empty list = matches any mode (no constraint).

    Empty result means the task is mode-agnostic.
    """
    modes = task.get("start_modes")
    if isinstance(modes, list):
        return [str(m) for m in modes]
    legacy = task.get("start_mode")
    if isinstance(legacy, list):
        return [str(m) for m in legacy]
    if isinstance(legacy, str) and legacy:
        return [legacy]
    return []


def _task_has_any_snapshot(spec) -> bool:
    """True if the choreographer saved any snapshot PNG under
    examples/task_scenes/<scene>/<short>/."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    task_dir = repo_root / "examples" / "task_scenes" / spec.scene / spec.short_name
    if not task_dir.exists():
        return False
    return any(task_dir.glob("snapshot_*.png"))


class TaskRegistryManager:
    """Drop-in replacement for ``TaskManager`` that sources tasks from
    ``mujoco_vr_teleop.tasks`` instead of a JSON file.

    Two iteration modes:
        ``ordering="random"``    — weighted-without-replacement sample (matches
                                    legacy TaskManager). Used by collector mode.
        ``ordering="sequential"`` — iterate through tasks in registry order.
                                    Used by choreographer mode.
    """

    def __init__(self, scene: str, ordering: str = "random"):
        from mujoco_vr_teleop.tasks import tasks_for_scene

        self.scene = scene
        self.ordering = ordering
        # `<scene>/playground` is the playground-mode reset, not a recordable
        # task — exclude it from the task cycle in both orderings.
        all_specs = sorted(
            (s for s in tasks_for_scene(scene) if s.short_name != "playground"),
            key=lambda s: s.id,
        )
        # Collector mode requires choreographer-recorded snapshots for the
        # in-headset slideshow. Filter out tasks that have none. Choreographer
        # mode (sequential ordering) is exactly how those snapshots get
        # captured, so it sees every task.
        if ordering == "random":
            self.specs = [s for s in all_specs if _task_has_any_snapshot(s)]
            skipped = [s.id for s in all_specs if s not in self.specs]
            if skipped:
                print(f"Collector mode skipping {len(skipped)} task(s) "
                      f"with no choreographer snapshots: {skipped}")
        else:
            self.specs = all_specs
        # Legacy-shaped dict tasks for compatibility with downstream consumers.
        self.tasks = [s.to_dict() for s in self.specs]
        self.current_index: int | None = None
        self.start_mode: str | None = None
        self.rng = np.random.default_rng()
        self._pending_indices: list[int] = []
        self._history: list[int] = []
        print(f"Loaded {len(self.tasks)} tasks for scene {scene!r} "
              f"(ordering={ordering})")
        self.set_start_mode(None)

    # --- Pseudo-API for ordering shuffling ---------------------------------

    def set_start_mode(self, start_mode: str | None) -> dict | None:
        # The new tasks/ system doesn't constrain by start_mode (each task
        # has its own reset). We accept the call for API compatibility but
        # only use it to (re)initialize the queue if needed.
        self.start_mode = start_mode
        if self.current_task is not None:
            return self.current_task
        self._build_queue()
        if self._pending_indices:
            return self._select_task(self._pending_indices.pop(0))
        return None

    def _build_queue(self) -> None:
        self._pending_indices.clear()
        self._history.clear()
        if not self.specs:
            return
        if self.ordering == "sequential":
            self._pending_indices = list(range(len(self.specs)))
            return
        # Weighted permutation (matches legacy TaskManager semantics).
        weights = np.array(
            [max(0.0, float(s.probability)) for s in self.specs],
            dtype=float,
        )
        if not np.any(weights > 0):
            weights = np.ones(len(self.specs), dtype=float)
        remaining = list(range(len(self.specs)))
        remaining_weights = weights.copy()
        order: list[int] = []
        while remaining:
            total = float(remaining_weights.sum())
            if total <= 0.0:
                order.extend(int(i) for i in self.rng.permutation(remaining))
                break
            probs = remaining_weights / total
            choice = int(self.rng.choice(len(remaining), p=probs))
            order.append(int(remaining[choice]))
            del remaining[choice]
            remaining_weights = np.delete(remaining_weights, choice)
        self._pending_indices = order

    def _select_task(self, idx: int, *, remember: bool = True) -> dict:
        self.current_index = int(idx)
        if remember and (not self._history or self._history[-1] != self.current_index):
            self._history.append(self.current_index)
        spec = self.specs[self.current_index]
        print(f"Task {self.current_index + 1}/{len(self.specs)}: {spec.title}")
        return self.tasks[self.current_index]

    @property
    def current_task(self) -> dict | None:
        if not self.tasks or self.current_index is None:
            return None
        return self.tasks[self.current_index % len(self.tasks)]

    @property
    def current_spec(self):
        """The TaskSpec for the current task."""
        if not self.specs or self.current_index is None:
            return None
        return self.specs[self.current_index % len(self.specs)]

    @property
    def task_info(self) -> dict:
        task = self.current_task
        if task is None:
            return {}
        return {
            "task_id": task.get("id"),
            "title": task.get("title", ""),
            "instruction": task.get("instruction", ""),
            "difficulty": task.get("difficulty", ""),
            "skill": task.get("skill", ""),
            "template": task.get("template", ""),
            "parameters": task.get("parameters", {}),
            "task_index": self.current_index if self.current_index is not None else -1,
            "total_tasks": len(self.specs),
        }

    def next_task(self) -> dict | None:
        if not self._pending_indices:
            self._build_queue()
        if not self._pending_indices:
            return None
        return self._select_task(self._pending_indices.pop(0))

    def peek_next_task(self) -> dict | None:
        if not self._pending_indices:
            self._build_queue()
        if not self._pending_indices:
            return None
        return self.tasks[self._pending_indices[0]]

    def prev_task(self) -> dict | None:
        if len(self._history) > 1:
            self._history.pop()
            return self._select_task(self._history[-1], remember=False)
        return self.current_task

    def on_task_marked(self, result, model, data):
        """Hook called after the operator marks a task complete or skipped.

        Default (``None`` return): the streamer falls back to the post-task
        modal so the operator picks A=repeat / C=switch.

        Custom managers (e.g. dishrack's forced rack→plate cycle) override
        this to return a ``TaskDirective`` that drives the next state
        automatically, skipping the modal entirely.
        """
        return None
