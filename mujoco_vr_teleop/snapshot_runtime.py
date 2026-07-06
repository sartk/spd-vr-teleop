"""Per-task snapshot state for choreographer mode.

The choreographer presses A to capture flat-numbered still images of the
current scene state. Snapshots live under
``examples/task_scenes/<scene>/<short>/snapshot_NN.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mujoco_vr_teleop.tasks.base import TaskSpec


@dataclass
class SnapshotRuntime:
    spec: TaskSpec | None
    snapshot_root: Path
    # False until the first snapshot is captured for this runtime instance.
    # The first capture wipes any snapshots left over from a prior session so
    # a re-recorded task starts from snapshot_01; later captures append.
    _captured_this_session: bool = False

    def task_dir(self) -> Path | None:
        if self.spec is None:
            return None
        return self.snapshot_root / self.spec.scene / self.spec.short_name

    def existing_snapshots(self) -> list[Path]:
        d = self.task_dir()
        if d is None or not d.exists():
            return []
        return sorted(d.glob("snapshot_*.png"))

    def next_snapshot_path(self) -> Path | None:
        """Path for the next capture. The first call in a session clears any
        stale snapshots so numbering restarts at 01."""
        d = self.task_dir()
        if d is None:
            return None
        if not self._captured_this_session:
            for stale in self.existing_snapshots():
                stale.unlink()
            self._captured_this_session = True
        return d / f"snapshot_{len(self.existing_snapshots()) + 1:02d}.png"

    def snapshot_urls(self) -> list[str]:
        if self.spec is None:
            return []
        scene, short = self.spec.scene, self.spec.short_name
        # ``?v=<mtime>`` busts the browser + frontend image cache when a
        # re-recorded snapshot reuses an existing filename.
        return [
            f"/snapshots/{scene}/{short}/{p.name}?v={int(p.stat().st_mtime)}"
            for p in self.existing_snapshots()
        ]

    def status_payload(self) -> dict | None:
        if self.spec is None:
            return None
        return {
            "task_id": self.spec.id,
            "snapshots": self.snapshot_urls(),
        }


def hide_arm_geoms(model) -> None:
    """Make the arm-link geoms invisible so they don't occlude the task in
    snapshots. The 'sharpa' hands are left visible. Clears the material id so
    the transparent geom_rgba actually takes effect during rendering."""
    import mujoco
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
        if "sharpa" not in name and ("right-arm" in name or "left-arm" in name):
            model.geom_matid[gid] = -1
            model.geom_rgba[gid] = (0.0, 0.0, 0.0, 0.0)


def render_snapshot(model, data, out_path: Path, *, camera: str = "mid",
                     width: int = 1024, height: int = 640) -> Path:
    """Render the current scene state to ``out_path`` from the given camera."""
    import mujoco
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"snapshot camera {camera!r} not found in scene")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=cam)
        Image.fromarray(np.asarray(r.render())).save(out_path)
    return out_path
