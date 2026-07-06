"""Custom task manager for the dishrack scene.

Forced rack→plate cycle, no operator choice between tasks:

    [variant pick + scatter] → #1 rack_dishes
        complete (same variants, snapshot end-state) ─→ #2 plate_dishes
        skip     (new variants + scatter) ─────────→ #1 rack_dishes
    #2 plate_dishes
        complete (new variants + scatter) ─────────→ #1 rack_dishes
        skip     (restore #2's start state) ───────→ #2 plate_dishes

The streamer detects forced-cycle managers by ``hasattr(manager,
"on_task_marked")``. Anything with that method participates in the
no-modal forced advance. Anything without uses the default modal flow.

This implements the small slice of TaskRegistryManager's interface the
streamer reads (current_task, current_spec, task_info, next_task,
prev_task, specs, tasks, set_start_mode) — no inheritance so we don't
inherit any surprise side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks import get_task


TASK_IDS: tuple[str, str] = (
    "dishrack/rack_dishes",
    "dishrack/plate_dishes",
)


@dataclass
class TaskDirective:
    """Tells the streamer how to advance after a task is marked complete or
    skipped under a forced-cycle manager."""
    next_task_id: str
    redo_variants: bool       # call reset_scene → new variants + scatter
    restore_snapshot: bool    # restore the manager's saved physics state


def _snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    snap = {
        "qpos": np.asarray(data.qpos, dtype=float).copy(),
        "qvel": np.asarray(data.qvel, dtype=float).copy(),
        "ctrl": np.asarray(data.ctrl, dtype=float).copy(),
    }
    if model.nmocap > 0:
        snap["mocap_pos"] = np.asarray(data.mocap_pos, dtype=float).copy()
        snap["mocap_quat"] = np.asarray(data.mocap_quat, dtype=float).copy()
    return snap


def restore_snapshot(model: mujoco.MjModel, data: mujoco.MjData,
                      snap: dict) -> None:
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    data.ctrl[:] = snap["ctrl"]
    if "mocap_pos" in snap and model.nmocap > 0:
        data.mocap_pos[:] = snap["mocap_pos"]
        data.mocap_quat[:] = snap["mocap_quat"]
    mujoco.mj_forward(model, data)


class DishrackTaskManager:
    """Forced 2-task cycle for the dishrack scene. Duck-typed compatible
    with TaskRegistryManager for the bits the streamer reads."""

    def __init__(self, scene: str, ordering: str = "random"):
        self.scene = scene
        self.ordering = ordering
        self.specs = [get_task(tid) for tid in TASK_IDS]
        missing = [tid for tid, s in zip(TASK_IDS, self.specs) if s is None]
        if missing:
            raise RuntimeError(
                f"DishrackTaskManager: missing tasks {missing}"
            )
        self.tasks = [s.to_dict() for s in self.specs]
        self._pointer = 0          # 0 = rack, 1 = plate
        self.start_mode: str | None = None
        self.snapshot: dict | None = None
        print(f"DishrackTaskManager: {len(self.specs)} tasks, "
              f"starting on {TASK_IDS[self._pointer]}")

    # ---- TaskRegistryManager-compatible interface ------------------- #

    @property
    def current_task(self) -> dict | None:
        return self.tasks[self._pointer]

    @property
    def current_spec(self):
        return self.specs[self._pointer]

    @property
    def current_index(self) -> int:
        # Exposed so the streamer's status payload can render "N/M".
        return self._pointer

    @property
    def task_info(self) -> dict:
        t = self.current_task
        return {
            "task_id": t.get("id"),
            "title": t.get("title", ""),
            "instruction": t.get("instruction", ""),
            "difficulty": t.get("difficulty", ""),
            "skill": t.get("skill", ""),
            "template": t.get("template", ""),
            "parameters": t.get("parameters", {}),
            "task_index": self._pointer,
            "total_tasks": len(self.specs),
        }

    def next_task(self) -> dict | None:
        # No-op in forced-cycle mode; advance is driven by on_task_marked().
        return self.current_task

    def prev_task(self) -> dict | None:
        return self.current_task

    def set_start_mode(self, start_mode: str | None) -> dict | None:
        self.start_mode = start_mode
        return self.current_task

    # ---- Forced-cycle driver --------------------------------------- #

    def on_task_marked(self, result: Literal["complete", "skipped"],
                        model: mujoco.MjModel,
                        data: mujoco.MjData) -> TaskDirective:
        """Update internal state per the cycle rules. Returns a directive
        the streamer applies."""
        at_rack = self._pointer == 0
        at_plate = self._pointer == 1

        if at_rack and result == "complete":
            # Snapshot scene at end of rack, advance to plate.
            self.snapshot = _snapshot(model, data)
            self._pointer = 1
            return TaskDirective(TASK_IDS[1], redo_variants=False,
                                  restore_snapshot=False)

        if at_rack and result == "skipped":
            # Redo rack with fresh variants + scatter.
            self.snapshot = None
            return TaskDirective(TASK_IDS[0], redo_variants=True,
                                  restore_snapshot=False)

        if at_plate and result == "complete":
            # Cycle complete: new variants + scatter, back to rack.
            self.snapshot = None
            self._pointer = 0
            return TaskDirective(TASK_IDS[0], redo_variants=True,
                                  restore_snapshot=False)

        if at_plate and result == "skipped":
            # Restore plate's start state (= snapshot from rack-complete).
            return TaskDirective(
                TASK_IDS[1], redo_variants=False,
                restore_snapshot=self.snapshot is not None,
            )

        # Defensive fallback.
        return TaskDirective(self.current_task["id"],
                              redo_variants=False, restore_snapshot=False)
