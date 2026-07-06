"""Custom task managers for the two cup scenes: cup_stack and cup_ball.

Each cup scene's build is task-driven — ``variant_pools`` maps each task to its
cup count, ball presence, and (for ball tasks) a forced opaque-plastic family.
So the scene MUST be rebuilt whenever the operator switches to a task with a
different profile; a plain modal manager (which keeps one compiled model)
cannot.

``CupStackTaskManager`` (scene ``cup_stack``): three stacking tasks run as a
forced cycle — every mark (complete or skipped) advances one task, wrapping:

    stack_two_threes -> unstack -> pyramid -> stack_two_threes -> ...

The one special case: ``stack_two_threes`` complete -> ``unstack`` keeps the
just-built stacks as unstack's start state (no rebuild).

``CupBallTaskManager`` (scene ``cup_ball``): four ball tasks, plain modal — no
forced edge, every (task, result) just rebuilds for the pointed task.

Both call ``variant_pools.set_cup_task`` before every (re)build so the build
bakes in the active task's profile, both at construction (first build) and
inside ``on_task_marked`` / ``next_task`` / ``prev_task``.

Duck-typed to the slice of TaskRegistryManager the streamer reads
(current_task, current_spec, task_info, next_task, prev_task, specs, tasks,
set_start_mode) — no inheritance, like DishrackTaskManager.
"""
from __future__ import annotations

from typing import Literal

import mujoco

from mujoco_vr_teleop import variant_pools
from mujoco_vr_teleop.dishrack_task_manager import TaskDirective
from mujoco_vr_teleop.tasks import get_task, tasks_for_scene


# Stacking tasks, stable display order.
CUP_STACK_TASK_IDS: tuple[str, ...] = (
    "cup_stack/stack_two_threes",
    "cup_stack/unstack",
    "cup_stack/pyramid",
)

# Ball tasks, stable display order. playground is intentionally excluded — it
# is free-play, reachable via the streamer's playground path.
CUP_BALL_TASK_IDS: tuple[str, ...] = (
    "cup_ball/shuffle_ball",
    "cup_ball/pong",
)


def _short(task_id: str) -> str:
    """The part after ``<scene>/`` — the key into the task profile."""
    return task_id.split("/", 1)[1]


class _CupTaskManager:
    """Shared modal-manager body for the cup scenes. Duck-typed compatible with
    TaskRegistryManager for the bits the streamer reads. Subclasses set
    ``SCENE`` and ``TASK_IDS``."""

    SCENE: str = ""
    TASK_IDS: tuple[str, ...] = ()

    def __init__(self, scene: str, ordering: str = "random"):
        if scene != self.SCENE:
            raise RuntimeError(
                f"{type(self).__name__}: scene {scene!r} != {self.SCENE!r}")
        self.scene = scene
        self.ordering = ordering
        specs = {t.id: get_task(t.id) for t in tasks_for_scene(scene)}
        self.specs = [specs[tid] for tid in self.TASK_IDS if specs.get(tid)]
        missing = [tid for tid in self.TASK_IDS if not specs.get(tid)]
        if missing:
            raise RuntimeError(f"{type(self).__name__}: missing tasks {missing}")
        self.tasks = [s.to_dict() for s in self.specs]
        self._pointer = 0
        self.start_mode: str | None = None
        self.snapshot: dict | None = None
        # Bake the first build for the starting task.
        self._sync_build_task()
        print(f"{type(self).__name__}: {len(self.specs)} tasks, "
              f"starting on {self.TASK_IDS[self._pointer]}")

    # ---- build coupling -------------------------------------------- #

    def _sync_build_task(self) -> None:
        """Tell variant_pools which task the next build_scene must bake in."""
        variant_pools.set_cup_task(
            self.SCENE, _short(self.TASK_IDS[self._pointer]))

    # ---- TaskRegistryManager-compatible interface ------------------ #

    @property
    def current_task(self) -> dict | None:
        return self.tasks[self._pointer]

    @property
    def current_spec(self):
        return self.specs[self._pointer]

    @property
    def current_index(self) -> int:
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
        self._pointer = (self._pointer + 1) % len(self.specs)
        self._sync_build_task()
        return self.current_task

    def prev_task(self) -> dict | None:
        self._pointer = (self._pointer - 1) % len(self.specs)
        self._sync_build_task()
        return self.current_task

    def set_start_mode(self, start_mode: str | None) -> dict | None:
        self.start_mode = start_mode
        return self.current_task


class CupStackTaskManager(_CupTaskManager):
    """Three-task manager for the cup_stack scene with one forced edge
    (stack_two_threes → unstack)."""

    SCENE = "cup_stack"
    TASK_IDS = CUP_STACK_TASK_IDS

    # ---- forced-cycle driver --------------------------------------- #

    def on_task_marked(self, result: Literal["complete", "skipped"],
                        model: mujoco.MjModel,
                        data: mujoco.MjData) -> TaskDirective:
        """Advance the forced cycle by one task, always — the three tasks run
        in order, wrapping (stack_two_threes -> unstack -> pyramid -> ...),
        regardless of complete/skipped.

        The one special case: stack_two_threes COMPLETE -> unstack reuses the
        current sim as unstack's start state (the just-built stacks), so no
        rebuild. Every other transition rebuilds with fresh cup variants."""
        completed_stack = (self.TASK_IDS[self._pointer]
                           == "cup_stack/stack_two_threes"
                           and result == "complete")
        self._pointer = (self._pointer + 1) % len(self.specs)
        self._sync_build_task()
        nxt = self.TASK_IDS[self._pointer]
        # stack_two_threes complete -> unstack: keep the stacks as-is, no rebuild.
        redo = not completed_stack
        return TaskDirective(nxt, redo_variants=redo, restore_snapshot=False)


class CupBallTaskManager(_CupTaskManager):
    """Plain modal manager for the cup_ball scene — no forced edge."""

    SCENE = "cup_ball"
    TASK_IDS = CUP_BALL_TASK_IDS

    def on_task_marked(self, result: Literal["complete", "skipped"],
                        model: mujoco.MjModel,
                        data: mujoco.MjData) -> TaskDirective | None:
        """No forced edge: return None so the streamer shows the post-task
        modal (A=repeat, C=switch) and the operator picks. The modal's switch
        path calls next_task(), which steps the pointer and re-syncs the build.
        """
        return None
