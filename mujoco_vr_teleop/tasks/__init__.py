"""Task definitions: one TaskSpec per (scene, task).

Each scene has its own subpackage (e.g. ``tasks/jenga/``) with one module per
task. Each module exposes ``TASK: TaskSpec``. The registry below discovers
them lazily.

Task IDs are ``<scene>/<task>`` (e.g. ``jenga/tower``).
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

from mujoco_vr_teleop.tasks.base import (
    RewardContext,
    TaskSpec,
)


_REGISTRY: dict[str, TaskSpec] | None = None


def _discover() -> dict[str, TaskSpec]:
    """Walk every scene subpackage of ``mujoco_vr_teleop.tasks``, import each
    submodule, and collect any module-level ``TASK`` attribute."""
    out: dict[str, TaskSpec] = {}
    pkg = import_module("mujoco_vr_teleop.tasks")
    for scene_info in iter_modules(pkg.__path__):
        if not scene_info.ispkg:
            continue
        scene_pkg = import_module(f"mujoco_vr_teleop.tasks.{scene_info.name}")
        for mod_info in iter_modules(scene_pkg.__path__):
            if mod_info.name.startswith("_") or mod_info.name == "common":
                continue
            try:
                mod = import_module(f"mujoco_vr_teleop.tasks.{scene_info.name}.{mod_info.name}")
            except Exception as exc:
                print(f"[tasks] failed to import {scene_info.name}.{mod_info.name}: {exc}")
                continue
            task = getattr(mod, "TASK", None)
            if not isinstance(task, TaskSpec):
                continue
            if not task.id.startswith(f"{scene_info.name}/"):
                print(f"[tasks] {mod_info.name}: TASK.id {task.id!r} does not match scene {scene_info.name!r}")
                continue
            out[task.id] = task
    return out


def all_tasks() -> dict[str, TaskSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def get_task(task_id: str) -> TaskSpec | None:
    return all_tasks().get(task_id)


def tasks_for_scene(scene: str) -> list[TaskSpec]:
    prefix = f"{scene}/"
    return [t for tid, t in all_tasks().items() if tid.startswith(prefix)]


def scene_randomization(scene: str) -> dict[str, bool]:
    """Per-scene DR defaults declared in tasks/<scene>/__init__.py via
    SCENE_RANDOMIZATION = {"size": True, "friction": True}. Returns {} if
    nothing declared."""
    try:
        mod = import_module(f"mujoco_vr_teleop.tasks.{scene}")
    except Exception:
        return {}
    return dict(getattr(mod, "SCENE_RANDOMIZATION", {}))


def randomization_for(task_id: str) -> dict[str, bool]:
    """Effective DR settings for a task: scene defaults overridden by the
    task's own ``randomize_overrides`` (if any)."""
    spec = get_task(task_id)
    if spec is None:
        return {}
    base = scene_randomization(spec.scene)
    if spec.randomize_overrides:
        base.update(spec.randomize_overrides)
    return base


__all__ = [
    "RewardContext",
    "TaskSpec",
    "all_tasks",
    "get_task",
    "randomization_for",
    "scene_randomization",
    "tasks_for_scene",
]
