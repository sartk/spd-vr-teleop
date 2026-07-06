"""Core task types.

A `TaskSpec` is the single source of truth for one VR teleop task:
    - reset:   how to initialize the scene (imperative function on model+data)
    - text/metadata: instruction, title, target_duration, etc.

`RewardContext` is the per-tick state passed to reward predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Reward evaluation context
# ---------------------------------------------------------------------------

@dataclass
class RewardContext:
    """Per-tick state passed to every rubric predicate.

    `scratch` is shared across the whole task (one dict per episode); criteria
    that need memory (e.g. sticky 'block was lifted') stash flags here.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    scratch: dict[str, Any]
    elapsed: float
    physics_dt: float

    def body_pos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in model")
        return np.asarray(self.data.xpos[bid]).copy()

    def body_quat(self, name: str) -> np.ndarray:
        """Body world quaternion (w, x, y, z)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in model")
        return np.asarray(self.data.xquat[bid]).copy()

    def body_lin_vel(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in model")
        return np.asarray(self.data.cvel[bid, 3:6]).copy()

    def body_ang_vel(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in model")
        return np.asarray(self.data.cvel[bid, 0:3]).copy()


# ---------------------------------------------------------------------------
# Preview spec: translucent ghost poses, rendered client-side
# ---------------------------------------------------------------------------

@dataclass
class BodyPoseTarget:
    body: str
    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None  # w, x, y, z


@dataclass
class PreviewSpec:
    targets: list[BodyPoseTarget] = field(default_factory=list)

    def to_payload(self) -> list[dict]:
        return [
            {
                "body": t.body,
                "pos": list(t.pos) if t.pos is not None else None,
                "quat": list(t.quat) if t.quat is not None else None,
            }
            for t in self.targets
        ]


# ---------------------------------------------------------------------------
# TaskSpec: the full task definition
# ---------------------------------------------------------------------------

ResetFn = Callable[[mujoco.MjModel, mujoco.MjData, np.random.Generator], None]


@dataclass
class TaskSpec:
    id: str  # "<scene>/<task>", e.g. "jenga/tower"
    title: str
    instruction: str
    difficulty: str
    skill: str
    target_duration_s: float
    probability: float
    reset: ResetFn
    template: str = ""
    # Per-task DR overrides; if None, the scene default applies.
    # Conventional keys: "size", "friction".
    randomize_overrides: dict[str, bool] | None = None

    @property
    def scene(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def short_name(self) -> str:
        return self.id.split("/", 1)[1] if "/" in self.id else self.id

    def to_dict(self) -> dict:
        """Dict view of the spec, consumed by TaskRegistryManager and the
        recorder/streamer task metadata."""
        return {
            "id": self.id,
            "title": self.title,
            "instruction": self.instruction,
            "difficulty": self.difficulty,
            "skill": self.skill,
            "template": self.template,
            "parameters": {"target_duration_s": self.target_duration_s},
            "probability": self.probability,
        }


__all__ = [
    "BodyPoseTarget",
    "PreviewSpec",
    "ResetFn",
    "RewardContext",
    "TaskSpec",
]
