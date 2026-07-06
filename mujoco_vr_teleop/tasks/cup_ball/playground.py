"""cup_ball/playground — free play, no scored objective.

Cups (and a ball, if the build has one) start scattered. Used for warm-up and
unstructured data collection.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_cups
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_cups(model, data, rng)


TASK = make_task(
    task_id="cup_ball/playground",
    title="Cups playground",
    instruction="Free play with the cups.",
    reset=reset,
    skill="free play",
    target_duration_s=120.0,
)
