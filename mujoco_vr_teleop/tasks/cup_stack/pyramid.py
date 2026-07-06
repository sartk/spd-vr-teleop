"""cup_stack/pyramid — build a 3-2-1 cup triangle.

Six cups start scattered on the table. The operator's goal is to arrange them
upright into a flat 3-2-1 triangle.
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
    task_id="cup_stack/pyramid",
    title="Build a cup pyramid",
    instruction="Arrange the six cups into a 3-2-1 triangle.",
    reset=reset,
    skill="bimanual arrangement",
    target_duration_s=180.0,
)
