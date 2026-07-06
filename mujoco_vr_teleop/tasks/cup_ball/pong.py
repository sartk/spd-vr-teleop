"""cup_ball/pong — throw the ball into a cup.

Six cups start arranged in a 3-2-1 triangle (a pong rack) against the back of
the table, with the ball near the operator. The operator throws the ball to
land in one of the cups.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import triangle_cups
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    triangle_cups(model, data, rng, jitter=True)


TASK = make_task(
    task_id="cup_ball/pong",
    title="Cup pong",
    instruction="Throw the ball so it lands in one of the cups.",
    reset=reset,
    skill="throwing",
    target_duration_s=120.0,
)
