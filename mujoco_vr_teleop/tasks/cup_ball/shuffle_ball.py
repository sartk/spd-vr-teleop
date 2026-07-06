"""cup_ball/shuffle_ball — the shell game.

Three upside-down cups start in a row with the ball hidden under a random one.
The operator lifts cups to find the ball, does a quick set of shuffles, then
lifts the cup hiding the ball.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import row_cups
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    row_cups(model, data, rng)


TASK = make_task(
    task_id="cup_ball/shuffle_ball",
    title="Shell Game",
    instruction=(
        "Find the ball, give the cups a quick shuffle, then lift the cup "
        "hiding the ball."
    ),
    reset=reset,
    skill="bimanual manipulation",
    target_duration_s=120.0,
)
