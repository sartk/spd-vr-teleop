"""tea_time/stack_and_serve."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/stack_and_serve",
    title="Two place stacks",
    instruction=(
        "Build a saucer + cup + spoon-on-top column at the front-left of the table.\n"
        "Build the same column at the front-right.\n"
        "Teapot and sugar bowl stay where they are."
    ),
    reset=reset,
    skill="ordered stacking",
    template="Two saucer-cup-spoon columns, front-left and front-right.",
    target_duration_s=360.0,
)
