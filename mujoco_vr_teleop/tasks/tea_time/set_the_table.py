"""tea_time/set_the_table."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/set_the_table",
    title="Set the table",
    instruction=(
        "Place each cup on a saucer.\n"
        "Rest a spoon on each saucer rim.\n"
        "Teapot in the center back, sugar bowl beside it."
    ),
    reset=reset,
    skill="precise placement",
    template="Two place settings + center service.",
    target_duration_s=300.0,
)
