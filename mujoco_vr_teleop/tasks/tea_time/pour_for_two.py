"""tea_time/pour_for_two."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/pour_for_two",
    title="Pour for two",
    instruction=(
        "Place each cup on its saucer.\n"
        "Lift the teapot with one hand.\n"
        "Tip it over the first cup, then over the second cup.\n"
        "Return the teapot to its starting spot upright."
    ),
    reset=reset,
    skill="pour + return",
    template="Pour into two cups, return teapot upright.",
    target_duration_s=300.0,
)
