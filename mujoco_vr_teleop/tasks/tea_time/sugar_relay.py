"""tea_time/sugar_relay."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/sugar_relay",
    title="Sugar relay",
    instruction=(
        "Lift the sugar bowl lid with one hand and hold it aside.\n"
        "With the other hand, dip a spoon into the bowl, then drop the spoon into a cup.\n"
        "Dip the other spoon, drop it into the other cup.\n"
        "Replace the sugar lid."
    ),
    reset=reset,
    skill="bimanual coordinated dispensing",
    template="One hand holds lid; the other dispenses sugar via spoon.",
    target_duration_s=300.0,
)
