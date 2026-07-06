"""shapes/ball_drop."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/ball_drop",
    title="Drop balls, fish",
    instruction=(
        "Drop each cylinder into the ball-sorting toy.\n"
        "Drop each sphere into the ball-sorting toy.\n"
        "Then drop both fish into the abc-box."
    ),
    reset=reset,
    skill="grasp + drop into container",
    template="All balls into ball-sorter, then fish into abc-box.",
    target_duration_s=300.0,
)
