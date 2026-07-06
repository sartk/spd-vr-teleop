"""shapes/empty_all."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/empty_all",
    title="Empty and re-sort",
    instruction=(
        "Tip the ball-sorter, abc-box, and shape sorter so each piece falls out onto the table.\n"
        "Then put every piece back in its correct toy."
    ),
    reset=reset,
    skill="container reorient + re-sort",
    template="Empty every toy, then sort everything back.",
    target_duration_s=420.0,
)
