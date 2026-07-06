"""shapes/color_match."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/color_match",
    title="Red and square only",
    instruction=(
        "Red sphere and red cylinder into the ball-sorter.\n"
        "Both fish into the abc-box.\n"
        "The square block goes into its hole on the shape sorter.\n"
        "Leave the rest where they are."
    ),
    reset=reset,
    skill="selective sort by color",
    template="Selective subset insert per color.",
    target_duration_s=300.0,
)
