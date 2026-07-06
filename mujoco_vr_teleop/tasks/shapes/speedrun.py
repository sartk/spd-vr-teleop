"""shapes/speedrun."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/speedrun",
    title="All into toys",
    instruction=(
        "All 4 shape blocks in the shape sorter.\n"
        "All 6 cylinders and spheres in the ball-sorter.\n"
        "Both fish in the abc-box.\n"
        "Nothing left outside its toy."
    ),
    reset=reset,
    skill="time-pressured comprehensive sort",
    template="Everything into its toy, as fast as you can.",
    target_duration_s=300.0,
)
