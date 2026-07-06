"""shapes/sort_blocks."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/sort_blocks",
    title="Sort blocks, fish",
    instruction=(
        "Drop each block into its matching hole on the shape sorter.\n"
        "Hex, square, triangle, circle.\n"
        "Then drop both fish into the abc-box slots."
    ),
    reset=reset,
    skill="shape matching + insertion",
    template="Match every shape; bin both fish.",
    target_duration_s=360.0,
)
