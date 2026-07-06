"""cups_and_mugs/bottoms_up."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/bottoms_up",
    title="Inverted cup tower",
    instruction=(
        "Stack all 4 cups rim-down into a tower on the table.\n"
        "Balance one mug upright on top of the tower.\n"
        "Hang the other mug on the tree."
    ),
    reset=reset,
    skill="inverted stacking + balance",
    template="Cup tower upside-down with mug on top.",
    target_duration_s=360.0,
)
