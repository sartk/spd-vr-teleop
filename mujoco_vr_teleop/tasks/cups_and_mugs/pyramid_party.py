"""cups_and_mugs/pyramid_party."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/pyramid_party",
    title="Cup pyramid",
    instruction=(
        "Build a 3-cup base in a row, rim-up.\n"
        "Stand the 4th cup on top in the center.\n"
        "Place a mug at each end of the base as bookends."
    ),
    reset=reset,
    skill="precision stacking",
    template="3-cup base + 1 cup top + mug bookends.",
    target_duration_s=300.0,
)
