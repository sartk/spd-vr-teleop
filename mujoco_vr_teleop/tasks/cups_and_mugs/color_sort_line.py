"""cups_and_mugs/color_sort_line."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/color_sort_line",
    title="Color-sorted line",
    instruction=(
        "Line the 4 cups in a row at the front of the table, ordered red, yellow, green, blue.\n"
        "Place one mug at each end as bookends, handles facing outward.\n"
        "Nest no cups; all four stay rim-up in the row."
    ),
    reset=reset,
    skill="ordered placement",
    template="Row R-Y-G-B with mug bookends.",
    target_duration_s=300.0,
)
