"""cleanup/recycle_sort."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/recycle_sort",
    title="Bin papers, sort bottles",
    instruction=(
        "Sweep every paper into the dustpan with the brush.\n"
        "Dump the dustpan into the bin.\n"
        "Lay the 3 bottles in a row on the dustpan instead.\n"
        "Move the loaded dustpan to the back of the table.\n"
        "Line the bin beside it."
    ),
    reset=reset,
    skill="sort + transport",
    template="Bin papers, then move bottles on the dustpan.",
    target_duration_s=420.0,
)
