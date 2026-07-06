"""cleanup/precision_sweep."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/precision_sweep",
    title="Sweep around bottles",
    instruction=(
        "Stand all 3 bottles upright in a line down the middle of the table.\n"
        "Sweep every paper past the bottles into the dustpan with the brush.\n"
        "Do not knock a bottle over.\n"
        "Dump the dustpan into the bin.\n"
        "Place the bottles in the bin one by one."
    ),
    reset=reset,
    skill="precision tool use",
    template="Sweep around upright bottles without knocking them.",
    target_duration_s=480.0,
)
