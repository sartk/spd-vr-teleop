"""cups_and_mugs/rainbow_tower."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/rainbow_tower",
    title="Nest cups",
    instruction=(
        "Nest the 4 cups rim-up, smallest inside largest.\n"
        "Hang both mugs on the mug tree by their handles.\n"
        "Place the nested cup stack in front of the tree."
    ),
    reset=reset,
    skill="nesting + hanging",
    template="Nest cups, hang mugs, place stack.",
    target_duration_s=300.0,
)
