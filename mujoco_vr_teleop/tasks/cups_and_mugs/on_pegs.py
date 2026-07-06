"""cups_and_mugs/on_pegs."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/on_pegs",
    title="Hang on pegs",
    instruction=(
        "Invert each cup (rim-down).\n"
        "Hang one cup on each mug-tree peg.\n"
        "Hang both mugs on the remaining pegs by their handles."
    ),
    reset=reset,
    skill="reorientation + insertion",
    template="Invert cups; hang everything on the tree.",
    target_duration_s=360.0,
)
