"""cups_and_mugs/color_drill."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/color_drill",
    title="Bimanual tip-over",
    instruction=(
        "Pick up the red cup with the left hand and the blue cup with the right hand at the same time.\n"
        "Tip both over a mug simultaneously.\n"
        "Return both upright.\n"
        "Repeat with yellow and green."
    ),
    reset=reset,
    skill="bimanual symmetric pour",
    template="Symmetric two-hand pour over mugs by color pair.",
    target_duration_s=300.0,
)
