"""cups_and_mugs/bimanual_hang."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/bimanual_hang",
    title="Bimanual hang",
    instruction=(
        "Pick up two cups, one in each hand.\n"
        "Invert and hang both on tree pegs at the same time.\n"
        "Repeat with the other two cups.\n"
        "Then hang both mugs simultaneously, one per hand."
    ),
    reset=reset,
    skill="bimanual parallel hang",
    template="Two-handed peg hangs in pairs.",
    target_duration_s=360.0,
)
