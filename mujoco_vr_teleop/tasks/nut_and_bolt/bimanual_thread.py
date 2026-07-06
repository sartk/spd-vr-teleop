"""nut_and_bolt/bimanual_thread."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/bimanual_thread",
    title="Bimanual thread",
    instruction=(
        "Pick up the square nut with the left hand and the round nut with the right hand at the same time.\n"
        "Lower both onto their matching pegs simultaneously.\n"
        "Then pick up the keyed peg in one hand and the hook tool in the other; insert and pull simultaneously."
    ),
    reset=reset,
    skill="bimanual symmetric assembly",
    template="Two-hand simultaneous thread + insert-and-pull.",
    target_duration_s=360.0,
)
