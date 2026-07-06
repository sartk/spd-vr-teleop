"""nut_and_bolt/thread_both."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/thread_both",
    title="Thread and key",
    instruction=(
        "Drop the square nut over the square peg.\n"
        "Drop the round nut over the round peg.\n"
        "Insert the keyed peg into the slot."
    ),
    reset=reset,
    skill="shape-matching insertion",
    template="Both nuts on their pegs; keyed peg in slot.",
    target_duration_s=300.0,
)
