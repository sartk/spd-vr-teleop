"""nut_and_bolt/full_set."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/full_set",
    title="Full assembly",
    instruction=(
        "Thread both nuts on their pegs.\n"
        "Insert the keyed peg in the slot.\n"
        "Hook-pull the slider all the way back."
    ),
    reset=reset,
    skill="comprehensive sequence",
    template="Thread both nuts; key the slot; pull the slider.",
    target_duration_s=360.0,
)
