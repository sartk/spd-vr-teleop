"""nut_and_bolt/nut_stacker."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/nut_stacker",
    title="Stacked nuts",
    instruction=(
        "Thread the square nut on the square peg.\n"
        "Stack the round nut on top of the square nut (still on the same peg).\n"
        "Insert the keyed peg in the slot.\n"
        "Hook-pull the slider all the way."
    ),
    reset=reset,
    skill="stacked insertion",
    template="Stack two nuts on one peg; then key + pull.",
    target_duration_s=360.0,
)
