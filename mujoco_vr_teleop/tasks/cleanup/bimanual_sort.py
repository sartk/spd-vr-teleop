"""cleanup/bimanual_sort."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/bimanual_sort",
    title="Bimanual cleanup",
    instruction=(
        "Left hand: pick papers one at a time and drop them in the dustpan.\n"
        "Right hand: pick bottles one at a time and stand them upright in a row at the back.\n"
        "Work both hands in parallel.\n"
        "When done, dump the dustpan into the bin and then drop the bottles in too."
    ),
    reset=reset,
    skill="bimanual parallel sort",
    template="Hands work in parallel: papers vs. bottles.",
    target_duration_s=480.0,
)
