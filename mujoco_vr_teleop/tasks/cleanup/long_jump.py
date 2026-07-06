"""cleanup/long_jump."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/long_jump",
    title="Toss into far bin",
    instruction=(
        "Move the bin to the far edge of the table.\n"
        "Underhand-toss every paper into the bin from the front edge.\n"
        "Underhand-toss every bottle into the bin from the front edge.\n"
        "Hook the bin with the brush and drag it back to the center.\n"
        "Dump the bin out onto the dustpan when it's back."
    ),
    reset=reset,
    skill="dynamic toss + tool retrieval",
    template="Toss everything in from afar, then drag the bin back.",
    target_duration_s=480.0,
)
