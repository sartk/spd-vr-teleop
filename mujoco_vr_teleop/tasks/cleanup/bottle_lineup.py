"""cleanup/bottle_lineup."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/bottle_lineup",
    title="Lay and restand bottles",
    instruction=(
        "Lay all 3 bottles on their sides, end-to-end, forming one long line.\n"
        "Sweep every paper into the dustpan with the brush.\n"
        "Dump the dustpan into the bin.\n"
        "Stand the bottles back upright in their original line."
    ),
    reset=reset,
    skill="multi-pose object manipulation",
    template="Lay bottles flat, sweep, then stand them upright.",
    target_duration_s=420.0,
)
