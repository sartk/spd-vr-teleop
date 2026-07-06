"""cleanup/full_routine."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/full_routine",
    title="Sweep and bin",
    instruction=(
        "Pick up the brush.\n"
        "Sweep every paper into the dustpan.\n"
        "Dump the dustpan into the bin.\n"
        "Place each bottle in the bin by hand.\n"
        "Rest the brush across the empty dustpan."
    ),
    reset=reset,
    skill="tool use + sort + place",
    template="Sweep papers, dump, then bin bottles by hand.",
    target_duration_s=420.0,
)
