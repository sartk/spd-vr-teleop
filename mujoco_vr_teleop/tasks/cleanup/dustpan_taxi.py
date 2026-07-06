"""cleanup/dustpan_taxi."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/dustpan_taxi",
    title="Dustpan taxi",
    instruction=(
        "Load every paper into the dustpan by hand (no brush).\n"
        "Carry the loaded dustpan across the table.\n"
        "Tip it into the bin.\n"
        "Return the dustpan to its starting spot.\n"
        "Then place the bottles in the bin one by one."
    ),
    reset=reset,
    skill="container transport",
    template="Hand-load papers, carry, dump, return.",
    target_duration_s=480.0,
)
