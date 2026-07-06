"""cleanup/brush_and_dustpan_duo."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cleanup", "scattered")


TASK = make_task(
    task_id="cleanup/brush_and_dustpan_duo",
    title="Mobile sweep",
    instruction=(
        "Hold the dustpan with one hand and the brush with the other.\n"
        "Walk the dustpan and brush together across the table, sweeping every paper directly into the held dustpan.\n"
        "Dump the dustpan into the bin.\n"
        "Place the bottles in the bin by hand."
    ),
    reset=reset,
    skill="bimanual tool use",
    template="One hand holds dustpan, other sweeps; mobile sweep.",
    target_duration_s=480.0,
)
