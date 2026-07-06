"""tea_time/bimanual_stir."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/bimanual_stir",
    title="Bimanual stir",
    instruction=(
        "Pick up one spoon in each hand.\n"
        "Stir 3 circles inside both cups at the same time, one hand per cup.\n"
        "Return each spoon to its cup's saucer."
    ),
    reset=reset,
    skill="bimanual periodic motion",
    template="Stir both cups in parallel; return spoons.",
    target_duration_s=300.0,
)
