"""tea_time/tower_of_porcelain."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/tower_of_porcelain",
    title="Porcelain tower",
    instruction=(
        "Stack in this order from the table up:\n"
        "saucer, cup rim-down, saucer, cup rim-down, spoon on top.\n"
        "The second spoon rests across the sugar bowl."
    ),
    reset=reset,
    skill="precision multi-piece stack",
    template="Build a 4-tier porcelain tower; spoon on top.",
    target_duration_s=360.0,
)
