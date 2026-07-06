"""tea_time/lid_swap."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/lid_swap",
    title="Swap lids",
    instruction=(
        "Remove the teapot lid.\n"
        "Remove the sugar bowl lid.\n"
        "Put the teapot lid on the sugar bowl.\n"
        "Put the sugar lid on the teapot."
    ),
    reset=reset,
    skill="lid manipulation + swap",
    template="Swap teapot and sugar bowl lids.",
    target_duration_s=240.0,
)
