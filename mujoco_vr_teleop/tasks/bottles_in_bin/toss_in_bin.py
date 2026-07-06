"""bottles_in_bin/toss_in_bin — the single task for the bottles_in_bin scene.

3-6 bottles start scattered on the table in front of the operator with the
bin against the back wall. Goal: get every bottle into the bin.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_bottles_in_bin
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_bottles_in_bin(model, data, rng)


TASK = make_task(
    task_id="bottles_in_bin/toss_in_bin",
    title="Toss the bottles in the bin",
    instruction="Toss every bottle into the bin.",
    reset=reset,
    skill="bimanual toss",
    target_duration_s=180.0,
)
