"""jenga/handover_left_to_right — transfer 9 blocks from the left pile to
the right pile, one at a time, with a mid-air bimanual handover for each.

Reset: equal_pile_reset (9:9 split between the two lateral piles).
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.jenga.common import equal_pile_reset


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    equal_pile_reset(model, data, rng)


_N = 9


TASK = make_task(
    task_id="jenga/handover_left_to_right",
    title="Handover L->R",
    instruction=(
        f"Move all {_N} blocks from the left pile to the right pile, one at a time.\n"
        "Pick each block with the left hand, hand it to the right hand mid-air, then place."
    ),
    reset=reset,
    skill="bimanual mid-air handover",
    template="9 blocks transferred left-to-right; every one handed across mid-air.",
    target_duration_s=300.0,
)
