"""jenga/handover_right_to_left — transfer 9 blocks from the right pile to
the left pile, one at a time, with a mid-air bimanual handover for each.

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
    task_id="jenga/handover_right_to_left",
    title="Handover R->L",
    instruction=(
        f"Move all {_N} blocks from the right pile to the left pile, one at a time.\n"
        "Pick each block with the right hand, hand it to the left hand mid-air, then place."
    ),
    reset=reset,
    skill="bimanual mid-air handover",
    template="9 blocks transferred right-to-left; every one handed across mid-air.",
    target_duration_s=300.0,
)
