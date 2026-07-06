"""cup_stack/stack_two_threes — task #1 of the stack→unstack forced cycle.

Six cups start scattered on the table. The operator's goal is to nest them
into two upright stacks of three. The end state is snapshotted by
CupsTaskManager and handed to ``unstack`` as its start state.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_cups
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_cups(model, data, rng)


TASK = make_task(
    task_id="cup_stack/stack_two_threes",
    title="Stack the cups",
    instruction="Nest the six cups into two stacks of three.",
    reset=reset,
    skill="bimanual stacking",
    target_duration_s=180.0,
)
