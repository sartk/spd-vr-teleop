"""cup_stack/unstack — task #2 of the stack→unstack forced cycle.

Six cups start as two upright stacks of three (the snapshotted end state of
``stack_two_threes``, or a freshly built stack when started cold). The
operator's goal is to take the stacks apart and spread the cups out.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import stack_cups
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    stack_cups(model, data, rng, n_stacks=2, per_stack=3)


TASK = make_task(
    task_id="cup_stack/unstack",
    title="Flip and Unstack",
    instruction="Flip the stacks upside down and unstack the cups.",
    reset=reset,
    skill="bimanual unstacking",
    target_duration_s=180.0,
)
