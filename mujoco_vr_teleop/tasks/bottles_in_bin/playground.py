"""bottles_in_bin/playground — boot state for playground mode."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_bottles_in_bin
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_bottles_in_bin(model, data, rng)


TASK = make_task(
    task_id="bottles_in_bin/playground",
    title="Playground",
    instruction="Practice freely.",
    reset=reset,
    skill="free practice",
    target_duration_s=0.0,
)
