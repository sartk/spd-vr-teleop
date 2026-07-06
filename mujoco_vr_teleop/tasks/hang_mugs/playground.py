"""hang_mugs/playground — boot state for playground mode."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_hang_mugs
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_hang_mugs(model, data, rng)


TASK = make_task(
    task_id="hang_mugs/playground",
    title="Playground",
    instruction="Practice freely.",
    reset=reset,
    skill="free practice",
    target_duration_s=0.0,
)
