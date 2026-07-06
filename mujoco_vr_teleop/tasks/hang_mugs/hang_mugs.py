"""hang_mugs/hang_mugs — the single task for the hang_mugs scene.

Mugs start scattered on the table in front of the operator. Goal: hang
each mug on the mug tree.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_hang_mugs
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_hang_mugs(model, data, rng)


TASK = make_task(
    task_id="hang_mugs/hang_mugs",
    title="Hang the mugs",
    instruction="Hang each mug on the mug tree.",
    reset=reset,
    skill="bimanual hang",
    target_duration_s=180.0,
)
