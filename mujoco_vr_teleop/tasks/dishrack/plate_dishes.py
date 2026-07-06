"""dishrack/plate_dishes — task #2 of the rack→plate cycle.

Starts from the post-rack state (the snapshot captured when #1 was
marked complete). Operator unloads the rack into two place settings.
DishrackTaskManager owns the snapshot/restore — this task's own reset()
falls back to scatter_dishrack when no snapshot exists (e.g. someone
launches plate_dishes directly without doing rack_dishes first).
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_dishrack
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_dishrack(model, data, rng)


TASK = make_task(
    task_id="dishrack/plate_dishes",
    title="Set the table for two",
    instruction=(
        "Set two place settings, one on each side of the table.\n"
        "Each setting: plate centered, mug in the top corner of the plate."
    ),
    reset=reset,
    skill="bimanual unload + arrange",
    target_duration_s=360.0,
)
