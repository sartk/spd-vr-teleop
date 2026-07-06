"""dishrack/rack_dishes — task #1 of the rack→plate cycle.

Starting state is the standard scatter (plates and mugs spread out on
the operator-side half of the table, rack at the back). Operator's
goal is to put everything in the rack.
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
    task_id="dishrack/rack_dishes",
    title="Rack the dishes",
    instruction=(
        "Rack the plates.\n"
        "Place the mugs in the rack."
    ),
    reset=reset,
    skill="bimanual stack-and-stow",
    target_duration_s=300.0,
)
