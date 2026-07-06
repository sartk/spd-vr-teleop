"""shapes/handover_assembly."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/handover_assembly",
    title="Handover insert",
    instruction=(
        "For each shape block: pick with one hand, hand to the other mid-air, insert with the second hand.\n"
        "Do the same for every ball into the ball-sorter.\n"
        "Do the same for both fish into the abc-box."
    ),
    reset=reset,
    skill="bimanual handover + insert",
    template="Hand every item across before inserting.",
    target_duration_s=420.0,
)
