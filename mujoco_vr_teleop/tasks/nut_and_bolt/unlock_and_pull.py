"""nut_and_bolt/unlock_and_pull."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/unlock_and_pull",
    title="Key and pull",
    instruction=(
        "Insert the keyed peg into the slot.\n"
        "Pick up the hook tool.\n"
        "Hook the slider ring and pull it all the way back.\n"
        "Return the hook tool to the table."
    ),
    reset=reset,
    skill="precise insertion + tool use",
    template="Key the slot, then hook-pull the slider.",
    target_duration_s=300.0,
)
