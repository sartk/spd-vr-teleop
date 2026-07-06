"""nut_and_bolt/key_rotation."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/key_rotation",
    title="In-hand key rotation",
    instruction=(
        "Pick up the keyed peg.\n"
        "Rotate it in-hand so the tab faces the slot notch.\n"
        "Lower it into the slot until the tab seats in the notch.\n"
        "Then thread both nuts on their pegs."
    ),
    reset=reset,
    skill="in-hand rotation + keyed insertion",
    template="Reorient the keyed peg, seat it, then thread nuts.",
    target_duration_s=360.0,
)
