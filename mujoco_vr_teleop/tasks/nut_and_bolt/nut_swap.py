"""nut_and_bolt/nut_swap."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/nut_swap",
    title="Nut detour swap",
    instruction=(
        "Thread the square nut on the round peg first.\n"
        "Lift it off, then thread the round nut on the round peg.\n"
        "Then thread the square nut on the square peg.\n"
        "Insert the keyed peg in the slot."
    ),
    reset=reset,
    skill="ordered swap",
    template="Detour-swap nuts onto correct pegs.",
    target_duration_s=360.0,
)
