"""nut_and_bolt/hook_relay."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "nut_and_bolt", "ready")


TASK = make_task(
    task_id="nut_and_bolt/hook_relay",
    title="Hold and insert",
    instruction=(
        "Hold the keyed peg in one hand, lined up over the slot.\n"
        "With the other hand, hook the slider ring and pull it back.\n"
        "While holding the slider back, insert the keyed peg with the first hand.\n"
        "Release both."
    ),
    reset=reset,
    skill="bimanual constraint coordination",
    template="One hand holds slider open; the other inserts keyed peg.",
    target_duration_s=300.0,
)
