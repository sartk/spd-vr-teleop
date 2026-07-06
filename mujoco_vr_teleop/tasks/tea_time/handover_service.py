"""tea_time/handover_service."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "tea_time", "place_setting")


TASK = make_task(
    task_id="tea_time/handover_service",
    title="Handover service",
    instruction=(
        "For each cup: pick with one hand, hand to the other, place on its saucer.\n"
        "For each spoon: pick with one hand, hand to the other, lay on the saucer with the cup.\n"
        "Lift the teapot with both hands together and place it center back."
    ),
    reset=reset,
    skill="bimanual handover + bimanual lift",
    template="Hand off cups+spoons; bimanual teapot lift.",
    target_duration_s=360.0,
)
