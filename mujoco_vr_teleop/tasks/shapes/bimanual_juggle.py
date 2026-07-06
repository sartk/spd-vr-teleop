"""shapes/bimanual_juggle."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/bimanual_juggle",
    title="Bimanual sort",
    instruction=(
        "Right hand: feed all 4 shape blocks into the shape sorter, one at a time.\n"
        "Left hand: feed all 6 balls (cylinders and spheres) into the ball-sorter.\n"
        "Work both hands in parallel.\n"
        "Finish by dropping the fish into the abc-box (one hand each)."
    ),
    reset=reset,
    skill="bimanual parallel insertion",
    template="Two hands feed two toys in parallel; finish with fish.",
    target_duration_s=360.0,
)
