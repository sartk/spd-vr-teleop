"""shapes/fish_wedding."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "shapes", "ready")


TASK = make_task(
    task_id="shapes/fish_wedding",
    title="Two-fish insert",
    instruction=(
        "Pick one fish in each hand.\n"
        "Lay them nose-to-nose in front of the abc-box.\n"
        "Push them as a pair simultaneously into the abc-box slots, one fish per slot, one hand each."
    ),
    reset=reset,
    skill="bimanual alignment + simultaneous insert",
    template="Two-fish nose-to-nose, simultaneous insert.",
    target_duration_s=270.0,
)
