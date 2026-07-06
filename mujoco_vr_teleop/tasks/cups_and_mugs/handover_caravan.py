"""cups_and_mugs/handover_caravan."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import apply_dr_start_mode, make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    apply_dr_start_mode(model, data, "cups_and_mugs", "ready")


TASK = make_task(
    task_id="cups_and_mugs/handover_caravan",
    title="Handover and hang",
    instruction=(
        "For each cup: pick with one hand, hand to the other, hang on a mug-tree peg.\n"
        "For each mug: pick with one hand, hand to the other, hang by the handle.\n"
        "Every hang must be made by the hand that received it."
    ),
    reset=reset,
    skill="bimanual handover + hang",
    template="Handover-then-hang every item on the tree.",
    target_duration_s=420.0,
)
