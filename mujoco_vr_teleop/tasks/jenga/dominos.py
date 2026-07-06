"""jenga/dominos — stand blocks in a line, then topple the chain.

Reset: per-block coin flip into a left or right lateral pile (see
``coin_flip_pile_reset``).
"""

from __future__ import annotations

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.jenga.common import coin_flip_pile_reset


TASK = make_task(
    task_id="jenga/dominos",
    title="Dominoes",
    instruction=(
        "Stand at least 5 dominos on end in a line.\n"
        "Each block one block-width from the last.\n"
        "Tip the first one to topple the chain."
    ),
    reset=coin_flip_pile_reset,
    skill="precision placement, dynamic interaction",
    target_duration_s=180.0,
)
