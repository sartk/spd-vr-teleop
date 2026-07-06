"""jenga/tower — assemble the full 6-layer, 18-block tower from scattered blocks.

Reset: per-block coin flip into a left or right lateral pile (see
``coin_flip_pile_reset``).
"""

from __future__ import annotations

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.jenga.common import coin_flip_pile_reset


TASK = make_task(
    task_id="jenga/tower",
    title="Full tower",
    instruction=(
        "Build the full 18-block tower.\n"
        "6 layers of 3.\n"
        "Alternate direction each layer."
    ),
    reset=coin_flip_pile_reset,
    skill="long-horizon tower building",
    target_duration_s=240.0,
)
