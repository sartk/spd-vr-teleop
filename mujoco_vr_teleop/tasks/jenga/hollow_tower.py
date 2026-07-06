"""jenga/hollow_tower — build a tall tower, 2 blocks per row with a hollow
centre.

Reset: per-block coin flip into a left or right lateral pile (see
``coin_flip_pile_reset``).
"""

from __future__ import annotations

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.jenga.common import coin_flip_pile_reset


TASK = make_task(
    task_id="jenga/hollow_tower",
    title="Hollow tower",
    instruction="Make a 9 row tower with 2 blocks per row, and a gap in the middle",
    reset=coin_flip_pile_reset,
    difficulty="easy",
    skill="precision stacking",
    target_duration_s=300.0,
)
