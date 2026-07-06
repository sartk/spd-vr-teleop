"""jenga/playground — the deterministic state shown in playground mode.

Playground is just a task: the streamer runs its ``reset`` at boot (instead
of domain randomization) so playground always shows a defined scene — for
jenga, the clean 18-block assembled tower.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.jenga.common import (
    TOWER_CENTER_X,
    TOWER_CENTER_Y,
    build_tower_at,
)


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    build_tower_at(model, data, rng, (TOWER_CENTER_X, TOWER_CENTER_Y))


TASK = make_task(
    task_id="jenga/playground",
    title="Playground",
    instruction="Practice freely.",
    reset=reset,
    skill="free practice",
    target_duration_s=0.0,
)
