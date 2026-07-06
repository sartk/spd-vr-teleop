"""dishrack/playground — the boot state shown in playground mode.

Same as every other dishrack task: scatter-drop a fresh rack + plates + mugs
so no two items penetrate. The streamer / vr_server runs this on initial
boot and on plain (no task selected) reset.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.scene_resets import scatter_dishrack
from mujoco_vr_teleop.tasks.common import make_task


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    scatter_dishrack(model, data, rng)


TASK = make_task(
    task_id="dishrack/playground",
    title="Playground",
    instruction="Practice freely.",
    reset=reset,
    skill="free practice",
    target_duration_s=0.0,
)
