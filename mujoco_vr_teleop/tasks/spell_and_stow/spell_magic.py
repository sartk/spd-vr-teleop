"""spell_and_stow/spell_magic."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.spell_and_stow.common import coin_flip_pile_reset


def reset(model: mujoco.MjModel, data: mujoco.MjData,
          rng: np.random.Generator) -> None:
    coin_flip_pile_reset(model, data, rng)


TASK = make_task(
    task_id="spell_and_stow/spell_magic",
    title="Spell MAGIC",
    instruction=(
        "Spell M-A-G-I-C face up on the table; "
        "stow the remaining blocks in the drawers."
    ),
    reset=reset,
    skill="opportunistic sort: spell or stow per pick",
    template="Pick any block; spell 'MAGIC' face up on the table or stow in any drawer.",
    target_duration_s=300.0,
)
