"""spell_and_stow/playground — the state shown in playground mode.

Playground keeps the scene's default layout — letter blocks in their grid,
drawers closed — but still draws the per-reset uniform scale so playground
matches what the tasks see. Because the blocks are placed at explicit
authored heights (not dropped), each block's spawn z is re-derived from the
scaled geometry so it rests flush on the table instead of floating/clipping.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_vr_teleop.tasks.common import make_task
from mujoco_vr_teleop.tasks.spell_and_stow.common import (
    LETTER_BLOCKS,
    SCALE_RANGE,
    _surface_top_z,
    apply_scale,
)


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Apply the per-reset uniform scale, then rebase the letter blocks onto
    the table so the scaled blocks rest flush rather than at their authored
    (unscaled) heights. The cabinet scales about its floor origin, so its
    freejoint spawn pose already keeps it table-flush — no rebase needed."""
    scale = float(rng.uniform(*SCALE_RANGE))
    apply_scale(model, scale)

    table_top = _surface_top_z(model, data)
    for name in LETTER_BLOCKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        jadr = int(model.body_jntadr[bid])
        if model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[jadr])
        # Authored clearance above the table scales with the block; xy and
        # orientation keep their authored grid values.
        authored_z = float(model.qpos0[qadr + 2])
        data.qpos[qadr] = model.qpos0[qadr]
        data.qpos[qadr + 1] = model.qpos0[qadr + 1]
        data.qpos[qadr + 2] = table_top + scale * (authored_z - table_top)
        data.qpos[qadr + 3:qadr + 7] = model.qpos0[qadr + 3:qadr + 7]
        data.qvel[int(model.jnt_dofadr[jadr]):int(model.jnt_dofadr[jadr]) + 6] = 0.0
    mujoco.mj_forward(model, data)


TASK = make_task(
    task_id="spell_and_stow/playground",
    title="Playground",
    instruction="Practice freely.",
    reset=reset,
    skill="free practice",
    target_duration_s=0.0,
)
