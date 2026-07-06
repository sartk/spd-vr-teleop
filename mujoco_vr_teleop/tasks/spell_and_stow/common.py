"""spell_and_stow scene constants."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_vr_teleop.tasks import scale_log

LETTER_BLOCKS = tuple(f"block_{c}" for c in "ABCDEFGHIJKLMNO")
VOWELS = ("block_A", "block_E", "block_I", "block_O")
CONSONANTS = tuple(b for b in LETTER_BLOCKS if b not in VOWELS)
DRAWERS = ("drawer-slide-top", "drawer-slide-middle", "drawer-slide-bottom")


# Lateral piles (operator-relative left and right of the table), mirroring
# the jenga setup. Same x as the central drawer in front of the operator.
LEFT_PILE_CENTER = (0.7, -0.30)
RIGHT_PILE_CENTER = (0.7, +0.30)
PILE_HALF_WIDTH = 0.10
DROP_HEIGHT = 0.05
# Long enough for a block dropped from the ~0.25 m hand-clearance height
# (scatter_blocks' min_drop_clearance) to fall and settle before the next
# one drops — short drop times left light blocks mid-fall, squishing into
# the table. Matches the jenga scatter timing.
DROP_SECONDS_PER_BLOCK = 0.45
FINAL_SETTLE_SECONDS = 0.5


def _random_so3_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random SO(3) quaternion as (w, x, y, z)."""
    seed = int(rng.integers(0, 2**31 - 1))
    q_xyzw = Rotation.random(random_state=seed).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _resolve_block_ids(model: mujoco.MjModel) -> list[int]:
    ids = []
    for name in LETTER_BLOCKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(bid)
    return ids


def _surface_top_z(model: mujoco.MjModel, data: mujoco.MjData,
                   geom_name: str = "table_plane") -> float:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"surface geom {geom_name!r} not in model")
    return float(data.geom_xpos[gid, 2] + model.geom_size[gid, 2])


# Cabinet placement jitter, applied to the drawer_body freejoint each reset.
CABINET_POS_JITTER = 0.05    # +/- metres in x and y
CABINET_YAW_JITTER = np.deg2rad(10.0)  # +/- radians about z

# Uniform scale randomization for the letter blocks + cabinet. One factor is
# drawn per reset and applied to *both* sets of bodies, so their relative
# proportions stay fixed. Mass/inertia are intentionally held constant.
SCALE_RANGE = (0.95, 1.05)


def _subtree_body_ids(model: mujoco.MjModel, root_name: str) -> list[int]:
    """All body ids in the subtree rooted at ``root_name`` (root included)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        return []
    ids = [root]
    for bid in range(model.nbody):
        cur = bid
        while cur > 0:
            cur = int(model.body_parentid[cur])
            if cur == root:
                ids.append(bid)
                break
    return ids


class _ScaleBaseline:
    """Snapshot of every model array the scale touches, captured at the
    model's authored (unscaled) state. Scaling is always applied relative to
    this baseline so repeated resets don't compound."""

    def __init__(self, model: mujoco.MjModel) -> None:
        # Letter blocks: per-block visual mesh verts + bounds, collision box.
        self.mesh_vert_ranges: list[tuple[int, int]] = []
        self.mesh_vert: list[np.ndarray] = []
        self.vis_geom_ids: list[int] = []
        self.col_geom_ids: list[int] = []
        for name in LETTER_BLOCKS:
            vis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_vis")
            col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_col")
            if vis < 0 or col < 0:
                continue
            mesh = int(model.geom_dataid[vis])
            adr = int(model.mesh_vertadr[mesh])
            num = int(model.mesh_vertnum[mesh])
            self.mesh_vert_ranges.append((adr, num))
            self.mesh_vert.append(model.mesh_vert[adr:adr + num].copy())
            self.vis_geom_ids.append(vis)
            self.col_geom_ids.append(col)
        self.geom_rbound = model.geom_rbound.copy()
        self.geom_aabb = model.geom_aabb.copy()
        self.geom_size = model.geom_size.copy()
        self.geom_pos = model.geom_pos.copy()
        # Cabinet subtree: child body offsets + slide joint ranges. (Geom
        # size/pos are covered by the model-wide copies above.)
        self.cabinet_body_ids = _subtree_body_ids(model, "drawer_body")
        self.body_pos = model.body_pos.copy()
        self.jnt_range = model.jnt_range.copy()


_BASELINE: _ScaleBaseline | None = None


def _baseline(model: mujoco.MjModel) -> _ScaleBaseline:
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = _ScaleBaseline(model)
    return _BASELINE


def apply_scale(model: mujoco.MjModel, factor: float) -> None:
    """Uniformly scale the letter blocks and cabinet by ``factor``, relative
    to the model's authored geometry. Edits ``MjModel`` in place; call before
    placing/settling bodies. Mass and inertia are left untouched."""
    base = _baseline(model)
    s = float(factor)

    # --- Letter blocks --------------------------------------------------
    for (adr, num), verts, vis, col in zip(
        base.mesh_vert_ranges, base.mesh_vert,
        base.vis_geom_ids, base.col_geom_ids,
    ):
        model.mesh_vert[adr:adr + num] = verts * s
        model.geom_rbound[vis] = base.geom_rbound[vis] * s
        model.geom_aabb[vis] = base.geom_aabb[vis] * s
        model.geom_size[col] = base.geom_size[col] * s

    # --- Cabinet subtree ------------------------------------------------
    # Every geom under drawer_body: scale half-extents and local offset.
    for bid in base.cabinet_body_ids:
        for gid in np.flatnonzero(model.geom_bodyid == bid):
            model.geom_size[gid] = base.geom_size[gid] * s
            model.geom_pos[gid] = base.geom_pos[gid] * s
            model.geom_rbound[gid] = base.geom_rbound[gid] * s
            model.geom_aabb[gid] = base.geom_aabb[gid] * s
        # Child-body local offsets (drawer-slide-* z stack) scale too; the
        # drawer_body root keeps its freejoint-driven pose.
        if bid != base.cabinet_body_ids[0]:
            model.body_pos[bid] = base.body_pos[bid] * s
        # Slide joints: travel range scales with the cabinet.
        jadr = int(model.body_jntadr[bid])
        for jid in range(jadr, jadr + int(model.body_jntnum[bid])):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                model.jnt_range[jid] = base.jnt_range[jid] * s

    scale_log.record("spell_and_stow", s)


def randomize_cabinet(model: mujoco.MjModel, data: mujoco.MjData,
                      rng: np.random.Generator) -> None:
    """Jitter the drawer_body freejoint by +/-5 cm in x/y and +/-10 deg yaw
    around the pose baked into the scene XML."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drawer_body")
    if bid < 0:
        return
    jid = int(model.body_jntadr[bid])
    if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        return  # cabinet isn't free-jointed in this scene
    qadr = int(model.jnt_qposadr[jid])
    # Base pose = the freejoint's spawn pose from the XML (model.qpos0).
    base_pos = model.qpos0[qadr:qadr + 3].copy()
    base_quat = model.qpos0[qadr + 3:qadr + 7].copy()
    dx, dy = rng.uniform(-CABINET_POS_JITTER, CABINET_POS_JITTER, size=2)
    dyaw = rng.uniform(-CABINET_YAW_JITTER, CABINET_YAW_JITTER)
    yaw_q = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
    new_quat = np.empty(4)
    mujoco.mju_mulQuat(new_quat, yaw_q, base_quat)
    data.qpos[qadr:qadr + 3] = base_pos + [dx, dy, 0.0]
    data.qpos[qadr + 3:qadr + 7] = new_quat
    data.qvel[int(model.jnt_dofadr[jid]):int(model.jnt_dofadr[jid]) + 6] = 0.0


def coin_flip_pile_reset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
) -> None:
    """Per-block coin flip into left / right piles, mirroring the jenga
    reset. Letter blocks have a box collision geom so jenga's scatter_blocks
    helper works directly on them.

    A single uniform scale factor is drawn per reset and applied to both the
    letter blocks and the cabinet before placement; ``scatter_blocks``
    computes drop heights from the (now-scaled) geometry, and the cabinet is
    scaled about its floor origin so it stays flush on the table."""
    from mujoco_vr_teleop.jenga_domain_randomization import scatter_blocks

    apply_scale(model, float(rng.uniform(*SCALE_RANGE)))
    randomize_cabinet(model, data, rng)

    body_ids = _resolve_block_ids(model)
    left_ids: list[int] = []
    right_ids: list[int] = []
    for bid in body_ids:
        (left_ids if bool(rng.integers(0, 2)) else right_ids).append(bid)

    table_top = _surface_top_z(model, data)
    for pile_ids, center in (
        (left_ids, LEFT_PILE_CENTER),
        (right_ids, RIGHT_PILE_CENTER),
    ):
        if not pile_ids:
            continue
        scatter_blocks(
            model, data, rng, pile_ids,
            initial_quats={bid: _random_so3_quat(rng) for bid in pile_ids},
            surface_top_z=table_top,
            center=center,
            square_half_width=PILE_HALF_WIDTH,
            drop_height=DROP_HEIGHT,
            drop_seconds=DROP_SECONDS_PER_BLOCK,
        )
    for _ in range(round(FINAL_SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
