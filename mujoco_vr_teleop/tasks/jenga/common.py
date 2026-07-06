"""Jenga-scene constants and shared helpers.

Block geometry from ``examples/task_scenes/jenga.xml``:
    body-frame half-extents = (0.0127, 0.02032, 0.060325)
        body-x = 0.5 in thin (vertical when block lies flat)
        body-y = 0.8 in cross
        body-z = 2.375 in long
    Full block: 1 x 1.6 x 4.75 inches (matches real jenga).
    18 blocks, named ``jenga_1`` .. ``jenga_18``
    table top z ~ 0.74
    initial tower spans z = 0.7530 .. 0.8840 (6 layers, spacing 0.0262)
    tower footprint center (x, y) = (0.645, 0.0)

Operator convention: faces +x, so 'left' is -y and 'right' is +y on this rig.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_vr_teleop.tasks import scale_log
from mujoco_vr_teleop.tasks.base import RewardContext
from mujoco_vr_teleop.tasks.common import long_axis_world

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

BLOCKS: list[str] = [f"jenga_{i}" for i in range(1, 19)]

BLOCK_HALF_LONG = 0.060325        # 2.375 in: half of long axis (body-z)
BLOCK_HALF_CROSS = 0.02032        # 0.8 in: half of cross dim (body-y)
BLOCK_HALF_THIN = 0.0127          # 0.5 in: half of thin dim (body-x; vertical when flat)
# Back-compat aliases (some helpers reference the old names).
BLOCK_HALF_LEN = BLOCK_HALF_LONG
BLOCK_HALF_THICK = BLOCK_HALF_THIN
BLOCK_LAYER_HEIGHT = 2 * BLOCK_HALF_THIN + 0.0008  # 0.0262 m (1 in + 0.8 mm gap)

TABLE_TOP_Z = 0.74                # play-surface z (table_plane top)
BLOCK_REST_Z_FLAT = TABLE_TOP_Z + BLOCK_HALF_THIN    # ~0.7527, block lying flat
BLOCK_REST_Z_UPRIGHT = TABLE_TOP_Z + BLOCK_HALF_LONG # ~0.8003, block on end

TOWER_CENTER_X = 0.795
TOWER_CENTER_Y = 0.0
TOWER_BASE_Z = 0.7530             # z of layer-1 block centers in the canonical tower
TOWER_LAYER_STEP = BLOCK_LAYER_HEIGHT
CROSS_OFFSET_M = 2 * BLOCK_HALF_CROSS + 0.0006  # 0.04124 m (1.6 in + 0.6 mm gap)


# ---------------------------------------------------------------------------
# Pose predicates (no scratch state needed)
# ---------------------------------------------------------------------------

def is_flat(ctx: RewardContext, body: str, cos_thresh: float = 0.85) -> bool:
    """True if the block's long axis lies in the horizontal plane.

    The block's long axis is its local +z (collision box size 0.0125 0.0125
    0.0386 → z is longest). For a flat-lying block, this axis is horizontal,
    so its world-frame z component is near zero.
    """
    axis = long_axis_world(ctx, body)
    return abs(axis[2]) < (1.0 - cos_thresh)


def is_upright(ctx: RewardContext, body: str, cos_thresh: float = 0.85) -> bool:
    """True if the block's long axis is near-vertical (standing on end)."""
    axis = long_axis_world(ctx, body)
    return abs(axis[2]) > cos_thresh


def long_axis_horizontal_dir(ctx: RewardContext, body: str) -> np.ndarray:
    """Unit horizontal projection of the long axis. Undefined if not flat."""
    axis = long_axis_world(ctx, body)
    horiz = np.array([axis[0], axis[1], 0.0])
    n = float(np.linalg.norm(horiz))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return horiz / n


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

def resolve_block_ids(model: mujoco.MjModel) -> list[int]:
    """Body IDs for jenga_1 .. jenga_18, in spawn order."""
    ids = []
    for name in BLOCKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body {name!r} not in model")
        ids.append(bid)
    return ids


def surface_top_z(model: mujoco.MjModel, data: mujoco.MjData,
                  geom_name: str = "table_plane") -> float:
    """World-frame top-of-surface z for the given geom (defaults to table_plane)."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"surface geom {geom_name!r} not in model")
    return float(data.geom_xpos[gid, 2] + model.geom_size[gid, 2])


def hand_body_ids(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """Return (left_hand_body_ids, right_hand_body_ids).

    A body counts as a 'hand' if it is a descendant of the side's wrist link
    (``<side>-arm-link_6`` when using YAM grippers) OR its name starts with
    ``<side>_`` (sharpa hand bodies, attached without arm-prefix).
    """
    left: set[int] = set()
    right: set[int] = set()

    def add_descendants(root_bid: int, target: set[int]) -> None:
        # mj has body_parentid; walk all bodies and check ancestry.
        for bid in range(model.nbody):
            cur = bid
            while cur > 0:
                if cur == root_bid:
                    target.add(bid)
                    break
                cur = int(model.body_parentid[cur])

    for side, target in (("left", left), ("right", right)):
        wrist_name = f"{side}-arm-link_6"
        wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_name)
        if wrist_bid >= 0:
            add_descendants(wrist_bid, target)
        # Also catch sharpa-style bodies named ``<side>_*``.
        prefix = f"{side}_"
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(prefix):
                target.add(bid)
    return left, right


def block_contacts_hand(model: mujoco.MjModel, data: mujoco.MjData,
                        block_bid: int, hand_bids: set[int]) -> bool:
    """True if any active contact has one geom on `block_bid` and the other
    on a body in `hand_bids`."""
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        if (b1 == block_bid and b2 in hand_bids) or (b2 == block_bid and b1 in hand_bids):
            return True
    return False


def random_so3_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random SO(3) quaternion as (w, x, y, z)."""
    # scipy returns (x, y, z, w); convert.
    seed = int(rng.integers(0, 2**31 - 1))
    q_xyzw = Rotation.random(random_state=seed).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


# Shared lateral-pile constants (used by tower, criss_cross, dominos, etc.)
LEFT_PILE_CENTER = (0.795, -0.30)
RIGHT_PILE_CENTER = (0.795, +0.30)
PILE_HALF_WIDTH = 0.08
DROP_HEIGHT = 0.05
# Long enough for a block dropped from the ~0.25 m hand-clearance height to
# fall and mostly settle before the next one drops on top of it.
DROP_SECONDS_PER_BLOCK = 0.45
FINAL_SETTLE_SECONDS = 0.5

# Uniform scale randomization for the jenga blocks. One factor is drawn per
# reset and applied to all 18 blocks, so they stay identical to each other.
# Mass/inertia are intentionally held constant (blocks are authored at 60 g).
SCALE_RANGE = (0.95, 1.05)


class _ScaleBaseline:
    """Snapshot of the model arrays the block scale touches, captured at the
    authored (unscaled) state. Scaling is always applied relative to this
    baseline so repeated resets don't compound."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.vis_geom_ids: list[int] = []
        self.col_geom_ids: list[int] = []
        for name in BLOCKS:
            vis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_vis")
            col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_col")
            if vis < 0 or col < 0:
                continue
            self.vis_geom_ids.append(vis)
            self.col_geom_ids.append(col)
        # All 18 blocks share one visual mesh asset; snapshot it once.
        self.mesh_id = int(model.geom_dataid[self.vis_geom_ids[0]])
        adr = int(model.mesh_vertadr[self.mesh_id])
        num = int(model.mesh_vertnum[self.mesh_id])
        self.mesh_vert_range = (adr, num)
        self.mesh_vert = model.mesh_vert[adr:adr + num].copy()
        self.geom_rbound = model.geom_rbound.copy()
        self.geom_aabb = model.geom_aabb.copy()
        self.geom_size = model.geom_size.copy()


_BASELINE: _ScaleBaseline | None = None


def _baseline(model: mujoco.MjModel) -> _ScaleBaseline:
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = _ScaleBaseline(model)
    return _BASELINE


def scale_blocks(model: mujoco.MjModel, factor: float) -> None:
    """Uniformly scale all jenga blocks by ``factor``, relative to the model's
    authored geometry. Edits ``MjModel`` in place; call before placing/settling
    blocks. Mass and inertia are left untouched."""
    base = _baseline(model)
    s = float(factor)
    adr, num = base.mesh_vert_range
    model.mesh_vert[adr:adr + num] = base.mesh_vert * s
    for vis, col in zip(base.vis_geom_ids, base.col_geom_ids):
        model.geom_rbound[vis] = base.geom_rbound[vis] * s
        model.geom_aabb[vis] = base.geom_aabb[vis] * s
        model.geom_rbound[col] = base.geom_rbound[col] * s
        model.geom_aabb[col] = base.geom_aabb[col] * s
        model.geom_size[col] = base.geom_size[col] * s
    scale_log.record("jenga", s)


def sample_block_scale(rng: np.random.Generator) -> float:
    """Draw one uniform scale factor from ``SCALE_RANGE``."""
    return float(rng.uniform(*SCALE_RANGE))


def _scatter_into_two_piles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    *,
    left_ids: list[int],
    right_ids: list[int],
) -> None:
    """Drop each pile into its center, applying a random SO(3) orientation.

    A single uniform scale factor is drawn per reset and applied to all
    blocks before they drop; ``scatter_blocks`` computes drop heights from
    the (now-scaled) geometry, so the piles settle correctly at any scale."""
    from mujoco_vr_teleop.jenga_domain_randomization import scatter_blocks

    scale_blocks(model, sample_block_scale(rng))
    table_top = surface_top_z(model, data)
    for pile_ids, center in (
        (left_ids, LEFT_PILE_CENTER),
        (right_ids, RIGHT_PILE_CENTER),
    ):
        if not pile_ids:
            continue
        scatter_blocks(
            model, data, rng, pile_ids,
            initial_quats={bid: random_so3_quat(rng) for bid in pile_ids},
            surface_top_z=table_top,
            center=center,
            square_half_width=PILE_HALF_WIDTH,
            drop_height=DROP_HEIGHT,
            drop_seconds=DROP_SECONDS_PER_BLOCK,
        )
    for _ in range(round(FINAL_SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)


def coin_flip_pile_reset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
) -> None:
    """Per-block coin flip into left / right piles. Pile sizes vary per draw
    (one pile may have more than the other). Used by tower / criss_cross /
    dominos / hollow_tower."""
    body_ids = resolve_block_ids(model)
    left_ids: list[int] = []
    right_ids: list[int] = []
    for bid in body_ids:
        (left_ids if bool(rng.integers(0, 2)) else right_ids).append(bid)
    _scatter_into_two_piles(model, data, rng,
                             left_ids=left_ids, right_ids=right_ids)


def equal_pile_reset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
) -> None:
    """Half the blocks (9:9) into each pile. Block-to-pile assignment is
    randomized per reset; pile sizes are always exactly equal. Used by the
    handover tasks where the operator transfers blocks one-by-one between
    full piles."""
    body_ids = resolve_block_ids(model)
    shuffled = list(body_ids)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    left_ids = shuffled[:half]
    right_ids = shuffled[half:]
    _scatter_into_two_piles(model, data, rng,
                             left_ids=left_ids, right_ids=right_ids)


# Tower-layout poses copied from examples/task_scenes/jenga.xml (the spawn
# state IS the assembled tower). Layers alternate 90 deg about world z;
# blocks within a layer are spread by `CROSS_OFFSET_M` along the cross axis.
# Pos is relative to tower center (TOWER_CENTER_X, TOWER_CENTER_Y); add an
# offset to relocate.
_QUAT_A = (0.707107, 0.0, 0.707107, 0.0)   # long axis along world +x
_QUAT_B = (0.5, -0.5, 0.5, 0.5)            # long axis along world +y
_OFF = CROSS_OFFSET_M
_LAYER_ZS = tuple(TOWER_BASE_Z + i * TOWER_LAYER_STEP for i in range(6))
_TOWER_POSES: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
for _layer_idx in range(6):
    _z = _LAYER_ZS[_layer_idx]
    if _layer_idx % 2 == 0:
        # Layer A: long axis along world +x; blocks spread along world +y.
        for _dy in (-_OFF, 0.0, +_OFF):
            _TOWER_POSES.append(((0.0, _dy, _z), _QUAT_A))
    else:
        # Layer B: long axis along world +y; blocks spread along world +x.
        for _dx in (-_OFF, 0.0, +_OFF):
            _TOWER_POSES.append(((_dx, 0.0, _z), _QUAT_B))


def build_tower_at(model: mujoco.MjModel, data: mujoco.MjData,
                   rng: np.random.Generator,
                   center_xy: tuple[float, float],
                   settle_seconds: float = 0.5) -> None:
    """Place all 18 blocks in the canonical 6x3 tower layout, with the
    tower's base center at ``center_xy``. The blocks' freejoints are written
    directly; the scene is then settled for ``settle_seconds``.

    ``_TOWER_POSES`` z-values assume the table top is at ``TABLE_TOP_Z``. The
    scene builder shifts the table by ``table_height_offset``, so rebase every
    block onto the actual ``table_plane`` surface — otherwise the tower spawns
    embedded in the table and explodes on the first settle step.

    A single uniform scale factor is drawn per reset and applied to the
    blocks. Unlike the scatter resets, the tower's spawn poses are explicit
    constants, so the layer heights and within-layer offsets are scaled by
    the same factor about the table surface to keep the stack consistent.
    """
    body_ids = resolve_block_ids(model)
    z_shift = surface_top_z(model, data) - TABLE_TOP_Z

    scale = sample_block_scale(rng)
    scale_blocks(model, scale)

    def freejoint_qadr(bid: int) -> int:
        jadr = int(model.body_jntadr[bid])
        jnum = int(model.body_jntnum[bid])
        for jid in range(jadr, jadr + jnum):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(model.jnt_qposadr[jid])
        raise ValueError(f"body {bid} has no freejoint")

    def freejoint_dadr(bid: int) -> int:
        jadr = int(model.body_jntadr[bid])
        jnum = int(model.body_jntnum[bid])
        for jid in range(jadr, jadr + jnum):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(model.jnt_dofadr[jid])
        raise ValueError(f"body {bid} has no freejoint")

    cx, cy = center_xy
    for bid, ((dx, dy, z), quat) in zip(body_ids, _TOWER_POSES):
        qadr = freejoint_qadr(bid)
        dadr = freejoint_dadr(bid)
        # Scale the within-layer offsets and the height above the table by
        # the block scale, then rebase onto the actual table surface.
        sx = cx + scale * dx
        sy = cy + scale * dy
        sz = TABLE_TOP_Z + scale * (z - TABLE_TOP_Z) + z_shift
        data.qpos[qadr:qadr + 3] = [sx, sy, sz]
        data.qpos[qadr + 3:qadr + 7] = quat
        data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(round(settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def long_axis_cardinal(ctx: RewardContext, body: str, tol_deg: float = 10.0) -> str | None:
    """Classify the block's long axis as 'x' / 'y' / None.

    Returns 'x' if the long axis is horizontal AND within tol_deg of the
    world x-axis (in the horizontal plane), 'y' similarly, else None.
    The block must be flat (long axis horizontal); a vertical/tilted block
    returns None.
    """
    axis = long_axis_world(ctx, body)
    # Long axis must be near-horizontal: world-z component small.
    if abs(axis[2]) > np.sin(np.deg2rad(tol_deg)):
        return None
    horiz = np.array([axis[0], axis[1]])
    n = float(np.linalg.norm(horiz))
    if n < 1e-6:
        return None
    horiz = horiz / n
    cos_tol = float(np.cos(np.deg2rad(tol_deg)))
    if abs(horiz[0]) >= cos_tol:
        return "x"
    if abs(horiz[1]) >= cos_tol:
        return "y"
    return None
