"""Dishrack reset.

Lay every item at a fixed slot; AABB-aware placement so each item's
base sits on the table.
"""
from __future__ import annotations

import math
import re

import mujoco
import numpy as np

from mujoco_vr_teleop.scatter_reset import _freejoint_dadr, _freejoint_qadr
from mujoco_vr_teleop.variant_pools import DISHRACK_POOLS, SCENE_POOLS

# A bottle slot body is ``bottle_slot_<letter>``; the attached variant
# subtree under it carries a ``bottle_slot_<letter>__...`` prefix, which this
# pattern deliberately excludes (single trailing letter, no "__").
_BOTTLE_SLOT_RE = re.compile(r"bottle_slot_[a-z]")
# Same convention for the cup scenes' cup slots: ``cup_slot_<letter>``.
_CUP_SLOT_RE = re.compile(r"cup_slot_[a-z]")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

_BASELINE: dict[int, dict] = {}


def _subtree(model, root_bid):
    """Set of body ids in the subtree rooted at root_bid."""
    out = {root_bid}
    while True:
        n = len(out)
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in out:
                out.add(b)
        if len(out) == n:
            return out


def _baseline(model):
    """Mesh/geom baseline arrays for the model, cached by id(model)."""
    key = id(model)
    if key not in _BASELINE:
        _BASELINE[key] = {
            "mesh_vert": np.asarray(model.mesh_vert).copy(),
            "geom_size": np.asarray(model.geom_size).copy(),
            "geom_aabb": np.asarray(model.geom_aabb).copy(),
            "geom_rbound": np.asarray(model.geom_rbound).copy(),
        }
    return _BASELINE[key]


def _scale(model, bid, factor):
    """Rescale the body subtree's meshes + geom sizes by ``factor`` (from
    baseline, so repeated calls don't compound)."""
    snap = _baseline(model)
    bids = _subtree(model, bid)
    seen_meshes: set[int] = set()
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in bids:
            continue
        model.geom_size[gid] = snap["geom_size"][gid] * factor
        model.geom_aabb[gid] = snap["geom_aabb"][gid] * factor
        model.geom_rbound[gid] = snap["geom_rbound"][gid] * factor
        if int(model.geom_type[gid]) != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(model.geom_dataid[gid])
        if mid < 0 or mid in seen_meshes:
            continue
        v0 = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        model.mesh_vert[v0 : v0 + n] = snap["mesh_vert"][v0 : v0 + n] * factor
        seen_meshes.add(mid)


_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=float,
)


def _aabb(model, data, bid):
    """World AABB of the body subtree's collision geoms."""
    mujoco.mj_forward(model, data)
    bids = _subtree(model, bid)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in bids:
            continue
        if int(model.geom_contype[gid]) == 0:
            continue
        c = model.geom_aabb[gid, 0:3]
        h = model.geom_aabb[gid, 3:6]
        R = data.geom_xmat[gid].reshape(3, 3)
        p = data.geom_xpos[gid]
        corners = (R @ (c + _CORNER_SIGNS * h).T).T + p
        lo = np.minimum(lo, corners.min(axis=0))
        hi = np.maximum(hi, corners.max(axis=0))
    return lo, hi


def _place(model, data, bid, x, y, base_z, yaw=0.0, quat_wxyz=None):
    """Set freejoint pose so the body's AABB bottom is at ``base_z``."""
    qadr = _freejoint_qadr(model, bid)
    dadr = _freejoint_dadr(model, bid)
    if quat_wxyz is None:
        h = 0.5 * yaw
        quat_wxyz = (math.cos(h), 0.0, 0.0, math.sin(h))
    data.qpos[qadr : qadr + 3] = [x, y, 1.0]
    data.qpos[qadr + 3 : qadr + 7] = list(quat_wxyz)
    data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    lo, hi = _aabb(model, data, bid)
    data.qpos[qadr + 2] = 1.0 + (base_z - lo[2])
    mujoco.mj_forward(model, data)
    return _aabb(model, data, bid)


def _table(model, data):
    """(cx, cy, hx, hy, top_z) of the table_visual geom."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    mujoco.mj_forward(model, data)
    cx, cy, cz = data.geom_xpos[gid]
    sx, sy, sz = model.geom_size[gid]
    return float(cx), float(cy), float(sx), float(sy), float(cz + sz)


def _slot(model, pool_name):
    """body id of the slot for ``pool_name``."""
    pool = None
    for pools in SCENE_POOLS.values():
        for p in pools:
            if p.name == pool_name:
                pool = p
                break
        if pool is not None:
            break
    if pool is None:
        raise RuntimeError(f"unknown pool {pool_name!r}")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pool.slot_body)
    if bid < 0:
        raise RuntimeError(f"missing slot body {pool.slot_body!r}")
    return bid


def _body(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"missing body {name!r}")
    return bid


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def scatter_dishrack(model, data, rng):
    cx, cy, hx, hy, top_z = _table(model, data)

    # Rack: scale, decide orientation, place at the back wall.
    rack = _slot(model, "dishrack_rack")
    _scale(model, rack, float(rng.uniform(1.0, 1.2)))
    _place(model, data, rack, 0.0, 0.0, top_z)
    lo, hi = _aabb(model, data, rack)
    yaw = math.pi / 2 if (hi[0] - lo[0]) > (hi[1] - lo[1]) else 0.0
    yaw += float(rng.uniform(-math.pi / 12, math.pi / 12))
    # Re-place at yaw=0 origin to read the rotated AABB, then shift x so
    # max-x is wall_gap from the table edge.
    _place(model, data, rack, 0.0, 0.0, top_z, yaw=yaw)
    _, hi = _aabb(model, data, rack)
    rack_x = (cx + hx) - float(rng.uniform(0.02, 0.10)) - hi[0]
    rack_y = float(rng.uniform(-0.05, 0.05))
    rack_lo, rack_hi = _place(model, data, rack, rack_x, rack_y,
                                top_z + 0.01, yaw=yaw)

    # Scatter region: middle of the table, away from the arms and using the
    # full y-extent. Shifted +0.15 m back with the lengthened (3 ft) table.
    sx_lo, sx_hi = 0.65, 0.93
    sy_lo, sy_hi = -0.58, 0.58

    # Plates: shared diameter, scattered flat in the front area.
    target_diam = float(rng.uniform(0.2032, 0.254))
    plate_ids = [_slot(model, f"dishrack_plate_{i}") for i in range(4)]
    for bid in plate_ids:
        _scale(model, bid, 1.0)
        _place(model, data, bid, 0.0, 0.0, top_z)
        lo, hi = _aabb(model, data, bid)
        _scale(model, bid, target_diam / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))

    # Measure mugs (need their footprint radii for rejection).
    mug_ids = [_slot(model, f"dishrack_mug_{i}") for i in range(2)]
    item_info: list[tuple[int, float, float]] = []  # (bid, radius, drop_h)
    for bid in plate_ids:
        lo, hi = _aabb(model, data, bid)
        r = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
        item_info.append((bid, r, 0.05))
    for bid in mug_ids:
        _scale(model, bid, float(rng.uniform(0.9, 1.1)))
        _place(model, data, bid, 0.0, 0.0, top_z)
        lo, hi = _aabb(model, data, bid)
        r = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
        item_info.append((bid, r, 0.08))

    def floor_at(x, y, r, bid):
        z = top_z
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) == bid:
                continue
            if int(model.geom_contype[gid]) == 0:
                continue
            c = model.geom_aabb[gid, 0:3]
            h = model.geom_aabb[gid, 3:6]
            R = data.geom_xmat[gid].reshape(3, 3)
            p = data.geom_xpos[gid]
            corners = (R @ (c + _CORNER_SIGNS * h).T).T + p
            if ((corners[:, 0].min() - r) <= x <= (corners[:, 0].max() + r)
                    and (corners[:, 1].min() - r) <= y <= (corners[:, 1].max() + r)):
                z = max(z, float(corners[:, 2].max()))
        return z

    for bid, r, _ in item_info:
        mujoco.mj_forward(model, data)
        # Sample several candidates; pick the one with the lowest floor
        # (i.e. most open space) so items spread out instead of stacking.
        best = None
        for _ in range(16):
            x = float(rng.uniform(sx_lo + r, sx_hi - r))
            y = float(rng.uniform(sy_lo + r, sy_hi - r))
            z = floor_at(x, y, r, bid)
            if best is None or z < best[2]:
                best = (x, y, z)
        x, y, z = best
        _place(model, data, bid, x, y, z + 0.01,
                yaw=float(rng.uniform(-math.pi, math.pi)))
        # Settle this item before dropping the next, so we don't have
        # 8 items in flight at once (which blows the contact arena).
        for _ in range(400):
            mujoco.mj_step(model, data)
            if float(np.abs(data.qvel).max()) < 0.02:
                break

    # Final settle.
    for _ in range(1500):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel).max()) < 0.02:
            break
    # Zero residual velocities so nothing drifts after the reset.
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def scatter_bottles_in_bin(model, data, rng):
    """Reset for bottles_in_bin: place the bin against the back wall,
    scatter 3-6 bottles in the operator's working area in front of it.
    Bin and each bottle get an independent 0.8-1.2× scale jitter."""
    cx, cy, hx, hy, top_z = _table(model, data)

    # Bin: scale + place against the back of the table.
    bin_bid = _slot(model, "bottles_in_bin_bin")
    _scale(model, bin_bid, float(rng.uniform(0.7, 1.5)))
    _place(model, data, bin_bid, 0.0, 0.0, top_z)
    lo, hi = _aabb(model, data, bin_bid)
    bin_x = (cx + hx) - 0.05 - hi[0]
    bin_y = float(rng.uniform(-0.05, 0.05))
    bin_lo, bin_hi = _place(model, data, bin_bid, bin_x, bin_y, top_z + 0.01)

    # Settle the bin so it sits flat on the table.
    for _ in range(400):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel).max()) < 0.02:
            break

    # Re-measure the bin AABB AFTER the settle — the bin can shift while it
    # settles, and a stale pre-settle front face lets a bottle's footprint
    # creep into the bin (floor_at then drops the bottle on the rim and tips
    # the light bin off the table).
    bin_lo, bin_hi = _aabb(model, data, bin_bid)

    # Bottle scatter region: operator's reach in front of the bin. Cap the
    # back edge well clear of the bin's settled front face — the margin must
    # cover the widest scaled bottle's half-footprint so no bottle ever lands
    # on the bin. Region shifted +0.15 m back with the lengthened (3 ft)
    # table; sx_lo stays clear of the arm-mount housing at the table front,
    # and the y-extent uses the full table working area so up to 6 bottles
    # each get a clear, non-overlapping spot.
    BIN_MARGIN = 0.15  # m clearance from the bin's settled front face
    sx_lo = 0.55
    sx_hi = min(0.93, bin_lo[0] - BIN_MARGIN)
    sy_lo, sy_hi = -0.58, 0.58

    # Discover bottles from the model itself, not the global SCENE_POOLS:
    # the build bakes in exactly the chosen number of bottle bodies, so the
    # bodies actually present ARE the bottle set. Reading SCENE_POOLS instead
    # can desync (it's mutated per build) and leave real bottles unplaced.
    # Slot bodies are named ``bottle_slot_<letter>`` exactly; the attached
    # variant subtree under each carries a ``bottle_slot_<letter>__`` prefix,
    # so match the bare slot name only.
    bottle_ids = [b for b in range(model.nbody)
                  if _BOTTLE_SLOT_RE.fullmatch(
                      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")]

    # Measure each bottle after its scale: keep the actual AABB half-footprint
    # (hx, hy), not a disc radius — the floor query below is a true AABB test.
    item_info: list[tuple[int, float, float, float]] = []
    for bid in bottle_ids:
        _scale(model, bid, float(rng.uniform(0.7, 1.5)))
        _place(model, data, bid, 0.0, 0.0, top_z)
        lo, hi = _aabb(model, data, bid)
        hfx = 0.5 * (hi[0] - lo[0])
        hfy = 0.5 * (hi[1] - lo[1])
        item_info.append((bid, hfx, hfy, 0.05))

    def _geom_world_aabb(gid):
        """(amin, amax) world AABB of geom ``gid``."""
        c = model.geom_aabb[gid, 0:3]
        h = model.geom_aabb[gid, 3:6]
        R = data.geom_xmat[gid].reshape(3, 3)
        p = data.geom_xpos[gid]
        corners = (R @ (c + _CORNER_SIGNS * h).T).T + p
        return corners.min(axis=0), corners.max(axis=0)

    def floor_at(x, y, hfx, hfy, bid):
        """Release height for a bottle of half-footprint (hfx, hfy) centered
        at (x, y): the highest AABB top among every geom whose world AABB its
        footprint overlaps in xy — table, garbage can, already-placed bottles,
        AND walls. Released just above this, the bottle never penetrates
        anything at t=0; if it overlaps a 2 m wall it is lifted clear of it
        too. The body being placed is skipped."""
        bx_lo, bx_hi = x - hfx, x + hfx
        by_lo, by_hi = y - hfy, y + hfy
        z = top_z
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) == bid:
                continue
            if int(model.geom_contype[gid]) == 0:
                continue
            amin, amax = _geom_world_aabb(gid)
            if (bx_lo <= amax[0] and amin[0] <= bx_hi
                    and by_lo <= amax[1] and amin[1] <= by_hi):
                z = max(z, float(amax[2]))
        return z

    # Region must be valid even after the bottle's half-footprint is
    # subtracted; if a big bottle is wider than the slot, collapse the range
    # to its midpoint.
    def axis_range(lo, hi, half):
        a, b = lo + half, hi - half
        return (a, b) if a <= b else (0.5 * (lo + hi), 0.5 * (lo + hi))

    for bid, hfx, hfy, drop_h in item_info:
        mujoco.mj_forward(model, data)
        xa, xb = axis_range(sx_lo, sx_hi, hfx)
        ya, yb = axis_range(sy_lo, sy_hi, hfy)
        # Rejection-sample an xy whose footprint lands on the bare table —
        # nothing (can, wall, or an already-placed bottle) under it, so the
        # bottle never drops onto another and topples. ``floor_at`` returns
        # ``top_z`` exactly when the spot is clear. Keep the clearest sample
        # seen as the pick if no fully-clear spot turns up.
        best = None
        for _ in range(16):
            x = float(rng.uniform(xa, xb))
            y = float(rng.uniform(ya, yb))
            z = floor_at(x, y, hfx, hfy, bid)
            if best is None or z < best[2]:
                best = (x, y, z)
            if z <= top_z + 1e-6:  # clear spot found — accept it
                break
        x, y, z = best
        # Bottle standing upright = quat (1,0,0,0); random yaw for variety.
        yaw = float(rng.uniform(-math.pi, math.pi))
        _place(model, data, bid, x, y, z + drop_h, yaw=yaw)
        # Settle this bottle before dropping the next so we never have
        # several bottles in flight at once.
        for _ in range(400):
            mujoco.mj_step(model, data)
            if float(np.abs(data.qvel).max()) < 0.02:
                break

    # Final settle.
    for _ in range(1500):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel).max()) < 0.02:
            break
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def scatter_hang_mugs(model, data, rng):
    """Reset for the hang_mugs scene: place mug tree + 2 mugs in the
    middle working area (x ∈ [0.75, 1.00], y ∈ [-0.30, 0.30]).
    The x-range is shifted +0.15 m back with the lengthened (3 ft) table."""
    from mujoco_vr_teleop.variant_pools import HANG_MUGS_POOLS  # local to avoid import cycles
    cx, cy, hx, hy, top_z = _table(model, data)

    sx_lo, sx_hi = 0.75, 1.00
    sy_lo, sy_hi = -0.30, 0.30

    # Mug tree: upright, somewhere in the working region. Use roughly the
    # center, weighted toward the back so the operator has space to swing.
    tree = _slot(model, "hang_mugs_mug_tree")
    upright = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    _scale(model, tree, float(rng.uniform(0.8, 1.2)))
    tx = float(rng.uniform(sx_lo + 0.05, sx_hi - 0.05))
    ty = float(rng.uniform(sy_lo + 0.10, sy_hi - 0.10))
    tree_lo, tree_hi = _place(model, data, tree, tx, ty, top_z + 0.01,
                                quat_wxyz=upright)

    # Settle tree alone so it lands on the table.
    for _ in range(400):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel).max()) < 0.02:
            break

    # Mugs: scatter every mug present in this build (the pool count is
    # randomized 1..4 at build time, see HANG_MUGS_POOLS / build_scene).
    from mujoco_vr_teleop.variant_pools import SCENE_POOLS
    mug_pool_names = [p.name for p in SCENE_POOLS["hang_mugs"]
                       if p.name.startswith("hang_mugs_mug_") and p.name != "hang_mugs_mug_tree"]
    active_ids = [_slot(model, n) for n in mug_pool_names]
    item_info: list[tuple[int, float, float]] = []
    for bid in active_ids:
        _scale(model, bid, float(rng.uniform(0.9, 1.1)))
        _place(model, data, bid, 0.0, 0.0, top_z)
        lo, hi = _aabb(model, data, bid)
        r = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
        item_info.append((bid, r, 0.08))

    def floor_at(x, y, r, bid):
        z = top_z
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) == bid:
                continue
            if int(model.geom_contype[gid]) == 0:
                continue
            c = model.geom_aabb[gid, 0:3]
            h = model.geom_aabb[gid, 3:6]
            R = data.geom_xmat[gid].reshape(3, 3)
            p = data.geom_xpos[gid]
            corners = (R @ (c + _CORNER_SIGNS * h).T).T + p
            if ((corners[:, 0].min() - r) <= x <= (corners[:, 0].max() + r)
                    and (corners[:, 1].min() - r) <= y <= (corners[:, 1].max() + r)):
                z = max(z, float(corners[:, 2].max()))
        return z

    # Track already-placed mug centers + radii so we can reject overlapping
    # spawn points instead of stacking new mugs on top of old ones.
    GAP = 0.03  # min clearance between mug edges (m)
    placed_xyr: list[tuple[float, float, float]] = []
    # Tree footprint as an obstacle too.
    tree_cx = 0.5 * (tree_lo[0] + tree_hi[0])
    tree_cy = 0.5 * (tree_lo[1] + tree_hi[1])
    tree_r = 0.5 * math.hypot(tree_hi[0] - tree_lo[0], tree_hi[1] - tree_lo[1])
    placed_xyr.append((tree_cx, tree_cy, tree_r))

    for bid, r, _ in item_info:
        mujoco.mj_forward(model, data)
        # Find a spot that does NOT overlap any previously placed item.
        chosen = None
        for _ in range(256):
            x = float(rng.uniform(sx_lo + r, sx_hi - r))
            y = float(rng.uniform(sy_lo + r, sy_hi - r))
            ok = True
            for (px, py, pr) in placed_xyr:
                if math.hypot(x - px, y - py) < (r + pr + GAP):
                    ok = False
                    break
            if ok:
                chosen = (x, y, top_z)
                break
        if chosen is None:
            # Fallback: pick the lowest-floor spot we can find (used to happen
            # in scatter_dishrack). With 4 mugs in a small region this is rare.
            best = None
            for _ in range(16):
                x = float(rng.uniform(sx_lo + r, sx_hi - r))
                y = float(rng.uniform(sy_lo + r, sy_hi - r))
                z = floor_at(x, y, r, bid)
                if best is None or z < best[2]:
                    best = (x, y, z)
            chosen = best
        x, y, z = chosen
        placed_xyr.append((x, y, r))
        _place(model, data, bid, x, y, z + 0.05,  # drop from 5cm so it settles cleanly
                yaw=float(rng.uniform(-math.pi, math.pi)))
        for _ in range(400):
            mujoco.mj_step(model, data)
            if float(np.abs(data.qvel).max()) < 0.02:
                break

    # Final settle.
    for _ in range(1500):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel).max()) < 0.02:
            break
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Cups scene resets
#
# One scene, six tasks. Each build bakes in 3 or 6 cup slot bodies (named
# ``cup_slot_<letter>``) and, for ball tasks, a ``ball`` body. Cups are
# discovered from the model — the build is authoritative, not SCENE_POOLS.
# ---------------------------------------------------------------------------

def _cup_bids(model):
    """Body ids of every cup slot, in slot-letter order (a, b, c, ...)."""
    found: list[tuple[str, int]] = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if _CUP_SLOT_RE.fullmatch(name):
            found.append((name, b))
    found.sort()
    return [b for _, b in found]


def _ball_bid(model):
    """Body id of the ball, or None for a no-ball build."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    return bid if bid >= 0 else None


def _cup_build_scale() -> float:
    """The single uniform cup scale chosen at build time for the active cup
    scene. Falls back to 1.0 for non-cup scenes / bare builds."""
    from mujoco_vr_teleop.variant_pools import CUP_STACK_BUILD, CUP_BALL_BUILD
    # Whichever cup scene was built last has a non-None task; prefer it.
    for build in (CUP_STACK_BUILD, CUP_BALL_BUILD):
        if build.get("task") is not None and build.get("scale") is not None:
            return float(build["scale"])
    return 1.0


def _apply_cup_scale(model, data, rng):
    """Scale every cup by the single per-build cup scale, and the ball (if any)
    by a 0.95-1.05x jitter. The cup scale comes from the build (one shared
    factor for all cups, recorded for replay), not re-rolled here. Returns
    (cup_d, cup_h): the footprint diameter and height of a scaled cup."""
    cup_scale = _cup_build_scale()
    cup_ids = _cup_bids(model)
    for bid in cup_ids:
        _scale(model, bid, cup_scale)
    ball = _ball_bid(model)
    if ball is not None:
        _scale(model, ball, float(rng.uniform(0.95, 1.05)))
    # Measure one scaled cup laid at the origin.
    probe = cup_ids[0]
    _place(model, data, probe, 0.0, 0.0, 0.0)
    lo, hi = _aabb(model, data, probe)
    cup_d = max(hi[0] - lo[0], hi[1] - lo[1])
    cup_h = hi[2] - lo[2]
    return cup_d, cup_h


def _settle(model, data, steps):
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel).max()) < 0.02:
            break


def _final_settle(model, data):
    _settle(model, data, 1500)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _cup_floor_at(model, data, x, y, hf, bid):
    """Highest AABB top among colliding geoms whose xy footprint overlaps a
    square of half-extent ``hf`` centered at (x, y). The body ``bid`` is
    skipped. Mirrors scatter_bottles_in_bin.floor_at."""
    bx_lo, bx_hi = x - hf, x + hf
    by_lo, by_hi = y - hf, y + hf
    z = -np.inf
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            continue
        if int(model.geom_contype[gid]) == 0:
            continue
        c = model.geom_aabb[gid, 0:3]
        h = model.geom_aabb[gid, 3:6]
        R = data.geom_xmat[gid].reshape(3, 3)
        p = data.geom_xpos[gid]
        corners = (R @ (c + _CORNER_SIGNS * h).T).T + p
        amin = corners.min(axis=0)
        amax = corners.max(axis=0)
        if (bx_lo <= amax[0] and amin[0] <= bx_hi
                and by_lo <= amax[1] and amin[1] <= by_hi):
            z = max(z, float(amax[2]))
    return z


def scatter_cups(model, data, rng):
    """Scatter every cup upright in the operator's working area, ball (if any)
    dropped last at a clear spot. Used by stack_two_threes and pong-arrange."""
    cx, cy, hx, hy, top_z = _table(model, data)
    cup_d, cup_h = _apply_cup_scale(model, data, rng)
    hf = 0.5 * cup_d
    # Footprint used to test for a clear spot is padded so cups land with a
    # gap between rims — random yaw + a tiny drop must not clip a neighbour.
    hf_clear = hf + 0.5 * cup_d

    sx_lo, sx_hi = 0.55, 0.93
    sy_lo, sy_hi = -0.58, 0.58

    def drop(bid, drop_h, clear_hf):
        mujoco.mj_forward(model, data)
        best = None
        for _ in range(24):
            x = float(rng.uniform(sx_lo + hf, sx_hi - hf))
            y = float(rng.uniform(sy_lo + hf, sy_hi - hf))
            z = _cup_floor_at(model, data, x, y, clear_hf, bid)
            z = max(z, top_z)
            if best is None or z < best[2]:
                best = (x, y, z)
            if z <= top_z + 1e-6:
                break
        x, y, z = best
        # Drop from just above the resting surface so cups settle cleanly
        # instead of bouncing into a neighbour.
        _place(model, data, bid, x, y, z + drop_h,
                yaw=float(rng.uniform(-math.pi, math.pi)))
        _settle(model, data, 400)

    for bid in _cup_bids(model):
        drop(bid, 0.008, hf_clear)
    ball = _ball_bid(model)
    if ball is not None:
        drop(ball, 0.02, hf_clear)

    _final_settle(model, data)


def stack_cups(model, data, rng, n_stacks, per_stack):
    """Place cups as ``n_stacks`` upright nested stacks of ``per_stack`` cups.
    Used by unstack (2 stacks of 3)."""
    cx, cy, hx, hy, top_z = _table(model, data)
    cup_d, cup_h = _apply_cup_scale(model, data, rng)
    cup_ids = _cup_bids(model)
    # Nested cups rise by rim-to-rim spacing, not full height. ~0.3*cup_h is a
    # safe pitch for tapered party cups (deep nesting would interpenetrate).
    pitch = 0.30 * cup_h

    base_x = 0.72
    span_y = 0.26
    ys = np.linspace(-0.5 * span_y, 0.5 * span_y, n_stacks)
    for s in range(n_stacks):
        scx = base_x + float(rng.uniform(-0.03, 0.03))
        scy = float(ys[s]) + float(rng.uniform(-0.03, 0.03))
        for k in range(per_stack):
            bid = cup_ids[s * per_stack + k]
            _place(model, data, bid,
                    scx + float(rng.uniform(-0.002, 0.002)),
                    scy + float(rng.uniform(-0.002, 0.002)),
                    top_z + 0.005 + k * pitch,
                    yaw=float(rng.uniform(-math.pi, math.pi)))
            _settle(model, data, 400)

    _final_settle(model, data)


def triangle_cups(model, data, rng, jitter=False):
    """Arrange all 6 cups upright in a flat 3-2-1 triangle (pong-rack shape).
    Used by pong as the throw target; the ball, if present, is dropped near
    the operator side.

    ``jitter``: when True, adds a slight independent per-cup wobble on top of
    the global formation offset. pong passes True; the False default keeps a
    clean triangle for any caller that wants exact placement."""
    cx, cy, hx, hy, top_z = _table(model, data)
    cup_d, cup_h = _apply_cup_scale(model, data, rng)
    cup_ids = _cup_bids(model)

    # Row r (r=0,1,2) has 3-r cups. Rows step back in +x; cups in a row are
    # ~cup_d apart in y, a small gap so they settle without interpenetration.
    pitch = cup_d * 1.02
    row_dx = pitch * math.sqrt(3.0) / 2.0
    # Global jitter: one offset shifts the whole triangle.
    base_x = (cx + hx) - 0.18 + float(rng.uniform(-0.05, 0.05))
    base_y = cy + float(rng.uniform(-0.08, 0.08))
    cup_jit = 0.012 if jitter else 0.0

    idx = 0
    for r in range(3):
        count = 3 - r
        x = base_x + r * row_dx
        for c in range(count):
            y = base_y + (c - 0.5 * (count - 1)) * pitch
            _place(model, data, cup_ids[idx],
                    x + float(rng.uniform(-cup_jit, cup_jit)),
                    y + float(rng.uniform(-cup_jit, cup_jit)),
                    top_z + 0.005)
            _settle(model, data, 300)
            idx += 1

    ball = _ball_bid(model)
    if ball is not None:
        bx = 0.58 + float(rng.uniform(-0.02, 0.02))
        by = base_y + float(rng.uniform(-0.08, 0.08))
        _place(model, data, ball, bx, by, top_z + 0.05)
        _settle(model, data, 300)

    _final_settle(model, data)


def row_cups(model, data, rng):
    """Shuffle (shell-game) start: place 3 cups MOUTH-DOWN in a straight row
    across the operator's reach, with the ball hidden under a RANDOM cup. The
    operator lifts cups to find the ball, shuffles, then lifts again."""
    cx, cy, hx, hy, top_z = _table(model, data)
    cup_d, cup_h = _apply_cup_scale(model, data, rng)
    cup_ids = _cup_bids(model)
    ball = _ball_bid(model)

    pitch = cup_d + 0.06  # clear gap between cups
    # Global jitter: one offset shifts the whole row. Per-cup jitter: a slight
    # independent wobble on each cup so the row is not perfectly regular.
    base_x = 0.72 + float(rng.uniform(-0.05, 0.05))
    base_y = cy + float(rng.uniform(-0.07, 0.07))

    centers = []
    for c in range(len(cup_ids)):
        x = base_x + float(rng.uniform(-0.012, 0.012))
        y = base_y + (c - 1) * pitch + float(rng.uniform(-0.012, 0.012))
        centers.append((x, y))

    # Ball hidden under a random cup: place the ball first, then lower that
    # cup (mouth-down) over it.
    hide_idx = int(rng.integers(len(cup_ids)))
    if ball is not None:
        hx0, hy0 = centers[hide_idx]
        _place(model, data, ball, hx0, hy0, top_z + 0.005)
        _settle(model, data, 200)

    for c, bid in enumerate(cup_ids):
        x, y = centers[c]
        # Mouth-down (180 deg flip about x) + random yaw, so the cup is an
        # inverted dome that can cover the ball.
        h = 0.5 * float(rng.uniform(-math.pi, math.pi))
        yaw_q = (math.cos(h), 0.0, 0.0, math.sin(h))
        # quat = yaw * flip_x  (flip_x = (0,1,0,0)).
        quat = (-yaw_q[1], yaw_q[0], yaw_q[3], -yaw_q[2])
        _place(model, data, bid, x, y, top_z + 0.005, quat_wxyz=quat)
        _settle(model, data, 300)

    _final_settle(model, data)
