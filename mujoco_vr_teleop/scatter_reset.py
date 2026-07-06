"""Drop free-jointed bodies onto a surface without penetration.

scatter_bodies(model, data, rng, body_ids, region, priorities=None):
  - measure each body's xy radius, z bottom offset, total subtree mass
  - collect obstacle xy disks (arm/hand links etc. above the surface)
  - drop bodies one at a time, ordered by (priority asc, mass desc).
    Each drop:
      * rejection-sample (x,y) inside region; reject inside obstacles or
        inside any already-placed body's xy circle
      * teleport to (x, y, surface + clearance - z_bottom)
      * mj_step for settle_between_s
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from mujoco_vr_teleop.placement_helpers import auto_body_aabb


@dataclass
class ScatterRegion:
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    surface_z: float
    obstacle_margin: float = 0.08   # min clearance from arms/hands/etc.
    settle_between_s: float = 0.3
    final_settle_s: float = 0.5


_OBSTACLE_SKIP = frozenset({"world", "play_table"})
_OBSTACLE_HEIGHT_BAND = 0.40


def _freejoint_qadr(model, body_id):
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) == body_id and int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[jid])
    return None


def _freejoint_dadr(model, body_id):
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) == body_id and int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_dofadr[jid])
    return None


def _subtree(model, body_id):
    out = {body_id}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in out and int(model.body_parentid[b]) in out:
                out.add(b)
                changed = True
    return sorted(out)


def _measure(model, data, body_id):
    """(xy_radius, z_bottom_offset, mass) over the body's subtree."""
    bids = _subtree(model, body_id)
    root = np.asarray(data.xpos[body_id], dtype=float)
    gmin = np.full(3, np.inf)
    gmax = np.full(3, -np.inf)
    mass = 0.0
    for bid in bids:
        mass += float(model.body_mass[bid])
        bmin, bmax = auto_body_aabb(model, data, bid, collidable_only=True)
        if not np.isfinite(bmin).all():
            continue
        off = np.asarray(data.xpos[bid], dtype=float) - root
        gmin = np.minimum(gmin, bmin + off)
        gmax = np.maximum(gmax, bmax + off)
    if not np.isfinite(gmin).all():
        return 0.01, 0.0, mass
    xy_r = max(abs(gmin[0]), abs(gmax[0]), abs(gmin[1]), abs(gmax[1]))
    return float(xy_r), float(gmin[2]), mass


def _obstacle_disks(model, data, scatter_ids, region):
    skip = set()
    for bid in scatter_ids:
        skip.update(_subtree(model, bid))
    disks = []
    for bid in range(model.nbody):
        if bid in skip:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name in _OBSTACLE_SKIP:
            continue
        bmin, bmax = auto_body_aabb(model, data, bid, collidable_only=True)
        if not np.isfinite(bmin).all():
            continue
        root = np.asarray(data.xpos[bid], dtype=float)
        wmin = bmin + root
        wmax = bmax + root
        if wmin[2] > region.surface_z + _OBSTACLE_HEIGHT_BAND:
            continue
        if wmax[2] < region.surface_z - 0.05:
            continue
        cx = 0.5 * (wmin[0] + wmax[0])
        cy = 0.5 * (wmin[1] + wmax[1])
        r = 0.5 * max(wmax[0] - wmin[0], wmax[1] - wmin[1])
        if r < 0.005:
            continue
        disks.append((float(cx), float(cy), float(r)))
    return disks


def scatter_bodies(model, data, rng, body_ids, region: ScatterRegion,
                   priorities: dict | None = None):
    """Drop bodies onto a surface without penetration.

    Bodies are dropped one at a time, ordered by:
      1. ``priorities[body_id]`` ascending (lower = drops earlier).
         Missing entries get priority +inf.
      2. mass descending (heavier first).

    Each body is teleported to a rejection-sampled (x, y) inside ``region``
    that:
      - is at least ``region.gap`` clear of every already-placed body's
        xy circle (rejects forced penetration with placed items)
      - is at least ``region.obstacle_margin`` clear of any obstacle disk
        (arm/hand bodies within the drop column)
      - has the body's lowest collidable point ``clearance`` above
        ``surface_z`` (no surface penetration on spawn)

    Then mj_step for ``settle_between_s`` so the body comes to rest before
    the next one drops.
    """
    body_ids = list(body_ids)
    if not body_ids:
        return {"placements": []}

    priorities = priorities or {}
    qadr = {b: _freejoint_qadr(model, b) for b in body_ids}
    dadr = {b: _freejoint_dadr(model, b) for b in body_ids}
    for b in body_ids:
        if qadr[b] is None:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}"
            raise ValueError(f"scatter_bodies: body {name!r} has no freejoint")

    # Park scatter bodies far away so they don't pollute measurements.
    for i, b in enumerate(body_ids):
        data.qpos[qadr[b] : qadr[b] + 3] = [10.0 + 0.5 * i, -10.0, -5.0]
        data.qpos[qadr[b] + 3 : qadr[b] + 7] = [1.0, 0, 0, 0]
        if dadr[b] is not None:
            data.qvel[dadr[b] : dadr[b] + 6] = 0.0
    mujoco.mj_forward(model, data)

    measured = {b: _measure(model, data, b) for b in body_ids}
    obstacles = _obstacle_disks(model, data, body_ids, region)

    def sort_key(b):
        return (priorities.get(b, float("inf")), -measured[b][2])
    ordered = sorted(body_ids, key=sort_key)

    settle_n = max(0, int(round(region.settle_between_s / model.opt.timestep)))
    # Edge margin baked in: bodies must fit entirely inside the region.
    x_lo, x_hi = region.x_lo, region.x_hi
    y_lo, y_hi = region.y_lo, region.y_hi

    # Each placed item is (x, y, xy_r, top_z) so we know how high to spawn
    # above anything already on the table.
    placed: list[tuple[float, float, float, float]] = []
    placements = []
    # Sample candidates per body, pick the one farthest from already-placed
    # items. Hard reject obstacles (arms/hands). Overlapping items in xy is
    # fine — we spawn above them so no forced penetration at t=0.
    N_CANDIDATES = 64
    for b in ordered:
        xy_r, z_off, mass = measured[b]
        best = None
        for _ in range(N_CANDIDATES):
            x = float(rng.uniform(x_lo + xy_r, x_hi - xy_r))
            y = float(rng.uniform(y_lo + xy_r, y_hi - xy_r))
            # Reject if within obstacle_margin of any arm/hand body.
            if any(np.hypot(x - ox, y - oy) - or_ < region.obstacle_margin
                   for (ox, oy, or_) in obstacles):
                continue
            # Score = distance to the nearest placed item (max wins -> open space).
            item_clear = min(
                (np.hypot(x - px, y - py) - (xy_r + pr) for (px, py, pr, _) in placed),
                default=float("inf"),
            )
            if best is None or item_clear > best[0]:
                best = (item_clear, x, y)
        if best is None:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}"
            raise ValueError(f"scatter_bodies: no spawn for {name!r} (region covered by obstacles)")
        _, x, y = best
        # Spawn z = top of surface OR top of any overlapping placed item.
        floor_top = region.surface_z
        for (px, py, pr, ptop) in placed:
            if np.hypot(x - px, y - py) < (xy_r + pr) and ptop > floor_top:
                floor_top = ptop
        spawn_z = floor_top - z_off + 0.01  # 1cm above floor surface
        yaw = float(rng.uniform(-np.pi, np.pi))
        half = 0.5 * yaw
        data.qpos[qadr[b] : qadr[b] + 3] = [x, y, spawn_z]
        data.qpos[qadr[b] + 3 : qadr[b] + 7] = [np.cos(half), 0, 0, np.sin(half)]
        if dadr[b] is not None:
            data.qvel[dadr[b] : dadr[b] + 6] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(settle_n):
            mujoco.mj_step(model, data)
        if dadr[b] is not None:
            data.qvel[dadr[b] : dadr[b] + 6] = 0.0
        # Record this body's post-settle TOP z so subsequent drops know how
        # high to start.
        top_z = float(data.qpos[qadr[b] + 2]) + (xy_r if z_off == 0.0 else -z_off)
        placed.append((x, y, xy_r, top_z))
        placements.append({
            "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}",
            "xy": (x, y),
            "final_z": float(data.qpos[qadr[b] + 2]),
            "xy_radius": xy_r,
            "z_bottom_offset": z_off,
            "mass": mass,
            "priority": priorities.get(b, float("inf")),
        })

    n_final = int(round(region.final_settle_s / model.opt.timestep))
    for _ in range(n_final):
        mujoco.mj_step(model, data)
    return {"placements": placements}
