"""Generic scene-placement helpers.

These are scene-agnostic utilities for laying out free bodies onto a flat
surface without collisions. The flagship function, ``place_on_grid``, takes a
list of bodies + a region of the table + per-body radii, partitions the region
into a uniform grid of cells big enough for the largest object, and assigns
each body to a randomly-shuffled cell. Within each cell the body is placed at
the cell centre (+ optional small jitter) with random yaw composed onto its
baseline orientation. Because cells are disjoint and sized for the biggest
object, the resulting layout is collision-free by construction.

The same helpers are also used by the ``grid_place`` op in
``domain_randomization.py``, which exposes the function to the JSON DR config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import mujoco
import numpy as np


# ----------------------------- geometry helpers ---------------------------- #


def auto_body_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    *,
    collidable_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """World-aligned bounding cuboid (AABB) of a body's collision hull.

    After ``mj_forward(model, data)`` we have ``geom_xpos``/``geom_xmat`` for
    every geom. For each geom of the body we expand the corners of its local
    bounding box into world coords and take the global min/max. The result is
    a (min_xyz, max_xyz) pair, both relative to ``data.xpos[body_id]``.

    Use this when you want true xy footprint dimensions (width, depth) plus
    height, rather than a single bounding radius.

    With ``collidable_only=True``, geoms with both ``contype=0`` and
    ``conaffinity=0`` (pure visual decoration) are skipped — useful when the
    AABB is being used to place an object on a surface, where what matters is
    the bottom of the *collision* hull, not the bottom of a render mesh.
    """
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    if geom_ids.size == 0:
        return np.zeros(3), np.zeros(3)
    if collidable_only:
        geom_ids = np.array(
            [
                int(gid)
                for gid in geom_ids
                if int(model.geom_contype[int(gid)]) != 0
                or int(model.geom_conaffinity[int(gid)]) != 0
            ],
            dtype=int,
        )
        if geom_ids.size == 0:
            return np.zeros(3), np.zeros(3)
    body_xyz = np.asarray(data.xpos[body_id], dtype=float)
    gmin = np.full(3, np.inf)
    gmax = np.full(3, -np.inf)
    for gid in geom_ids:
        gpos = np.asarray(data.geom_xpos[gid], dtype=float)
        gmat = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        size = np.asarray(model.geom_size[gid], dtype=float)
        gtype = int(model.geom_type[gid])
        # Build the set of corner offsets we'll rotate by gmat.
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            corners = np.array([[ r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]])
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = size[:3]
            signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
            corners = signs * np.array([sx, sy, sz])
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            r = float(size[0]); h = float(size[1])
            # 8 corners around the cylinder bounding box (r, r, h half-extents).
            signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
            corners = signs * np.array([r, r, h])
        else:
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id < 0:
                continue
            vadr = int(model.mesh_vertadr[mesh_id])
            vnum = int(model.mesh_vertnum[mesh_id])
            corners = model.mesh_vert[vadr : vadr + vnum]
        world = (gmat @ corners.T).T + gpos
        gmin = np.minimum(gmin, world.min(axis=0))
        gmax = np.maximum(gmax, world.max(axis=0))
    return gmin - body_xyz, gmax - body_xyz


def auto_body_radius(
    model: mujoco.MjModel,
    body_id: int,
    *,
    data: mujoco.MjData | None = None,
) -> float:
    """Approximate xy bounding-circle radius of a body in its current pose.

    Computed from the body's world-aligned AABB: half of the xy diagonal.
    Requires a forward-stepped ``data`` for the geom transforms. Falls back
    to a coarse model-only estimate when ``data`` is ``None``.
    """
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    if geom_ids.size == 0:
        return 0.0
    if data is not None:
        bb_min, bb_max = auto_body_aabb(model, data, body_id)
        width = float(bb_max[0] - bb_min[0])
        depth = float(bb_max[1] - bb_min[1])
        return 0.5 * float(np.hypot(width, depth))
    # Model-only fallback: take each geom's size projected onto xy, ignoring
    # body rotation. Good enough for a startup estimate.
    max_r = 0.0
    for gid in geom_ids:
        size = np.asarray(model.geom_size[gid], dtype=float)
        gpos = np.asarray(model.geom_pos[gid], dtype=float)
        gtype = int(model.geom_type[gid])
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            local_r = float(size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            local_r = float(np.hypot(size[0], size[1]))
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            local_r = float(size[0])
        else:
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id < 0:
                continue
            vadr = int(model.mesh_vertadr[mesh_id])
            vnum = int(model.mesh_vertnum[mesh_id])
            verts = model.mesh_vert[vadr : vadr + vnum]
            local_r = float(np.max(np.hypot(verts[:, 0], verts[:, 1])))
        offset_xy = float(np.hypot(gpos[0], gpos[1]))
        max_r = max(max_r, offset_xy + local_r)
    return max_r


def body_z_offset_to_bottom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> float:
    """Distance from a body's origin to the lowest point of its collision hull.

    Computed from the body's world-aligned AABB (see :func:`auto_body_aabb`)
    rather than ``geom_rbound``. The bounding-sphere radius overestimates a
    thin object's height enormously (a flat plate's bounding sphere is the
    plate's *diameter*), so using it for z-placement lifts plates a dozen
    centimetres into the air. The AABB gives the true z extent.

    The return value is positive when the body's collision hull extends below
    the body origin (the usual case for objects whose origin is in the middle
    of their volume). Add this to your desired ``qpos.z`` so the bottom of the
    body sits at z = 0.
    """
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    if geom_ids.size == 0:
        return 0.0
    bb_min, _ = auto_body_aabb(model, data, body_id, collidable_only=True)
    # bb_min[2] is the lowest point of the body relative to body origin.
    # Negate to get the offset that puts that lowest point at z=0.
    return float(-bb_min[2])


# ------------------------------- grid maths -------------------------------- #


@dataclass
class GridLayout:
    """Result of laying out a grid over a rectangular region."""

    cell_size: float
    cell_centers: list[tuple[float, float]]
    cell_indices: list[tuple[int, int]]
    nx: int
    ny: int
    region: tuple[tuple[float, float], tuple[float, float]]


def compute_grid_cells(
    region_x: tuple[float, float],
    region_y: tuple[float, float],
    cell_size: float,
    *,
    edge_margin: float = 0.0,
    avoid_rects: Sequence[dict] = (),
) -> GridLayout:
    """Tile a rectangle with square cells of the given size.

    Cells whose centre falls inside any ``avoid_rects`` entry
    (``{"x": [lo, hi], "y": [lo, hi]}``) are dropped. The remaining cell
    centres are returned in row-major order.
    """
    x_lo = float(region_x[0]) + float(edge_margin)
    x_hi = float(region_x[1]) - float(edge_margin)
    y_lo = float(region_y[0]) + float(edge_margin)
    y_hi = float(region_y[1]) - float(edge_margin)
    width = x_hi - x_lo
    height = y_hi - y_lo
    if cell_size <= 0:
        raise ValueError(f"cell_size must be > 0, got {cell_size}")
    nx = int(np.floor(width / cell_size))
    ny = int(np.floor(height / cell_size))
    if nx <= 0 or ny <= 0:
        raise ValueError(
            f"region {(x_lo, x_hi, y_lo, y_hi)} too small for cell_size {cell_size}"
        )

    # Centre the grid inside the region so any leftover slack splits evenly.
    grid_w = nx * cell_size
    grid_h = ny * cell_size
    x_origin = x_lo + (width - grid_w) / 2.0 + cell_size / 2.0
    y_origin = y_lo + (height - grid_h) / 2.0 + cell_size / 2.0

    centers: list[tuple[float, float]] = []
    indices: list[tuple[int, int]] = []
    for j in range(ny):
        for i in range(nx):
            cx = x_origin + i * cell_size
            cy = y_origin + j * cell_size
            if _in_any_avoid(cx, cy, avoid_rects, cell_size / 2.0):
                continue
            centers.append((cx, cy))
            indices.append((i, j))
    return GridLayout(
        cell_size=float(cell_size),
        cell_centers=centers,
        cell_indices=indices,
        nx=nx,
        ny=ny,
        region=((x_lo, x_hi), (y_lo, y_hi)),
    )


def _pick_non_adjacent_cells(
    rng: np.random.Generator,
    cell_indices: Sequence[tuple[int, int]],
    needed: int,
    *,
    min_separation: int = 1,
) -> list[int]:
    """Pick ``needed`` cell indices into ``cell_indices`` with a Chebyshev gap.

    Greedy: shuffle the cell order, then walk through and accept each cell
    whose grid index is at Chebyshev distance > ``min_separation`` from every
    previously-accepted cell. If we run out of candidates before satisfying
    ``needed``, relax to ``min_separation - 1`` and try again, all the way down
    to 0 (which becomes "any free cell"). Falls back to "any cell" if there
    still aren't enough.

    Returns indices into ``cell_indices`` (and therefore into the parallel
    ``cell_centers`` list).
    """
    total = len(cell_indices)
    if needed > total:
        raise ValueError(
            f"grid_place: need {needed} cells but only {total} available"
        )

    def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    for sep in range(max(0, min_separation), -1, -1):
        order = rng.permutation(total)
        picked: list[int] = []
        for idx in order:
            ij = cell_indices[int(idx)]
            ok = True
            for prev_idx in picked:
                if chebyshev(ij, cell_indices[prev_idx]) <= sep:
                    ok = False
                    break
            if ok:
                picked.append(int(idx))
                if len(picked) == needed:
                    return picked
    # Shouldn't happen — sep=0 means "anything goes". Fall back to first N.
    return list(rng.permutation(total)[:needed])


def _in_any_avoid(
    cx: float,
    cy: float,
    avoid_rects: Sequence[dict],
    half: float,
) -> bool:
    """A cell is masked out if any part of its footprint overlaps an avoid rect.

    We treat the cell footprint as an axis-aligned square of side ``cell_size``
    centred at ``(cx, cy)``. This is strict: it discards cells that barely
    clip the rack zone rather than letting an object's footprint poke into it.
    """
    for rect in avoid_rects:
        xr = rect.get("x", (0.0, 0.0))
        yr = rect.get("y", (0.0, 0.0))
        if cx + half > xr[0] and cx - half < xr[1] and cy + half > yr[0] and cy - half < yr[1]:
            return True
    return False


# ---------------------------- quat composition ----------------------------- #


def _quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """wxyz quaternion from XYZ-intrinsic roll/pitch/yaw degrees."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    xyzw = rot.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def _qmul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


# -------------------------- core placement function ------------------------ #


@dataclass
class GridPlaceSpec:
    """Configuration for :func:`place_on_grid`.

    All distances are in meters and all angles in degrees. ``radii`` is a list
    of (name regex, radius) entries searched in order; the first match wins,
    falling back to ``default_radius`` (or, if that is ``None``,
    :func:`auto_body_radius`).
    """

    region_x: tuple[float, float]
    region_y: tuple[float, float]
    surface_z: float
    radii: list[tuple[str, float]] = field(default_factory=list)
    default_radius: float | None = None
    cell_size: float | None = None  # None => auto from max radius
    cell_gap: float = 0.01
    edge_margin: float = 0.02
    avoid_rects: Sequence[dict] = ()
    clearance: float = 0.003
    jitter_xy: float = 0.0
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0)
    yaw_step_deg: float = 10.0
    roll_range_deg: tuple[float, float] = (0.0, 0.0)
    pitch_range_deg: tuple[float, float] = (0.0, 0.0)
    settle_seconds: float = 0.2
    settle_per_body: float = 0.0
    hide_offset: tuple[float, float, float] = (0.0, 0.0, -5.0)
    # Minimum grid-index separation between any two placed cells. 0 = neighbors
    # are allowed; 1 = no 8-connected neighbors (skip-one chessboard pattern).
    # Higher values give bigger buffers between objects at the cost of using
    # fewer of the available cells.
    min_cell_separation: int = 0


def _radius_for(spec: GridPlaceSpec, model: mujoco.MjModel, body_id: int) -> float:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    for pattern, radius in spec.radii:
        if re.fullmatch(pattern, name):
            return float(radius)
    if spec.default_radius is not None:
        return float(spec.default_radius)
    return auto_body_radius(model, body_id)


def _body_freejoint_qadr(model: mujoco.MjModel, body_id: int) -> int | None:
    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    for joint_id in range(jnt_adr, jnt_adr + jnt_num):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint_id])
    return None


def _body_freejoint_dadr(model: mujoco.MjModel, body_id: int) -> int | None:
    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    for joint_id in range(jnt_adr, jnt_adr + jnt_num):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_dofadr[joint_id])
    return None


def _sample_uniform_step(rng, lo: float, hi: float, step: float) -> float:
    """Uniform sample on [lo, hi], snapped to ``step`` increments."""
    value = float(rng.uniform(lo, hi))
    if step > 0:
        value = round(value / step) * step
        value = max(lo, min(hi, value))
    return value


def place_on_grid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    body_ids: Sequence[int],
    spec: GridPlaceSpec,
    *,
    baseline_quats: dict[int, np.ndarray] | None = None,
) -> dict:
    """Battleship-style grid placement of free bodies.

    Returns a dict suitable for logging (cell_size, assignments, etc.).

    ``baseline_quats`` lets the caller supply each body's resting quaternion
    (typically captured before any randomization). Sampled euler rotations are
    composed *onto* this baseline so e.g. a bottle with a "laying on its side"
    baseline keeps that orientation even with random yaw applied.
    """
    if not body_ids:
        return {"placements": []}

    # Per-body radius and baseline quat.
    radii = {bid: _radius_for(spec, model, bid) for bid in body_ids}
    if baseline_quats is None:
        baseline_quats = {bid: model.body_quat[bid].copy() for bid in body_ids}

    cell_size = spec.cell_size
    if cell_size is None:
        cell_size = 2.0 * max(radii.values()) + spec.cell_gap
    layout = compute_grid_cells(
        spec.region_x,
        spec.region_y,
        cell_size,
        edge_margin=spec.edge_margin,
        avoid_rects=spec.avoid_rects,
    )
    if len(layout.cell_centers) < len(body_ids):
        raise ValueError(
            f"grid_place: need {len(body_ids)} cells but only {len(layout.cell_centers)} "
            f"available (grid {layout.nx}x{layout.ny}, cell {cell_size:.3f} m, "
            f"region x={spec.region_x}, y={spec.region_y})"
        )

    # Park everything underground first so partially-placed state can't make
    # the solver upset when we run mj_forward to read body_z_offsets.
    qaddrs = {bid: _body_freejoint_qadr(model, bid) for bid in body_ids}
    dadrs = {bid: _body_freejoint_dadr(model, bid) for bid in body_ids}
    for idx, bid in enumerate(body_ids):
        qadr = qaddrs[bid]
        if qadr is None:
            raise ValueError(f"body {bid} has no freejoint")
        park = [
            spec.hide_offset[0] + 0.4 * (idx % 6),
            spec.hide_offset[1] + 0.4 * (idx // 6),
            spec.hide_offset[2],
        ]
        data.qpos[qadr : qadr + 3] = park
        data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        dadr = dadrs[bid]
        if dadr is not None:
            data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    # Assign bodies to cells: bigger bodies first; cells shuffled with a
    # neighbor-skip pass so we don't place two objects in directly-adjacent
    # cells (their rims could brush each other at the cell boundary).
    ordered_bodies = sorted(body_ids, key=lambda b: -radii[b])
    cell_assignment = _pick_non_adjacent_cells(
        rng,
        layout.cell_indices,
        len(ordered_bodies),
        min_separation=spec.min_cell_separation,
    )
    assignments: list[dict] = []

    for slot_idx, bid in enumerate(ordered_bodies):
        cell_idx = cell_assignment[slot_idx]
        cx, cy = layout.cell_centers[cell_idx]
        qadr = qaddrs[bid]

        # Sample orientation and compose with the body's baseline quat.
        yaw = _sample_uniform_step(rng, *spec.yaw_range_deg, spec.yaw_step_deg)
        roll = _sample_uniform_step(rng, *spec.roll_range_deg, max(spec.yaw_step_deg, 1.0))
        pitch = _sample_uniform_step(rng, *spec.pitch_range_deg, max(spec.yaw_step_deg, 1.0))
        delta_quat = _quat_from_euler_deg(roll, pitch, yaw)
        base_quat = np.asarray(baseline_quats[bid], dtype=float)
        base_norm = float(np.linalg.norm(base_quat))
        if base_norm < 1e-9:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            base_quat = base_quat / base_norm
        quat = _qmul_wxyz(delta_quat, base_quat)
        quat = quat / float(np.linalg.norm(quat))

        # Tentative placement at z=0 so we can measure how far the bottom of
        # the body's collision hull falls below the body origin in its sampled
        # pose. We then offset to make that bottom sit at surface_z + clearance.
        max_jitter = max(0.0, layout.cell_size / 2.0 - radii[bid] - 0.005)
        jx = float(rng.uniform(-1.0, 1.0)) * min(spec.jitter_xy, max_jitter)
        jy = float(rng.uniform(-1.0, 1.0)) * min(spec.jitter_xy, max_jitter)
        data.qpos[qadr : qadr + 3] = [cx + jx, cy + jy, spec.surface_z + 0.1]
        data.qpos[qadr + 3 : qadr + 7] = quat
        dadr = dadrs[bid]
        if dadr is not None:
            data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        z_offset = body_z_offset_to_bottom(model, data, bid)
        target_z = spec.surface_z + spec.clearance + z_offset
        data.qpos[qadr + 2] = target_z
        if dadr is not None:
            data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(model, data)

        # Settle just this body in isolation before placing the next one. All
        # previously-placed bodies are already at rest on the table; the new
        # one drops the clearance gap, contacts the table, and gives up its
        # energy without sloshing anyone else. Without this per-body step the
        # collective settle after placement leaves curved-bottom bodies (mugs)
        # rocking off-axis indefinitely.
        if spec.settle_per_body > 0:
            steps = int(round(spec.settle_per_body / model.opt.timestep))
            for _ in range(max(0, steps)):
                mujoco.mj_step(model, data)

        assignments.append(
            {
                "body_id": int(bid),
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid),
                "cell_index": cell_idx,
                "cell_xy": [cx, cy],
                "jitter_xy": [jx, jy],
                "radius": radii[bid],
                "yaw_deg": yaw,
                "roll_deg": roll,
                "pitch_deg": pitch,
                "final_pos": data.qpos[qadr : qadr + 3].tolist(),
            }
        )

    # One settle to absorb any residual penetration from the placement.
    # Don't zero velocities at the end — the streamer's reset_scene runs an
    # additional `settle()` afterwards, and forcibly zeroing here while the
    # solver is mid-impulse just re-injects energy on the next step.
    if spec.settle_seconds > 0:
        steps = int(round(spec.settle_seconds / model.opt.timestep))
        for _ in range(max(0, steps)):
            mujoco.mj_step(model, data)

    return {
        "cell_size": layout.cell_size,
        "grid_nx": layout.nx,
        "grid_ny": layout.ny,
        "region": [list(spec.region_x), list(spec.region_y)],
        "avoid_rects": list(spec.avoid_rects),
        "placements": assignments,
        "unused_cells": [
            list(layout.cell_centers[i])
            for i in range(len(layout.cell_centers))
            if i not in set(cell_assignment)
        ],
    }
