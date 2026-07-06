from __future__ import annotations

import re

import mujoco
import numpy as np


def qmul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def body_box_geom_id(dr, body_id: int) -> int | None:
    geom_ids = np.flatnonzero(dr.model.geom_bodyid == body_id)
    boxes = [int(gid) for gid in geom_ids if dr.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX]
    return boxes[0] if boxes else None


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    xmat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(xmat, np.asarray(quat, dtype=float).reshape(4))
    return xmat.reshape(3, 3)


def yaw_quat(deg: float) -> np.ndarray:
    half = np.deg2rad(float(deg)) * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def half_extent_along(size: np.ndarray, xmat: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return 0.0
    return float(np.sum(np.abs(xmat.T @ (axis / norm)) * np.asarray(size, dtype=float).reshape(3)))


def body_index(dr, body_id: int) -> int:
    name = mujoco.mj_id2name(dr.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    match = re.search(r"([0-9]+)$", name)
    return int(match.group(1)) if match else body_id


def surface_top(dr, action: dict, snapshot: dict) -> tuple[str, float]:
    surface = action.get("surface") if isinstance(action.get("surface"), dict) else {}
    surface_geom = str(surface.get("geom_top", "table_plane"))
    surface_gid = dr._geom_id(surface_geom)
    if surface_gid < 0:
        raise ValueError(f"Unknown geom for jenga assembly surface: {surface_geom}")
    table_top = float(
        snapshot["geom_xpos"][surface_gid, 2]
        + snapshot["geom_size"][surface_gid, 2]
        + float(surface.get("clearance", 0.0))
    )
    return surface_geom, table_top


def scatter_blocks(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    body_ids: list[int],
    *,
    initial_quats: dict[int, np.ndarray],
    surface_top_z: float,
    center: tuple[float, float] | np.ndarray,
    square_half_width: float,
    drop_height: float = 0.0,
    drop_seconds: float = 0.15,
    hide_offset: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.5),
    min_drop_clearance: float = 0.25,
) -> list[dict]:
    """Drop a set of free-jointed bodies into a square region and settle them.

    Bodies are processed in the given order. Each body is teleported far away
    (using ``hide_offset``) until its turn; it is then placed above the pile
    surface (the table top plus anything already settled) at its caller-chosen
    quaternion, dropped, and physics is stepped for ``drop_seconds`` so it
    comes to rest. Returns per-block sample dicts for caller-side logging.

    Caller is responsible for picking ``initial_quats`` (yaw-only, SO(3),
    canonical poses — whatever policy is desired). Each body must have a
    freejoint and a box collision geom.

    Every block is dropped from at least ``min_drop_clearance`` above the
    table surface so it falls *past* a hand the operator may have left in the
    pile region, rather than spawning inside it.
    """
    if not body_ids:
        return []

    center = np.asarray(center, dtype=float).reshape(-1)[:2]
    hide_offset = np.asarray(hide_offset, dtype=float).reshape(3)

    def body_freejoint_qadr(body_id: int) -> int | None:
        jadr = int(model.body_jntadr[body_id])
        jnum = int(model.body_jntnum[body_id])
        for jid in range(jadr, jadr + jnum):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(model.jnt_qposadr[jid])
        return None

    def body_freejoint_dadr(body_id: int) -> int | None:
        jadr = int(model.body_jntadr[body_id])
        jnum = int(model.body_jntnum[body_id])
        for jid in range(jadr, jadr + jnum):
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(model.jnt_dofadr[jid])
        return None

    qadr_by_body: dict[int, int] = {}
    dadr_by_body: dict[int, int | None] = {}
    geom_by_body: dict[int, int] = {}
    for bid in body_ids:
        qadr = body_freejoint_qadr(bid)
        gid = body_box_geom_id_for_model(model, bid)
        if qadr is None or gid is None:
            raise ValueError(f"body {bid} missing freejoint or box collision geom")
        qadr_by_body[bid] = qadr
        dadr_by_body[bid] = body_freejoint_dadr(bid)
        geom_by_body[bid] = gid

    def zero_body_velocity(body_id: int) -> None:
        dadr = dadr_by_body[body_id]
        if dadr is not None:
            data.qvel[dadr : dadr + 6] = 0.0

    def hide_body(body_id: int, index: int) -> None:
        qadr = qadr_by_body[body_id]
        pos = np.array([center[0], center[1], 0.0]) + hide_offset
        pos[:2] += [0.2 * (index % 6), 0.2 * (index // 6)]
        data.qpos[qadr : qadr + 3] = pos
        zero_body_velocity(body_id)

    def placed_top(placed_body_ids: list[int]) -> float:
        top = surface_top_z
        for placed_body_id in placed_body_ids:
            gid = geom_by_body[placed_body_id]
            xmat = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            top = max(
                top,
                float(
                    data.geom_xpos[gid, 2]
                    + half_extent_along(model.geom_size[gid], xmat, np.array([0.0, 0.0, 1.0]))
                ),
            )
        return top

    samples: list[dict] = []
    for bid in body_ids:
        gid = geom_by_body[bid]
        xy = center + rng.uniform(-square_half_width, square_half_width, size=2)
        quat = np.asarray(initial_quats[bid], dtype=float).reshape(4)
        quat = quat / np.linalg.norm(quat)
        xmat = quat_to_mat(quat)
        half_z = half_extent_along(model.geom_size[gid], xmat, np.array([0.0, 0.0, 1.0]))
        samples.append(
            {
                "body_id": int(bid),
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid),
                "xy": xy.tolist(),
                "quat": quat.tolist(),
                "half_z": float(half_z),
            }
        )

    for idx, bid in enumerate(body_ids):
        hide_body(bid, idx)
    mujoco.mj_forward(model, data)

    placed: list[int] = []
    for idx, sample in enumerate(samples):
        bid = int(sample["body_id"])
        qadr = qadr_by_body[bid]
        for hidden_idx, hidden_bid in enumerate(body_ids[idx + 1 :]):
            hide_body(hidden_bid, hidden_idx)
        mujoco.mj_forward(model, data)

        drop_start_pos = np.array([sample["xy"][0], sample["xy"][1], 0.0], dtype=float)
        # Drop from above the pile surface, but never lower than
        # `min_drop_clearance` above the table — so the block falls past a
        # hand left in the pile region instead of spawning inside it.
        pile_drop_z = placed_top(placed) + float(sample["half_z"]) + drop_height
        drop_start_pos[2] = max(
            pile_drop_z,
            surface_top_z + float(sample["half_z"]) + min_drop_clearance,
        )
        data.qpos[qadr : qadr + 3] = drop_start_pos
        data.qpos[qadr + 3 : qadr + 7] = np.asarray(sample["quat"], dtype=float)
        zero_body_velocity(bid)
        mujoco.mj_forward(model, data)

        if drop_seconds > 0.0:
            for _ in range(max(0, round(drop_seconds / model.opt.timestep))):
                mujoco.mj_step(model, data)
        placed.append(bid)
        for placed_bid in placed:
            zero_body_velocity(placed_bid)
        mujoco.mj_forward(model, data)
        sample["drop_start_pos"] = drop_start_pos.tolist()
        sample["final_pos"] = data.qpos[qadr : qadr + 3].copy().tolist()

    return samples


def body_box_geom_id_for_model(model: mujoco.MjModel, body_id: int) -> int | None:
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    boxes = [int(gid) for gid in geom_ids if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX]
    return boxes[0] if boxes else None


def apply_scatter(dr, action: dict, snapshot: dict) -> None:
    body_ids = sorted(
        [
            bid
            for bid in dr._resolve_bodies(action.get("target"))
            if dr._body_freejoint_qadr(bid) is not None and body_box_geom_id(dr, bid) is not None
        ],
        key=lambda body_id: body_index(dr, body_id),
    )
    if not body_ids:
        return

    surface_geom, table_top = surface_top(dr, action, snapshot)
    source_positions = np.array(
        [
            snapshot["qpos"][dr._body_freejoint_qadr(bid) : dr._body_freejoint_qadr(bid) + 3]
            for bid in body_ids
        ],
        dtype=float,
    )
    center = dr._sample_vec3(action.get("center", source_positions.mean(axis=0).tolist()))
    square_half_width = float(action.get("square_half_width", 0.2))
    drop_height = float(action.get("drop_height", 0.0))
    drop_seconds = float(action.get("drop_seconds", 0.15))
    hide_offset = dr._sample_vec3(action.get("hide_offset", [0.0, 0.0, 1.5]))

    # Sample yaw-only quats per body using the DR sampler (back-compat with
    # existing JSON callers, which only specify yaw_deg).
    initial_quats: dict[int, np.ndarray] = {}
    yaw_degs: dict[int, float] = {}
    for bid in body_ids:
        qadr = dr._body_freejoint_qadr(bid)
        yaw = dr._sample_scalar(action.get("yaw_deg", 0.0))
        quat = qmul_wxyz(yaw_quat(yaw), snapshot["qpos"][qadr + 3 : qadr + 7])
        initial_quats[bid] = quat / np.linalg.norm(quat)
        yaw_degs[bid] = float(yaw)

    samples = scatter_blocks(
        dr.model,
        dr.data,
        dr.rng,
        body_ids,
        initial_quats=initial_quats,
        surface_top_z=table_top,
        center=center[:2],
        square_half_width=square_half_width,
        drop_height=drop_height,
        drop_seconds=drop_seconds,
        hide_offset=hide_offset,
    )
    for sample in samples:
        sample["yaw_deg"] = yaw_degs[int(sample["body_id"])]

    dr._log_sample(
        action,
        {
            "surface_geom": surface_geom,
            "table_top": table_top,
            "center": center.tolist(),
            "square_half_width": square_half_width,
            "drop_height": drop_height,
            "drop_seconds": drop_seconds,
            "body_names": dr._body_names(body_ids),
            "blocks": [
                {key: value for key, value in sample.items() if key not in {"body_id", "quat", "half_z"}}
                for sample in samples
            ],
        },
    )
