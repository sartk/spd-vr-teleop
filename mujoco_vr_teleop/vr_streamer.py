"""Lightweight VR teleop: server-side MuJoCo + Three.js WebXR streaming.

Runs MuJoCo physics on the server, exports body meshes as GLB at startup,
and streams only body transforms (position + quaternion) to a minimal
Three.js VR app running on the headset. No WASM, no heavy client-side
computation.

Builds a MuJoCo scene from the scene-builder config by default.

Usage:
    # Start the VR streaming server
    mujoco-vr-stream --builder.arm-type yam_ultra --builder.end-effector sharpa

    # Open in VR headset browser: http://<your-ip>:8012
"""

from __future__ import annotations

import asyncio
import copy
import json
import multiprocessing as mp
import os
import struct
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("MUJOCO_VR_VIDEO_WORKER") != "1":
    os.environ.pop("MUJOCO_GL", None)

import attrs
import mujoco
import numpy as np
import tyro
import trimesh
import trimesh.visual

from aiohttp import web
from scipy.spatial.transform import Rotation

from mujoco_vr_teleop.domain_randomization import DomainRandomizer
from mujoco_vr_teleop.native_stepper import NativeStepper
from mujoco_vr_teleop.scene_builder import (
    BuildSceneConfig,
    SHARPA_WRIST_MOCAP_SITE_PAIRS,
    build_scene,
)
from mujoco_vr_teleop.tasks import get_task, scale_log, tasks_for_scene


HAND_JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)

HAND_IK_SITE_BY_JOINT = {
    "wrist": "{prefix}sharpa_palm_site",
    "thumb-phalanx-proximal": "{prefix}sharpa_thumb_proximal_site",
    "thumb-phalanx-distal": "{prefix}sharpa_thumb_distal_site",
    "thumb-tip": "{prefix}sharpa_thumb_tip_site",
    "index-finger-phalanx-proximal": "{prefix}sharpa_index_proximal_site",
    "index-finger-phalanx-intermediate": "{prefix}sharpa_index_intermediate_site",
    "index-finger-phalanx-distal": "{prefix}sharpa_index_distal_site",
    "index-finger-tip": "{prefix}sharpa_index_tip_site",
    "middle-finger-phalanx-proximal": "{prefix}sharpa_middle_proximal_site",
    "middle-finger-phalanx-intermediate": "{prefix}sharpa_middle_intermediate_site",
    "middle-finger-phalanx-distal": "{prefix}sharpa_middle_distal_site",
    "middle-finger-tip": "{prefix}sharpa_middle_tip_site",
    "ring-finger-phalanx-proximal": "{prefix}sharpa_ring_proximal_site",
    "ring-finger-phalanx-intermediate": "{prefix}sharpa_ring_intermediate_site",
    "ring-finger-phalanx-distal": "{prefix}sharpa_ring_distal_site",
    "ring-finger-tip": "{prefix}sharpa_ring_tip_site",
    "pinky-finger-phalanx-proximal": "{prefix}sharpa_pinky_proximal_site",
    "pinky-finger-phalanx-intermediate": "{prefix}sharpa_pinky_intermediate_site",
    "pinky-finger-phalanx-distal": "{prefix}sharpa_pinky_distal_site",
    "pinky-finger-tip": "{prefix}sharpa_pinky_tip_site",
}

HAND_IK_PALM_POS_WEIGHT = 50.0
HAND_IK_FINGER_POS_WEIGHT = 20.0
HAND_IK_FINGER_ORI_WEIGHT = 0.25
ARM_DIAGNOSTIC_SIDES = ("right", "left")
ARM_DIAGNOSTIC_JOINTS = 6
HAND_DIAGNOSTIC_JOINTS = 22


@attrs.define(frozen=True)
class VRStreamerConfig:
    builder: BuildSceneConfig
    domain_randomization: bool = True
    port: int = 8012
    control_rate: float = 60.0
    physics_rate: float = 480.0
    stream_rate: float = 60.0
    record_rate: float | None = None
    collision_mode: bool = False
    hand_skeleton: bool = False
    ik_mode: bool = False
    ik_workers_per_arm: int = 2
    wrist_hybrid_label_horizon: int = 4
    weld_enter_thresh: float = 0.03
    ik_smoothing: float = 0.8
    ik_ori_weight: float = 5.0
    mink_solver: str = "daqp"
    mink_iterations: int = 4
    mink_gain: float = 0.20
    mink_max_velocity: float = 3.0
    mink_max_joint_delta: float = 0.12
    mink_limit_margin: float = 0.1
    mink_limit_gain: float = 0.5
    mink_posture_cost: float = 5e-3
    joint_ik_xml_right: str | None = None
    joint_ik_xml_left: str | None = None
    scene_pos: tuple[float, float, float] = (0.0, 0.2, 0.2)
    vr_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vr_target: tuple[float, float, float] = (0.0, 0.75, 0.0)
    output_dir: str = "data"
    session_name: str = "default"
    record_delay: float = 0.5
    arm_opacity: float = 0.1
    # choreographer: tasks iterated in order; A captures a flat-numbered
    #   snapshot of the current scene state. Used by the operator to author
    #   the illustration images that collectors see.
    # collector: tasks randomized; A saves progress, B pauses/resumes,
    #   C reverts/skips (a revert or skip records a failure since the last A).
    # admin: legacy admin recording mode (full instructions shown, A press
    #   saves a checkpoint only).
    mode: str = "collector"


def _apply_streamer_defaults(args: VRStreamerConfig, defaults: dict | None) -> VRStreamerConfig:
    if not defaults:
        return args
    values = copy.deepcopy(defaults)
    fields = attrs.fields_dict(VRStreamerConfig)
    known_values = {key: value for key, value in values.items() if key in fields}
    return attrs.evolve(args, **known_values)


def _set_process_affinity(cpu_ids: list[int] | None, label: str) -> list[int] | None:
    if not cpu_ids or not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        os.sched_setaffinity(0, set(int(cpu) for cpu in cpu_ids))
        active = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        print(f"{label} CPU affinity: {active}")
        return active
    except Exception as exc:
        print(f"{label} CPU affinity disabled: {exc}")
        return None


def _make_cpu_affinity_planner(enabled: bool):
    if not enabled or not hasattr(os, "sched_getaffinity"):
        return None, None

    try:
        available_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception:
        return None, None

    if len(available_cpus) <= 1:
        return None, None

    main_core_count = 2 if len(available_cpus) >= 4 else 1
    main_cpu_ids = available_cpus[:main_core_count]
    worker_pool = available_cpus[main_core_count:] or available_cpus[main_core_count - 1:] or available_cpus
    next_worker_idx = 0

    def allocate_worker_cpu_ids() -> list[int]:
        nonlocal next_worker_idx
        cpu_ids = [worker_pool[next_worker_idx % len(worker_pool)]]
        next_worker_idx += 1
        return cpu_ids

    return main_cpu_ids, allocate_worker_cpu_ids


def _mink_solver_kwargs(args: VRStreamerConfig) -> dict:
    """Mink tunables shared by both arm-IK and joint-IK configs."""
    return {
        "mink_solver": args.mink_solver,
        "mink_dt": 1.0 / max(float(args.control_rate), 1e-6),
        "mink_iterations": int(args.mink_iterations),
        "mink_gain": float(args.mink_gain),
        "mink_max_velocity": float(args.mink_max_velocity),
        "mink_limit_gain": float(args.mink_limit_gain),
        "mink_limit_margin": float(args.mink_limit_margin),
    }


def _spawn_worker_pool(
    mp_ctx,
    target,
    *,
    count: int,
    name_prefix: str,
    proc_args,
    allocate_worker_cpu_ids,
) -> tuple[list[dict], dict]:
    """Spawn `count` worker processes, do the info handshake, return (workers, info).

    `proc_args(child_conn, cpu_ids)` builds the positional args tuple for the worker target.
    Each worker dict contains proc/conn plus seq/send_ts bookkeeping.
    """
    workers: list[dict] = []
    worker_info: dict | None = None
    for worker_idx in range(count):
        parent_conn, child_conn = mp_ctx.Pipe()
        cpu_ids = allocate_worker_cpu_ids() if allocate_worker_cpu_ids is not None else None
        proc = mp_ctx.Process(
            target=target,
            args=proc_args(child_conn, cpu_ids),
            daemon=True,
            name=f"{name_prefix}-{worker_idx}",
        )
        proc.start()
        child_conn.close()
        parent_conn.send({"cmd": "info"})
        info = parent_conn.recv()
        if worker_info is None:
            worker_info = info
        workers.append(
            {
                "proc": proc,
                "conn": parent_conn,
                "in_flight": False,
                "seq": -1,
                "send_ts": 0.0,
            }
        )
    assert worker_info is not None
    return workers, worker_info



def _joint_ik_worker_main(cfg_dict: dict, conn, cpu_ids: list[int] | None = None) -> None:
    _set_process_affinity(cpu_ids, "Joint IK worker")
    from mujoco_vr_teleop.ik_controller import JointIKConfig, JointIKSolver

    home_joints = cfg_dict.get("home_joints")
    cfg = JointIKConfig(
        xml_path=Path(cfg_dict["xml_path"]),
        controlled_joint_names=list(cfg_dict["controlled_joint_names"]),
        target_frame_names=list(cfg_dict["target_frame_names"]),
        backend=cfg_dict.get("backend", "mink"),
        home_joints=None if home_joints is None else np.asarray(home_joints, dtype=float),
        qpos0=None if cfg_dict.get("qpos0") is None else np.asarray(cfg_dict["qpos0"], dtype=float),
        smoothing_alpha=float(cfg_dict["smoothing_alpha"]),
        mink_solver=cfg_dict.get("mink_solver", "daqp"),
        mink_dt=float(cfg_dict.get("mink_dt", 1.0 / 30.0)),
        mink_iterations=int(cfg_dict.get("mink_iterations", 4)),
        mink_gain=float(cfg_dict.get("mink_gain", 0.20)),
        mink_max_velocity=float(cfg_dict.get("mink_max_velocity", 3.0)),
        mink_limit_gain=float(cfg_dict.get("mink_limit_gain", 0.5)),
        mink_limit_margin=float(cfg_dict.get("mink_limit_margin", 0.1)),
        max_joint_delta=float(cfg_dict.get("max_joint_delta", 0.12)),
    )
    solver = JointIKSolver(cfg)
    solver_info = solver.info()

    try:
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")

            if cmd == "close":
                return

            if cmd == "info":
                conn.send(solver_info)
                continue

            if cmd == "reset":
                home_joints = msg.get("home_joints")
                body_positions = msg.get("body_positions")
                if isinstance(body_positions, dict):
                    solver.set_body_positions(
                        {
                            str(name): np.asarray(pos, dtype=float)
                            for name, pos in body_positions.items()
                        }
                    )
                solver.reset(
                    None if home_joints is None else np.asarray(home_joints, dtype=float)
                )
                conn.send({"ok": True})
                continue

            if cmd != "solve":
                raise RuntimeError(f"Unknown joint IK command: {cmd}")

            solve_t0 = time.perf_counter()
            result = solver.solve(
                targets=list(msg.get("targets", [])),
                seed_q=np.asarray(msg["seed_q"], dtype=float),
            )
            solve_ms = (time.perf_counter() - solve_t0) * 1000.0
            conn.send({"joint_positions": result.joint_positions, "solve_ms": solve_ms})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

def _get_geom_rgba(model, geom_id):
    matid = model.geom_matid[geom_id]
    if matid >= 0:
        return model.mat_rgba[matid].copy()
    rgba = model.geom_rgba[geom_id].copy()
    if rgba.sum() == 0:
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba


def _create_primitive_mesh(model, geom_id):
    """Convert a MuJoCo primitive geom to trimesh with texture support."""
    from PIL import Image

    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    rgba = _get_geom_rgba(model, geom_id)
    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
        plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
        mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
    else:
        return None

    # Check if this geom has a textured material
    matid = model.geom_matid[geom_id]
    has_texture = False
    if matid >= 0 and matid < model.nmat:
        texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        if texid < 0:
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
        if texid >= 0 and texid < model.ntex:
            has_texture = True
            mat_rgba = model.mat_rgba[matid]
            tex_w = model.tex_width[texid]
            tex_h = model.tex_height[texid]
            tex_nc = model.tex_nchannel[texid]
            tex_adr = model.tex_adr[texid]
            tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]
            texrepeat = model.mat_texrepeat[matid]

            if tex_nc == 3:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 3).astype(np.uint8)), mode="RGB")
            elif tex_nc == 4:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 4).astype(np.uint8)), mode="RGBA")
            elif tex_nc == 1:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w).astype(np.uint8)), mode="L")
            else:
                has_texture = False

    if has_texture:
        verts = mesh.vertices
        uv = np.zeros((len(verts), 2))
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_range = bb_max - bb_min
        bb_range[bb_range == 0] = 1
        uv[:, 0] = (verts[:, 0] - bb_min[0]) / bb_range[0] * texrepeat[0]
        uv[:, 1] = (verts[:, 1] - bb_min[1]) / bb_range[1] * texrepeat[1]
        # Force white baseColorFactor when a texture is bound. MuJoCo's
        # native renderer treats material rgba as "tint when no texture" —
        # textured geoms render at the texture's own color. The glTF spec
        # multiplies baseColorTexture × baseColorFactor, so leaving rgba
        # in (which can be e.g. dark red like 0.8 0.12 0.12) darkens the
        # texture. Preserve the alpha channel so transparent materials
        # still work.
        tex_base = [1.0, 1.0, 1.0, float(mat_rgba[3])]
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=tex_base,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    else:
        vertex_colors = np.tile(rgba_uint8, (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)

    return mesh


def _mujoco_mesh_to_trimesh(model, geom_id):
    """Convert a MuJoCo mesh geom to trimesh with texture support."""
    from PIL import Image

    mesh_id = model.geom_dataid[geom_id]
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start:vert_start + vert_count]
    faces = model.mesh_face[face_start:face_start + face_count]

    texcoord_adr = model.mesh_texcoordadr[mesh_id]
    texcoord_num = model.mesh_texcoordnum[mesh_id]

    if texcoord_num > 0:
        texcoords = model.mesh_texcoord[texcoord_adr:texcoord_adr + texcoord_num]
        face_texcoord_idx = model.mesh_facetexcoord[face_start:face_start + face_count]

        # Expand vertices/UVs per-face for independent texcoord indexing
        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        matid = model.geom_matid[geom_id]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            if texid < 0:
                texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])

            if texid >= 0 and texid < model.ntex:
                tex_w = model.tex_width[texid]
                tex_h = model.tex_height[texid]
                tex_nc = model.tex_nchannel[texid]
                tex_adr = model.tex_adr[texid]
                tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]

                image = None
                if tex_nc == 1:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w)).astype(np.uint8), mode="L")
                elif tex_nc == 3:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 3)).astype(np.uint8), mode="RGB")
                elif tex_nc == 4:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 4)).astype(np.uint8), mode="RGBA")

                if image is not None:
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=rgba,
                        baseColorTexture=image,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
                else:
                    rgba_255 = (rgba * 255).astype(np.uint8)
                    mesh.visual = trimesh.visual.ColorVisuals(
                        vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
            else:
                rgba_255 = (rgba * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
        else:
            rgba = _get_geom_rgba(model, geom_id)
            rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = _get_geom_rgba(model, geom_id)
        rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1)))

    return mesh


def _patch_glb_add_material(glb_path: str):
    """Patch GLB: add default PBR material for primitives that lack one.

    trimesh exports COLOR_0 vertex colors but no material, which causes
    Three.js GLTFLoader to ignore the vertex colors. Textured meshes
    already have materials and are left untouched.
    """
    import json as _json

    with open(glb_path, "rb") as f:
        header = f.read(12)
        chunk0_len = struct.unpack("<I", f.read(4))[0]
        chunk0_type = f.read(4)
        json_bytes = f.read(chunk0_len)
        rest = f.read()  # binary chunk

    gltf = _json.loads(json_bytes)

    # Force doubleSided on every material so inner-liner faces with
    # inward-pointing normals (e.g. cup interior) still render.
    if "materials" not in gltf:
        gltf["materials"] = []
    for mat in gltf["materials"]:
        mat["doubleSided"] = True

    # Check if any primitives need a default material
    needs_patch = False
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" not in prim or prim["material"] is None:
                needs_patch = True
                break

    if needs_patch:
        default_idx = len(gltf["materials"])
        gltf["materials"].append({
            "pbrMetallicRoughness": {
                "baseColorFactor": [1, 1, 1, 1],
                "metallicFactor": 0.1,
                "roughnessFactor": 0.7,
            },
            "doubleSided": True,
        })
        for mesh in gltf.get("meshes", []):
            for prim in mesh.get("primitives", []):
                if "material" not in prim or prim["material"] is None:
                    prim["material"] = default_idx

    # Re-encode
    new_json = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_json) % 4 != 0:
        new_json += b" "

    with open(glb_path, "wb") as f:
        total = 12 + 8 + len(new_json) + len(rest)
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<I", len(new_json)))
        f.write(b"JSON")
        f.write(new_json)
        f.write(rest)


def export_body_glbs(model, output_dir: Path, collision_mode: bool = False):
    """Export each body's merged mesh as a GLB file. Returns body info dict."""

    output_dir.mkdir(parents=True, exist_ok=True)
    visible_groups = {0, 1, 2}
    body_geoms: dict[int, list[int]] = {}

    def is_collision_geom(geom_id: int) -> bool:
        return int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0

    for body_id in range(model.nbody):
        collision_only_geom_ids: list[int] = []
        visible_geom_ids: list[int] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            in_visible_group = int(model.geom_group[geom_id]) in visible_groups
            # Skip geoms explicitly authored as invisible (alpha=0 via
            # geom_rgba; ignored when the geom has a material since the
            # material's color overrides). These are typically "region"
            # marker geoms (bounding boxes, liquid volumes, interior
            # cylinders) that show up as gray garbage when rendered.
            geom_alpha_zero = (
                int(model.geom_matid[geom_id]) < 0
                and float(model.geom_rgba[geom_id][3]) == 0.0
            )
            if in_visible_group and not geom_alpha_zero:
                # Anything in groups 0/1/2 is meant to be seen, even if it
                # also has contype/conaffinity for collisions (e.g. walls).
                visible_geom_ids.append(geom_id)
            elif is_collision_geom(geom_id):
                collision_only_geom_ids.append(geom_id)

        if collision_mode and collision_only_geom_ids:
            body_geoms[body_id] = collision_only_geom_ids
        elif visible_geom_ids:
            body_geoms[body_id] = visible_geom_ids
        elif collision_only_geom_ids:
            body_geoms[body_id] = collision_only_geom_ids

    bodies = {}
    for body_id, geom_ids in body_geoms.items():
        meshes = []
        for gid in geom_ids:
            geom_type = model.geom_type[gid]
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = _mujoco_mesh_to_trimesh(model, gid)
            else:
                mesh = _create_primitive_mesh(model, gid)
            if mesh is None:
                continue

            # Geom-local transform
            qw = model.geom_quat[gid]
            rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = model.geom_pos[gid]
            mesh.apply_transform(T)

            # Convert Z-up to Y-up for Three.js: (x,y,z) -> (x,z,-y)
            verts = np.array(mesh.vertices)
            new_verts = np.empty_like(verts)
            new_verts[:, 0] = verts[:, 0]
            new_verts[:, 1] = verts[:, 2]
            new_verts[:, 2] = -verts[:, 1]
            mesh.vertices = new_verts

            meshes.append(mesh)

        if not meshes:
            continue

        is_fixed = (model.body_weldid[body_id] == 0
                    and model.body_mocapid[model.body_rootid[body_id]] < 0)

        glb_path = output_dir / f"body_{body_id}.glb"

        # Export as Scene to preserve multiple materials (textured + colored)
        has_texture = any(isinstance(m.visual, trimesh.visual.TextureVisuals) for m in meshes)
        if has_texture or len(meshes) > 1:
            scene = trimesh.Scene(meshes)
            scene.export(str(glb_path), file_type="glb")
        else:
            meshes[0].export(str(glb_path), file_type="glb")

        # Patch GLB: add material for vertex-colored meshes (trimesh omits it)
        _patch_glb_add_material(str(glb_path))

        bodies[body_id] = {
            "file": f"body_{body_id}.glb",
            "is_fixed": is_fixed,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(body_id),
        }

    return bodies


# Scenes whose variant pools randomize the *count* of bodies per build (not
# just which variant each slot gets). For these, a per-task reset needs a
# full rebuild_scene() to actually re-roll the count.
#
# cup_stack / cup_ball are included because their build is fully decided at
# compile time — cup family, count, ball presence, the single shared cup id,
# and the cup scale are all chosen in pools_for(); a reset that does not
# rebuild keeps the same cup variant.
REBUILD_ON_RESET_SCENES: set[str] = {
    "hang_mugs", "bottles_in_bin", "cup_stack", "cup_ball",
}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def main():
    args = tyro.cli(VRStreamerConfig)
    # Identifies this server process; the frontend polls this and reloads all
    # assets if it changes (i.e. a new server is running).

    # Tee stdout / stderr through the terminal AND into <session_dir>/server.log
    # / .stderr.log once the session directory exists. The supervisor TUI can
    # then tail those files for its log pane.
    from mujoco_vr_teleop.tui import install_log_tee

    stdout_tee, stderr_tee = install_log_tee()
    # Attach a boot log right away so traces from choreographer-mode sessions
    # that never save an episode (and therefore never create a session_dir)
    # still hit disk. Once the recorder creates the real session_dir we'll
    # also attach the per-session server.log on top.
    from datetime import datetime as _dt
    _boot_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _boot_dir = Path(args.output_dir) / args.session_name / "_boot_logs"
    _boot_dir.mkdir(parents=True, exist_ok=True)
    _boot_log = _boot_dir / f"server_{_boot_ts}.log"
    _boot_errlog = _boot_dir / f"server_{_boot_ts}.stderr.log"
    try:
        stdout_tee.set_log_file(_boot_log)
        stderr_tee.set_log_file(_boot_errlog)
        print(f"Boot log: {_boot_log}")
    except Exception as _exc:
        print(f"Failed to attach boot log: {_exc}")
    if args.ik_workers_per_arm < 1:
        raise SystemExit("--ik-workers-per-arm must be >= 1")
    repo_root = Path(__file__).resolve().parents[1]
    # The streamer always builds the main VR scene with safe joint ranges; pre-baked
    # joint-IK scenes still use the wide ranges from their own BuildSceneConfig.
    builder_config = attrs.evolve(args.builder, safe_joint_ranges=True)
    # The cup scenes are task-driven: the build's cup count / ball depend on
    # the active task. The cup task manager (constructed below) drives this
    # thereafter, but the FIRST build runs before it exists — pin the task it
    # will start on so the initial build matches.
    if builder_config.scene_type in ("cup_stack", "cup_ball"):
        from mujoco_vr_teleop.cups_task_manager import (
            CUP_STACK_TASK_IDS, CUP_BALL_TASK_IDS)
        from mujoco_vr_teleop import variant_pools as _cups_vp
        _first = (CUP_STACK_TASK_IDS if builder_config.scene_type == "cup_stack"
                  else CUP_BALL_TASK_IDS)[0]
        _cups_vp.set_cup_task(builder_config.scene_type,
                              _first.split("/", 1)[1])
    generated_scene = build_scene(builder_config)
    xml_path = generated_scene.xml_path.resolve()
    scene_streamer_config = generated_scene.streamer_config
    args = _apply_streamer_defaults(args, scene_streamer_config)
    wrist_hybrid_weld_mode = args.builder.wrist_control == "hybrid_weld"
    wrist_weld_scene_mode = wrist_hybrid_weld_mode
    if args.wrist_hybrid_label_horizon < 1:
        raise SystemExit("--wrist-hybrid-label-horizon must be >= 1")
    # Mutable container so reset_scene() can swap model/data for a true
    # variant-rebuild without every inner closure capturing stale references.
    # Inner functions that need to track post-rebuild state should read
    # state.model / state.data instead of model / data. Migration is incremental.
    state = SimpleNamespace()
    state.session_id = uuid.uuid4().hex
    state.model = generated_scene.model
    state.model.opt.timestep = 1.0 / float(args.physics_rate)
    hand_scale = float(args.builder.hand_scale)

    def resolve_config_path(value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        for base in (repo_root, xml_path.parent, Path.cwd()):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (repo_root / path).resolve()

    state.data = mujoco.MjData(state.model)

    def align_wrist_mocaps_to_palms() -> None:
        if not wrist_weld_scene_mode:
            return
        mujoco.mj_forward(state.model, state.data)
        for mocap_name, site_name in SHARPA_WRIST_MOCAP_SITE_PAIRS:
            bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
            sid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if bid < 0 or sid < 0:
                continue
            mid = int(state.model.body_mocapid[bid])
            if mid < 0:
                continue
            quat = np.zeros(4, dtype=float)
            mujoco.mju_mat2Quat(quat, state.data.site_xmat[sid].reshape(9))
            state.data.mocap_pos[mid] = state.data.site_xpos[sid]
            state.data.mocap_quat[mid] = quat
        mujoco.mj_forward(state.model, state.data)

    from mujoco_vr_teleop.recorder import save_keyframe_file

    if state.model.nkey > 0:
        mujoco.mj_resetDataKeyframe(state.model, state.data, 0)
    mujoco.mj_forward(state.model, state.data)
    state.data.ctrl[:] = 0.0
    align_wrist_mocaps_to_palms()
    mujoco.mj_forward(state.model, state.data)
    domain_randomization_config = (
        scene_streamer_config.get("domain_randomization")
        if args.domain_randomization
        else {"enabled": False, "settle": False}
    )
    state.domain_randomizer = DomainRandomizer.create(
        state.model,
        state.data,
        domain_randomization_config,
    )

    keyframe_file_path = xml_path.with_suffix(".keyframe.json")

    print(f"Scene: {xml_path.name}")
    print(f"Model: nbody={state.model.nbody}, ngeom={state.model.ngeom}, nu={state.model.nu}, nkey={state.model.nkey}")

    # Export meshes (clean dir to avoid stale files from other scenes)
    import shutil
    mesh_dir = Path("/tmp/mujoco_vr_meshes")
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    state.body_info = export_body_glbs(state.model, mesh_dir, collision_mode=bool(args.collision_mode))
    # Playground boots into a deterministic, defined state — the scene's
    # `<scene>/playground` task reset (jenga = the clean assembled tower) —
    # NOT a domain-randomized scene. Domain randomization only runs as part of
    # a real task reset, never at boot. If a scene has no playground task, the
    # raw XML state is used as-is.
    _playground_spec = get_task(f"{builder_config.scene_type}/playground")
    if _playground_spec is not None:
        _playground_spec.reset(state.model, state.data, np.random.default_rng())
        print(f"Playground state set via {_playground_spec.id}")
    align_wrist_mocaps_to_palms()
    state.domain_randomizer.settle()
    R_conv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    def get_body_pose_yup(body_id: int):
        xpos = state.data.xpos[body_id] + state.domain_randomizer.body_visual_pos_offset[body_id]
        xmat = state.data.xmat[body_id].reshape(3, 3)
        pos_yup = R_conv @ xpos
        det = np.linalg.det(xmat)
        if abs(det) > 1e-6:
            mat_yup = R_conv @ xmat @ R_conv.T
            quat_yup = Rotation.from_matrix(mat_yup).as_quat()  # xyzw
        else:
            quat_yup = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return pos_yup, quat_yup

    def get_body_scale(body_id: int) -> list[float]:
        return np.asarray(state.domain_randomizer.body_scale[body_id], dtype=float).reshape(3).tolist()

    def refresh_body_info_poses() -> None:
        for bid, info in state.body_info.items():
            pos_yup, quat_yup = get_body_pose_yup(bid)
            info["position"] = pos_yup.tolist()
            info["quaternion"] = quat_yup.tolist()
            info["scale"] = get_body_scale(bid)

    refresh_body_info_poses()

    arm_opacity = float(args.arm_opacity)
    for info in state.body_info.values():
        name = info["name"].lower()
        if "sharpa" not in name and ("right-arm" in name or "left-arm" in name):
            info["opacity"] = arm_opacity

    fixed_ids = [bid for bid, info in state.body_info.items() if info["is_fixed"]]
    dynamic_ids = [bid for bid, info in state.body_info.items() if not info["is_fixed"]]
    state.stream_body_ids = sorted(set(dynamic_ids) | state.domain_randomizer.stream_body_ids)
    print(
        f"Exported {len(state.body_info)} body meshes ({len(fixed_ids)} fixed, {len(dynamic_ids)} dynamic, "
        f"{len(state.stream_body_ids)} streamed, collision_mode={bool(args.collision_mode)})"
    )

    # Geom IDs on any Sharpa hand body (palm + fingers, both hands). Used to
    # reject save_checkpoint while a hand is in contact with an external object.
    # Sharpa finger bodies are named like 'left_index_DP', 'right_thumb_MC'; the
    # wrist mounts are 'sharpa_hand_left/right'.
    _hand_tokens = ("thumb", "index", "middle", "ring", "pinky", "palm")
    state.hand_geom_ids: set[int] = set()
    for gid in range(state.model.ngeom):
        bid = int(state.model.geom_bodyid[gid])
        bname = (mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
        is_sharpa_mount = "sharpa" in bname
        is_finger = (
            (bname.startswith("left_") or bname.startswith("right_"))
            and any(token in bname for token in _hand_tokens)
        )
        if is_sharpa_mount or is_finger:
            state.hand_geom_ids.add(gid)
    print(f"Checkpoint-block: tracking {len(state.hand_geom_ids)} hand geoms for contact detection")

    def hand_in_contact_with_object() -> bool:
        ncon = int(state.data.ncon)
        hand_external = []
        for ci in range(ncon):
            contact = state.data.contact[ci]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            in1 = g1 in state.hand_geom_ids
            in2 = g2 in state.hand_geom_ids
            if in1 != in2:
                n1 = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
                n2 = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
                hand_external.append((n1, n2))
        print(
            f"Checkpoint contact scan: ncon={ncon} hand_geoms={len(state.hand_geom_ids)} "
            f"hand-vs-other={len(hand_external)}"
        )
        for n1, n2 in hand_external[:5]:
            print(f"  {n1} <-> {n2}")
        return bool(hand_external)

    # Resolve table geom once; pose/size are re-read each time DR moves it.
    state.table_gid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    if state.table_gid < 0:
        state.table_gid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
    if state.table_gid < 0:
        raise SystemExit(
            "Workspace clamp requires a 'table_visual' or 'table_plane' geom in the scene."
        )

    # Per-hand workspace cuboid (z-up MuJoCo coords): (lo, hi) tuples.
    state.workspace_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def recompute_workspace_bounds() -> None:
        mujoco.mj_forward(state.model, state.data)
        center = state.data.geom_xpos[state.table_gid]
        size = state.model.geom_size[state.table_gid]
        x_buffer = 0.05  # back of table (toward robot)
        x_front_buffer = 0.10  # extend past front edge so user can reach in
        x_lo = float(center[0] - size[0] - x_buffer)
        x_hi = float(center[0] + size[0] + x_front_buffer)
        y_max = float(size[1])  # table half-width in y
        z_top = float(center[2] + size[2])
        z_lo = z_top - 0.02
        z_hi = z_top + 0.6
        # +y is left, -y is right (robots are at y=+0.31 / -0.31).
        # Right-hand cuboid covers y ∈ [-y_max, +y_max/3]; left mirrored.
        state.workspace_bounds["right"] = (
            np.array([x_lo, -y_max, z_lo], dtype=float),
            np.array([x_hi, +y_max / 3.0, z_hi], dtype=float),
        )
        state.workspace_bounds["left"] = (
            np.array([x_lo, -y_max / 3.0, z_lo], dtype=float),
            np.array([x_hi, +y_max, z_hi], dtype=float),
        )

    recompute_workspace_bounds()

    native_stepper = NativeStepper()

    # Save initial qpos as state reference
    state.init_qpos = state.data.qpos.copy()

    # Gather mocap body info (for interactive control in the viewer)
    state.mocap_info = []
    state.mocap_id_to_name = {}
    state.hand_mocap_mapping = {"left": {}, "right": {}}
    for i in range(state.model.nbody):
        mid = state.model.body_mocapid[i]
        if mid >= 0:
            pos_yup = R_conv @ state.data.mocap_pos[mid]
            mj_quat = state.data.mocap_quat[mid]  # wxyz
            mat_zup = Rotation.from_quat([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]]).as_matrix()
            q_yup = Rotation.from_matrix(R_conv @ mat_zup @ R_conv.T).as_quat()  # xyzw
            name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"mocap_{mid}"
            state.mocap_id_to_name[int(mid)] = name
            for side in ("left", "right"):
                prefix = f"{side}-"
                if name.startswith(prefix):
                    joint_name = name[len(prefix):]
                    if joint_name in HAND_JOINT_NAMES:
                        state.hand_mocap_mapping[side][joint_name] = int(mid)
            state.mocap_info.append({
                "mocap_id": int(mid),
                "body_id": int(i),
                "name": name,
                "position": pos_yup.tolist(),
                "quaternion": q_yup.tolist(),  # xyzw
            })
            print(f"Mocap target: {name} (mocap_id={mid}, body_id={i})")
    # Pending mocap/trigger updates from the web client
    pending_mocap = []
    pending_triggers = []

    def find_actuator_for_joint(joint_name):
        jid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return -1
        for i in range(state.model.nu):
            if state.model.actuator_trnid[i][0] == jid:
                return i
        return -1

    def find_yam_arm_actuator_ids(side: str) -> list[int]:
        prefix = f"{side}-arm-"
        ids = [find_actuator_for_joint(f"{prefix}joint{i}") for i in range(1, 7)]
        return ids if all(i >= 0 for i in ids) else []

    state.leader_arm_actuator_ids = {
        "left": find_yam_arm_actuator_ids("left"),
        "right": find_yam_arm_actuator_ids("right"),
    }
    state.arm_actuator_gains = {
        aid: (state.model.actuator_gainprm[aid].copy(), state.model.actuator_biasprm[aid].copy())
        for act_ids in state.leader_arm_actuator_ids.values()
        for aid in act_ids
    }

    def set_arm_actuators_enabled(side: str, enabled: bool) -> None:
        for aid in state.leader_arm_actuator_ids.get(side, []):
            gain, bias = state.arm_actuator_gains[aid]
            if enabled:
                state.model.actuator_gainprm[aid, :] = gain
                state.model.actuator_biasprm[aid, :] = bias
            else:
                state.model.actuator_gainprm[aid, :] = 0.0
                state.model.actuator_biasprm[aid, :] = 0.0

    state.wrist_weld_eq_ids: dict[str, int] = {}
    state.wrist_mocap_ids: dict[str, int] = {}
    state.wrist_palm_site_ids: dict[str, int] = {}
    for mocap_name, site_name in SHARPA_WRIST_MOCAP_SITE_PAIRS:
        side = mocap_name.split("-")[0]
        mocap_site_name = f"{mocap_name}-site-mocap"
        mocap_site_id = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_SITE, mocap_site_name)
        palm_site_id = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        body_id = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        if body_id >= 0 and int(state.model.body_mocapid[body_id]) >= 0:
            state.wrist_mocap_ids[side] = int(state.model.body_mocapid[body_id])
        if palm_site_id >= 0:
            state.wrist_palm_site_ids[side] = palm_site_id
        for eq_id in range(state.model.neq):
            if state.model.eq_type[eq_id] != mujoco.mjtEq.mjEQ_WELD:
                continue
            if state.model.eq_objtype[eq_id] != mujoco.mjtObj.mjOBJ_SITE:
                continue
            site1 = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_SITE, state.model.eq_obj1id[eq_id])
            site2 = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_SITE, state.model.eq_obj2id[eq_id])
            if {site1, site2} == {mocap_site_name, site_name}:
                state.wrist_weld_eq_ids[side] = eq_id
                break

    state.wrist_hybrid_weld_active = {side: False for side in ARM_DIAGNOSTIC_SIDES}
    state.wrist_hybrid_error_pos = {side: np.nan for side in ARM_DIAGNOSTIC_SIDES}
    if wrist_hybrid_weld_mode:
        for eq_id in state.wrist_weld_eq_ids.values():
            state.model.eq_active0[eq_id] = 0
            state.data.eq_active[eq_id] = 0
        print(f"Hybrid wrist weld mode: found weld constraints for {sorted(state.wrist_weld_eq_ids)}")

    # Figure out which actuators are grippers (finger/gripper actuators)
    # For the YAM: actuators 6 (left_gripper) and 13 (right_gripper)
    # For pick_place: actuator 0 (gripper-fingers_actuator)
    # Heuristic: find actuators with "grip" or "finger" in the name
    state.gripper_actuator_ids = {"left": None, "right": None}
    for i in range(state.model.nu):
        aname = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        aname_lower = aname.lower()
        if "grip" in aname_lower or "finger" in aname_lower:
            ctrl_range = state.model.actuator_ctrlrange[i]
            if "left" in aname_lower:
                state.gripper_actuator_ids["left"] = (i, float(ctrl_range[0]), float(ctrl_range[1]))
            elif "right" in aname_lower:
                state.gripper_actuator_ids["right"] = (i, float(ctrl_range[0]), float(ctrl_range[1]))
            elif state.gripper_actuator_ids["right"] is None:
                # Single gripper — assign to right controller
                state.gripper_actuator_ids["right"] = (i, float(ctrl_range[0]), float(ctrl_range[1]))
    for side, info in state.gripper_actuator_ids.items():
        if info:
            print(f"Gripper actuator ({side}): id={info[0]}, range=[{info[1]:.3f}, {info[2]:.3f}]")

    # --- Recording session ---
    from mujoco_vr_teleop.recorder import Session, task_allowed_start_modes

    session = Session(
        output_dir=args.output_dir,
        session_name=args.session_name,
        command=str(args),
        xml_path=xml_path,
        streamer_config=attrs.asdict(args),
    )

    task_manager = None
    # Tasks come from the tasks/ registry: TaskRegistryManager iterates them
    # sequentially for choreographer, randomized for collector.
    from mujoco_vr_teleop.recorder import TaskRegistryManager
    from mujoco_vr_teleop.dishrack_task_manager import DishrackTaskManager
    from mujoco_vr_teleop.cups_task_manager import (
        CupStackTaskManager, CupBallTaskManager)

    if tasks_for_scene(builder_config.scene_type):
        ordering = "sequential" if args.mode == "choreographer" else "random"
        # Dishrack scene uses a custom 2-task forced cycle (rack→plate) that
        # suppresses the post-task modal. See dishrack_task_manager.py.
        if builder_config.scene_type == "dishrack":
            task_manager = DishrackTaskManager(builder_config.scene_type,
                                                 ordering=ordering)
        # cup_stack is task-driven (cup count varies per task) with one forced
        # edge (stack→unstack). See cups_task_manager.py.
        elif builder_config.scene_type == "cup_stack":
            task_manager = CupStackTaskManager(builder_config.scene_type,
                                                ordering=ordering)
        # cup_ball is task-driven (cup count / ball vary per task), plain modal.
        elif builder_config.scene_type == "cup_ball":
            task_manager = CupBallTaskManager(builder_config.scene_type,
                                               ordering=ordering)
        else:
            task_manager = TaskRegistryManager(builder_config.scene_type, ordering=ordering)
        if args.mode == "collector" and not task_manager.specs:
            raise SystemExit(
                f"Collector mode for scene {builder_config.scene_type!r} has no tasks "
                f"with choreographer snapshots. Run choreographer mode first to "
                f"capture snapshots under examples/task_scenes/."
            )
        # Choreographer mode: resume on the first task that has no snapshots
        # captured yet, so a re-launched session continues where it left off
        # instead of restarting at task 0.
        if args.mode == "choreographer" and task_manager.specs:
            _task_scenes_root = repo_root / "examples" / "task_scenes"

            def _task_has_snapshots(spec) -> bool:
                d = _task_scenes_root / spec.scene / spec.short_name
                return d.is_dir() and any(d.glob("snapshot_*.png"))

            for _ in range(len(task_manager.specs)):
                if not _task_has_snapshots(task_manager.current_spec):
                    break
                task_manager.next_task()
            else:
                print("Choreographer: every task already has snapshots; "
                      "starting on the first task.")
            print(f"Choreographer: resuming on "
                  f"{task_manager.current_spec.id}")

    def sync_task_manager_to_start_mode() -> None:
        mode = state.domain_randomizer.active_start_mode()
        if task_manager:
            task_manager.set_start_mode(mode)

    sync_task_manager_to_start_mode()

    # Choreographer snapshot runtime: A captures flat-numbered stills under
    # ``examples/task_scenes/<scene>/<task_short>/snapshot_NN.png``. Collector
    # mode reads those same snapshots back for its headset slideshow.
    from mujoco_vr_teleop.snapshot_runtime import (
        SnapshotRuntime,
        hide_arm_geoms,
    )

    task_scenes_root = repo_root / "examples" / "task_scenes"
    current_runtime: list[SnapshotRuntime | None] = [None]
    # Separate model for rendering snapshots: same XML as the sim model (so
    # `data` is layout-compatible) but with the arm links hidden so they don't
    # occlude the task. Kept apart from `model` so the operator's live view is
    # unaffected.
    state.snapshot_model = mujoco.MjModel.from_xml_path(str(xml_path))
    hide_arm_geoms(state.snapshot_model)
    # Collector mode: the slideshow of the choreographer's snapshots auto-plays
    # in the headset; the frontend drives playback, the operator does not
    # control it.

    def refresh_snapshot_state() -> None:
        current_runtime[0] = None
        if args.mode not in ("choreographer", "collector") or task_manager is None:
            return
        task_id = (task_manager.current_task or {}).get("id")
        spec = get_task(task_id) if task_id else None
        if spec is not None:
            current_runtime[0] = SnapshotRuntime(spec=spec, snapshot_root=task_scenes_root)

    refresh_snapshot_state()

    state.body_names = [
        mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
        for bid in range(state.model.nbody)
    ]

    def save_episode_initial_state() -> None:
        # Merge the per-reset uniform object-scale factors (jenga blocks,
        # spell_and_stow letter blocks + cabinet) into the DR metadata so
        # episodes are filterable by scale.
        #
        # The resolved geometry is NOT snapshotted: replay reconstructs it via
        # scene_builder.build_scene_from_recording, which re-applies the DR
        # sample log (the size multipliers, keyed by body name) onto the spec.
        # body_scale + the domain_randomization sample log are the source of
        # truth.
        dr_metadata = state.domain_randomizer.metadata()
        object_scale = scale_log.snapshot()
        if object_scale:
            dr_metadata = {**dr_metadata, "object_scale": object_scale}
        session.save_keyframe(
            state.model,
            state.data,
            domain_randomization=dr_metadata,
            body_scale=state.domain_randomizer.body_scale,
            body_names=state.body_names,
        )

    # Save keyframe state for resets
    state.keyframe_qpos = state.data.qpos.copy()
    state.keyframe_qvel = state.data.qvel.copy()
    state.keyframe_mocap_pos = state.data.mocap_pos.copy() if state.model.nmocap > 0 else None
    state.keyframe_mocap_quat = state.data.mocap_quat.copy() if state.model.nmocap > 0 else None
    save_episode_initial_state()

    def capture_sim_state() -> dict:
        snap = {
            "time": float(state.data.time),
            "qpos": state.data.qpos.copy(),
            "qvel": state.data.qvel.copy(),
            "ctrl": state.data.ctrl.copy(),
        }
        if state.model.na > 0:
            snap["act"] = state.data.act.copy()
        if state.model.nmocap > 0:
            snap["mocap_pos"] = state.data.mocap_pos.copy()
            snap["mocap_quat"] = state.data.mocap_quat.copy()
        return snap

    def restore_sim_state(snap: dict) -> None:
        state.data.time = float(snap["time"])
        state.data.qpos[:] = snap["qpos"]
        state.data.qvel[:] = snap["qvel"]
        state.data.ctrl[:] = snap["ctrl"]
        if state.model.na > 0 and "act" in snap:
            state.data.act[:] = snap["act"]
        if state.model.nmocap > 0 and "mocap_pos" in snap:
            state.data.mocap_pos[:] = snap["mocap_pos"]
            state.data.mocap_quat[:] = snap["mocap_quat"]
        mujoco.mj_forward(state.model, state.data)

    last_saved_state = [capture_sim_state()]

    mp_ctx = mp.get_context("spawn")
    main_cpu_ids, allocate_worker_cpu_ids = _make_cpu_affinity_planner(enabled=True)

    # --- IK mode setup ---
    ik_workers = {}  # side -> worker metadata
    ik_arm_actuator_ids = {}  # side -> list of actuator indices
    ik_arm_ctrl_limits = {}  # side -> (soft lower, soft upper) ctrl arrays
    ik_diagnostic_arm_actuator_ids = {}  # side -> physical arm actuator ids only
    ik_diagnostic_hand_actuator_ids = {}  # side -> Sharpa actuator ids only
    latest_arm_ik_target_pos = {
        side: np.full(3, np.nan, dtype=float) for side in ARM_DIAGNOSTIC_SIDES
    }
    latest_arm_ik_target_quat = {
        side: np.full(4, np.nan, dtype=float) for side in ARM_DIAGNOSTIC_SIDES
    }
    latest_arm_ik_solution_qpos = {
        side: np.full(ARM_DIAGNOSTIC_JOINTS, np.nan, dtype=float)
        for side in ARM_DIAGNOSTIC_SIDES
    }
    latest_hand_ik_solution_qpos = {
        side: np.full(HAND_DIAGNOSTIC_JOINTS, np.nan, dtype=float)
        for side in ARM_DIAGNOSTIC_SIDES
    }
    ik_calibrated = {"right": False, "left": False}

    def actuator_ctrl_limits(act_ids: list[int], margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(len(act_ids), -np.inf, dtype=float)
        upper = np.full(len(act_ids), np.inf, dtype=float)
        margin = max(0.0, float(margin))
        for idx, aid in enumerate(act_ids):
            if not state.model.actuator_ctrllimited[aid]:
                continue
            lo, hi = state.model.actuator_ctrlrange[aid]
            if hi - lo > 2.0 * margin:
                lo += margin
                hi -= margin
            lower[idx] = lo
            upper[idx] = hi
        return lower, upper

    joint_ik_mode = bool(args.ik_mode and args.builder.finger_control == "ik")

    if args.ik_mode:
        def find_sharpa_joints(side: str):
            joint_names = []
            act_ids = []
            prefix = f"{side}_"
            finger_tokens = ("thumb", "index", "middle", "ring", "pinky")
            for aid in range(state.model.nu):
                jid = state.model.actuator_trnid[aid][0]
                jname = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                if jname.startswith(prefix) and any(token in jname for token in finger_tokens):
                    joint_names.append(jname)
                    act_ids.append(aid)
            return joint_names, act_ids

        def joint_ik_targets_for_side(side: str):
            prefix = "left_" if side == "left" else ""
            mocap_to_site = {}
            target_sites = []
            for joint_name, template in HAND_IK_SITE_BY_JOINT.items():
                site_name = template.format(prefix=prefix)
                if mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_SITE, site_name) < 0:
                    continue
                mocap_to_site[f"{side}-{joint_name}"] = site_name
                target_sites.append(site_name)
            return mocap_to_site, target_sites

        arm_configs = [(side, f"{side}-arm-") for side in ARM_DIAGNOSTIC_SIDES]

        for side, prefix in arm_configs:
            jnames = [f"{prefix}joint{i}" for i in range(1, 7)]
            act_ids = [find_actuator_for_joint(name) for name in jnames]
            if not all(i >= 0 for i in act_ids):
                print(f"IK mode: no arm joints found for side '{side}', skipping")
                continue

            base_body_name = f"{side}-arm"
            if mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name) < 0:
                base_body_name = None

            if joint_ik_mode:
                hand_jnames, hand_act_ids = find_sharpa_joints(side)
                mocap_to_site, target_sites = joint_ik_targets_for_side(side)
                if not hand_act_ids or not target_sites:
                    print(f"Joint IK mode: no Sharpa joints/targets found for side '{side}', skipping")
                    continue
                ik_xml_path = resolve_config_path(getattr(args, f"joint_ik_xml_{side}", None))
                if ik_xml_path is None or not ik_xml_path.exists():
                    raise SystemExit(
                        f"--builder.finger-control ik requires prebuilt joint_ik_xml_{side}. "
                        "Regenerate the scene with mujoco_vr_teleop.build_scene."
                    )
                joint_names = jnames + hand_jnames
                joint_act_ids = act_ids + hand_act_ids
                ik_cfg_dict = {
                    "xml_path": str(ik_xml_path),
                    "controlled_joint_names": joint_names,
                    "target_frame_names": target_sites,
                    "backend": "mink",
                    "smoothing_alpha": float(args.ik_smoothing),
                    "home_joints": state.data.ctrl[joint_act_ids].copy().tolist(),
                    "max_joint_delta": float(args.mink_max_joint_delta),
                    **_mink_solver_kwargs(args),
                }
                workers, worker_info = _spawn_worker_pool(
                    mp_ctx,
                    _joint_ik_worker_main,
                    count=args.ik_workers_per_arm,
                    name_prefix=f"joint-ik-worker-{side}",
                    proc_args=lambda child, cpu: (ik_cfg_dict, child, cpu),
                    allocate_worker_cpu_ids=allocate_worker_cpu_ids,
                )
                ik_workers[side] = {
                    "workers": workers,
                    "cfg": ik_cfg_dict,
                    "backend": worker_info["backend"],
                    "devices": worker_info["devices"],
                    "joint_mode": True,
                    "mocap_to_site": mocap_to_site,
                    "base_body_name": base_body_name,
                    "target_map": {},
                    "target_seq": -1,
                    "dispatched_seq": -1,
                    "latest_result": None,
                    "latest_result_seq": -1,
                    "last_applied_seq": -1,
                }
                ik_arm_actuator_ids[side] = joint_act_ids
                ik_diagnostic_arm_actuator_ids[side] = act_ids
                ik_diagnostic_hand_actuator_ids[side] = hand_act_ids
                ik_arm_ctrl_limits[side] = actuator_ctrl_limits(
                    joint_act_ids,
                    args.mink_limit_margin,
                )
                print(
                    f"Joint IK mode ({side}): {len(act_ids)} arm joints + "
                    f"{len(hand_act_ids)} hand joints, {len(target_sites)} targets, "
                    f"xml={ik_xml_path.name}, "
                    f"backend={worker_info['backend']}, devices={worker_info['devices']}, "
                    f"workers={len(workers)}"
                )

        if not wrist_hybrid_weld_mode:
            for side, eq_id in state.wrist_weld_eq_ids.items():
                if side in ik_workers:
                    state.model.eq_active0[eq_id] = 0
                    state.data.eq_active[eq_id] = 0
                    print(f"IK mode: disabled wrist weld constraint {eq_id} ({side})")

    def joint_state_for_actuators(act_ids: list[int], width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos = np.full(width, np.nan, dtype=float)
        qvel = np.full(width, np.nan, dtype=float)
        ctrl = np.full(width, np.nan, dtype=float)
        for idx, aid in enumerate(act_ids[:width]):
            jid = int(state.model.actuator_trnid[aid][0])
            qadr = int(state.model.jnt_qposadr[jid])
            dadr = int(state.model.jnt_dofadr[jid])
            qpos[idx] = float(state.data.qpos[qadr])
            qvel[idx] = float(state.data.qvel[dadr])
            ctrl[idx] = float(state.data.ctrl[aid])
        return qpos, qvel, ctrl

    def arm_ik_diagnostics_frame() -> dict[str, np.ndarray]:
        if not ik_diagnostic_arm_actuator_ids:
            return {}
        target_pos = np.stack([latest_arm_ik_target_pos[side] for side in ARM_DIAGNOSTIC_SIDES])
        target_quat = np.stack([latest_arm_ik_target_quat[side] for side in ARM_DIAGNOSTIC_SIDES])
        solution_qpos = np.stack(
            [latest_arm_ik_solution_qpos[side] for side in ARM_DIAGNOSTIC_SIDES]
        )
        hand_solution_qpos = np.stack(
            [latest_hand_ik_solution_qpos[side] for side in ARM_DIAGNOSTIC_SIDES]
        )
        joint_qpos = np.full((len(ARM_DIAGNOSTIC_SIDES), ARM_DIAGNOSTIC_JOINTS), np.nan, dtype=float)
        joint_qvel = np.full_like(joint_qpos, np.nan)
        joint_ctrl = np.full_like(joint_qpos, np.nan)
        hand_joint_qpos = np.full((len(ARM_DIAGNOSTIC_SIDES), HAND_DIAGNOSTIC_JOINTS), np.nan, dtype=float)
        hand_joint_qvel = np.full_like(hand_joint_qpos, np.nan)
        hand_joint_ctrl = np.full_like(hand_joint_qpos, np.nan)
        for side_idx, side in enumerate(ARM_DIAGNOSTIC_SIDES):
            act_ids = ik_diagnostic_arm_actuator_ids.get(side)
            if act_ids:
                qpos, qvel, ctrl = joint_state_for_actuators(act_ids, ARM_DIAGNOSTIC_JOINTS)
                joint_qpos[side_idx] = qpos
                joint_qvel[side_idx] = qvel
                joint_ctrl[side_idx] = ctrl
            hand_act_ids = ik_diagnostic_hand_actuator_ids.get(side)
            if hand_act_ids:
                qpos, qvel, ctrl = joint_state_for_actuators(hand_act_ids, HAND_DIAGNOSTIC_JOINTS)
                hand_joint_qpos[side_idx] = qpos
                hand_joint_qvel[side_idx] = qvel
                hand_joint_ctrl[side_idx] = ctrl
        frame = {
            "arm_ik_target_pos": target_pos,
            "arm_ik_target_quat_wxyz": target_quat,
            "arm_ik_solution_qpos": solution_qpos,
            "arm_joint_qpos": joint_qpos,
            "arm_joint_qvel": joint_qvel,
            "arm_ctrl": joint_ctrl,
        }
        if ik_diagnostic_hand_actuator_ids:
            frame.update(
                {
                    "hand_ik_solution_qpos": hand_solution_qpos,
                    "hand_joint_qpos": hand_joint_qpos,
                    "hand_joint_qvel": hand_joint_qvel,
                    "hand_ctrl": hand_joint_ctrl,
                }
            )
        return frame

    def set_hybrid_wrist_weld(side: str, active: bool) -> None:
        eq_id = state.wrist_weld_eq_ids.get(side)
        if eq_id is None or state.wrist_hybrid_weld_active.get(side) == active:
            return
        state.data.eq_active[eq_id] = 1 if active else 0
        set_arm_actuators_enabled(side, not active)
        if not active:
            act_ids = state.leader_arm_actuator_ids.get(side, [])
            if act_ids:
                qpos, _, _ = joint_state_for_actuators(act_ids, len(act_ids))
                state.data.ctrl[act_ids] = qpos
        state.wrist_hybrid_weld_active[side] = active

    def reset_hybrid_wrist_state() -> None:
        if not wrist_hybrid_weld_mode:
            return
        for side in ARM_DIAGNOSTIC_SIDES:
            eq_id = state.wrist_weld_eq_ids.get(side)
            if eq_id is not None:
                state.data.eq_active[eq_id] = 0
            set_arm_actuators_enabled(side, True)
            state.wrist_hybrid_weld_active[side] = False
            state.wrist_hybrid_error_pos[side] = np.nan

    def update_hybrid_wrist_modes() -> None:
        if not wrist_hybrid_weld_mode:
            return
        mujoco.mj_forward(state.model, state.data)
        for side in ARM_DIAGNOSTIC_SIDES:
            mid = state.wrist_mocap_ids.get(side)
            sid = state.wrist_palm_site_ids.get(side)
            if mid is None or sid is None:
                continue
            err = float(np.linalg.norm(state.data.site_xpos[sid] - state.data.mocap_pos[mid]))
            state.wrist_hybrid_error_pos[side] = err
            if side not in ik_workers or not ik_calibrated.get(side, False):
                continue
            if not state.wrist_hybrid_weld_active[side] and err < float(args.weld_enter_thresh):
                set_hybrid_wrist_weld(side, True)

    def wrist_hybrid_diagnostics_frame() -> dict[str, np.ndarray]:
        if not wrist_hybrid_weld_mode:
            return {}
        return {
            "wrist_hybrid_weld_mask": np.asarray(
                [state.wrist_hybrid_weld_active[side] for side in ARM_DIAGNOSTIC_SIDES],
                dtype=np.bool_,
            ),
            "state.wrist_hybrid_error_pos": np.asarray(
                [state.wrist_hybrid_error_pos[side] for side in ARM_DIAGNOSTIC_SIDES],
                dtype=float,
            ),
        }

    last_hybrid_ctrl_label_frame = [-1]

    def write_hybrid_arm_ctrl_label(frame_idx: int) -> None:
        if not wrist_hybrid_weld_mode:
            return
        for side in ARM_DIAGNOSTIC_SIDES:
            act_ids = ik_diagnostic_arm_actuator_ids.get(side) or state.leader_arm_actuator_ids.get(side, [])
            values = latest_arm_ik_solution_qpos[side]
            session.write_frame_ctrl(frame_idx, act_ids, values)
        last_hybrid_ctrl_label_frame[0] = max(last_hybrid_ctrl_label_frame[0], frame_idx)

    def flush_hybrid_arm_ctrl_labels() -> None:
        if not wrist_hybrid_weld_mode:
            return
        last_hybrid_ctrl_label_frame[0] = min(
            last_hybrid_ctrl_label_frame[0],
            session.frame_count - 1,
        )
        for frame_idx in range(last_hybrid_ctrl_label_frame[0] + 1, session.frame_count):
            write_hybrid_arm_ctrl_label(frame_idx)

    ik_wrist_names = {f"{side}-wrist" for side in ik_workers}

    _set_process_affinity(main_cpu_ids, "Main process")

    # Pending commands from the web client
    pending_commands = []
    # JSON messages produced by the physics thread to broadcast over websockets.
    pending_broadcasts: list[dict] = []
    record_delay_until = [0.0]  # timestamp when delay ends; 0 = no delay
    # Cooldown after exiting playground or loading a new task: B-tap can't
    # resume/start tracking until this timestamp passes, so an operator still
    # releasing the pedal doesn't immediately start the next recording.
    RESUME_COOLDOWN_S = 0.5
    resume_blocked_until = [0.0]

    def begin_resume_cooldown():
        resume_blocked_until[0] = time.time() + RESUME_COOLDOWN_S
    tracking_active = [False]
    tracking_paused = [False]
    # Boot in playground: tracking forwarded so the user can move around, but
    # no recording happens until they hold B 1.5s to leave playground.
    playground_mode = [True]
    # In collector mode, when the operator skips a task (C-hold) the streamer
    # pauses and waits for them to choose what to do next. A press repeats the
    # same task (fresh DR draw); C press advances to the next task.
    awaiting_post_task_choice = [False]
    # User-tunable vertical offset (meters) that the frontend applies to its
    # entire scene root. Sim coordinates are unchanged; this is purely a
    # visual lift so different-height users can place the table at a
    # comfortable height. Adjusted via A/C foot pedals while in playground;
    # persisted to ~/.vr_streamer/table_height.json across runs.
    from mujoco_vr_teleop import state_store
    from mujoco_vr_teleop.state_store import (
        clamp_height_offset,
        load_table_height_offset,
        save_table_height_offset,
    )
    table_height_offset = [load_table_height_offset()]
    print(f"Loaded table height offset: {table_height_offset[0]:+.3f} m")
    # Discrete step (meters) applied per raise_table/lower_table command, e.g.
    # from the supervisor TUI's up/down arrow keys.
    TABLE_HEIGHT_STEP = 0.02
    # RLock so reentrant callers (e.g. rebuild_scene invoked from inside the
    # physics_loop's already-locked command handler) don't deadlock.
    lock = threading.RLock()

    def _drain_ik_inflight():
        for pool in ik_workers.values():
            for worker in pool["workers"]:
                if worker.get('in_flight'):
                    try:
                        worker['conn'].recv()
                    except EOFError:
                        pass
                    worker['in_flight'] = False

    def clear_tracking_state():
        pending_mocap.clear()
        pending_triggers.clear()
        reset_hybrid_wrist_state()
        for side in ik_calibrated:
            ik_calibrated[side] = False
        _drain_ik_inflight()
        for side, pool in ik_workers.items():
            act_ids = ik_arm_actuator_ids.get(side)
            home = state.data.ctrl[act_ids].copy() if act_ids is not None else None
            msg = {
                "cmd": "reset",
                "home_joints": None if home is None else home.tolist(),
            }
            base_body_name = pool.get("base_body_name")
            if base_body_name:
                bid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
                if bid >= 0:
                    msg["body_positions"] = {
                        base_body_name: state.data.xpos[bid].copy().tolist(),
                    }
            pool["target_seq"] = -1
            pool["dispatched_seq"] = -1
            pool["latest_result"] = None
            pool["latest_result_seq"] = -1
            pool["last_applied_seq"] = -1
            pool["target_map"] = {}
            for worker in pool["workers"]:
                worker["conn"].send(msg)
        for side, pool in ik_workers.items():
            for worker in pool["workers"]:
                reply = worker["conn"].recv()
                if not reply.get("ok"):
                    raise RuntimeError(f"IK worker reset failed ({side})")

    def deactivate_tracking(reason: str):
        was_active = tracking_active[0]
        tracking_active[0] = False
        tracking_paused[0] = False
        record_delay_until[0] = 0.0
        clear_tracking_state()
        if was_active or reason:
            print(f"Tracking deactivated ({reason})")

    def activate_tracking():
        if tracking_active[0]:
            print("Tracking already active")
            return
        clear_tracking_state()
        tracking_active[0] = True
        tracking_paused[0] = False
        if args.record_delay > 0:
            record_delay_until[0] = time.time() + args.record_delay
            print(f"Tracking activated; recording starts in {args.record_delay:.2f}s")
        else:
            record_delay_until[0] = 0.0
            session.start_recording()
            print("Tracking activated")

    def pause_tracking():
        if not tracking_active[0] or tracking_paused[0]:
            return
        tracking_paused[0] = True
        record_delay_until[0] = 0.0
        pending_mocap.clear()
        pending_triggers.clear()
        reset_hybrid_wrist_state()
        if not playground_mode[0]:
            session.pause_recording()
        print("Tracking paused")

    def resume_tracking():
        if not tracking_active[0]:
            activate_tracking()
            return
        if not tracking_paused[0]:
            return
        clear_tracking_state()
        tracking_paused[0] = False
        if not playground_mode[0]:
            session.resume_recording()
        print("Tracking resumed")

    def toggle_tracking_pause():
        # Pausing is always allowed; resuming/starting is blocked during the
        # post-playground / post-task-load cooldown.
        resuming = not tracking_active[0] or tracking_paused[0]
        if resuming and time.time() < resume_blocked_until[0]:
            remaining = resume_blocked_until[0] - time.time()
            print(f"Resume blocked; cooldown {remaining:.1f}s remaining")
            return
        if not tracking_active[0]:
            activate_tracking()
        elif tracking_paused[0]:
            resume_tracking()
        else:
            pause_tracking()

    def advance_task_and_pick_mode() -> str | None:
        """Pop the next task and uniformly sample one of its allowed modes.

        Returns ``None`` if there's no task manager or the task is
        mode-agnostic (caller can let the DR sample its own mode). Otherwise
        returns the chosen DR ``start_mode`` to force on the randomizer.

        This commits the task pointer immediately so the subsequent
        ``reset_scene(forced_mode=...)`` and its internal
        ``sync_task_manager_to_start_mode`` don't reshuffle the task queue
        out from under us.
        """
        if task_manager is None:
            return None
        next_task = task_manager.next_task()
        if next_task is None:
            return None
        allowed = task_allowed_start_modes(next_task)
        if not allowed:
            return None
        dr_modes = list(
            (state.domain_randomizer.config.get("start_modes") or {}).keys()
        )
        valid = [m for m in allowed if m in dr_modes]
        if not valid:
            print(
                f"Warning: task {next_task.get('id')} allows {allowed} but "
                f"DR exposes {dr_modes}; falling back to DR sampling"
            )
            return None
        import random as _random
        return _random.choice(valid)

    def broadcast_loading(active: bool, label: str = "") -> None:
        """Send a {type: loading} message so the headset can show/hide the
        loading panel during a long-running operation."""
        pending_broadcasts.append({
            "type": "loading",
            "active": bool(active),
            "label": label,
        })

    def broadcast_scene_will_reload() -> None:
        """Tell the headset to start tearing down assets NOW, instead of
        waiting for the next /api/status poll to notice the session_id bump
        (which can be up to 500ms late and lets the user start teleoping
        into a half-reloaded scene)."""
        pending_broadcasts.append({"type": "scene_will_reload"})

    def rebuild_scene() -> None:
        """Rebuild the MuJoCo model from scratch, picking new variants.

        Re-runs build_scene so the variant_pools pick fresh variants, then
        refreshes every cached object that was derived from the old model:
        DomainRandomizer, body_info / GLB exports, hand geom IDs, table_gid,
        workspace bounds, mocap dictionaries, actuator dictionaries, wrist
        weld dictionaries, snapshot_model, body_names, init_qpos.

        NOT refreshed here:
          - recorder session metadata (an episode in flight would be
            misaligned; caller is responsible for save_episode_initial_state
            after the rebuild via reset_scene).
        """
        print("[rebuild_scene] rebuilding scene with fresh variant selection")
        broadcast_scene_will_reload()
        new_scene = build_scene(builder_config)

        # Acquire the physics lock so the physics_loop thread can't step
        # mid-swap. All mutations below happen while the loop is blocked.
        with lock:
            _do_rebuild_inside_lock(new_scene)
        print(f"[rebuild_scene] swap complete: nbody={state.model.nbody}, "
              f"ngeom={state.model.ngeom}, body_info={len(state.body_info)}, "
              f"new session_id={state.session_id[:8]}")

    def _do_rebuild_inside_lock(new_scene) -> None:
        # Hot-swap the model + data references everyone sees via state.
        state.model = new_scene.model
        state.model.opt.timestep = 1.0 / float(args.physics_rate)
        state.data = mujoco.MjData(state.model)
        if state.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(state.model, state.data, 0)
        mujoco.mj_forward(state.model, state.data)
        state.data.ctrl[:] = 0.0
        align_wrist_mocaps_to_palms()
        mujoco.mj_forward(state.model, state.data)

        # Re-create the DR against the new model (it captures geom/body refs).
        dr_cfg = (
            new_scene.streamer_config.get("domain_randomization")
            if args.domain_randomization
            else {"enabled": False, "settle": False}
        )
        state.domain_randomizer = DomainRandomizer.create(
            state.model, state.data, dr_cfg,
        )

        # Re-export body GLBs (the headset will need to refetch).
        import shutil as _shutil
        if mesh_dir.exists():
            _shutil.rmtree(mesh_dir)
        state.body_info = export_body_glbs(
            state.model, mesh_dir, collision_mode=bool(args.collision_mode),
        )
        # Refresh the streamed body-id list: variant rebuilds can renumber
        # bodies (one wrapper level deeper or shallower), so the integer IDs
        # captured at startup must be reread against the new model.
        dyn_ids = [bid for bid, info in state.body_info.items() if not info["is_fixed"]]
        state.stream_body_ids = sorted(
            set(dyn_ids) | state.domain_randomizer.stream_body_ids
        )
        if _playground_spec is not None:
            _playground_spec.reset(state.model, state.data,
                                    np.random.default_rng())
        align_wrist_mocaps_to_palms()
        state.domain_randomizer.settle()
        refresh_body_info_poses()
        _arm_opacity = float(args.arm_opacity)
        for _info in state.body_info.values():
            _name = _info["name"].lower()
            if "sharpa" not in _name and ("right-arm" in _name or "left-arm" in _name):
                _info["opacity"] = _arm_opacity

        # Hand geom IDs (Sharpa palm + fingers, both hands).
        _hand_tokens_rb = ("thumb", "index", "middle", "ring", "pinky", "palm")
        state.hand_geom_ids = set()
        for gid in range(state.model.ngeom):
            bid = int(state.model.geom_bodyid[gid])
            bname = (mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            is_sharpa = "sharpa" in bname
            is_finger = (
                (bname.startswith("left_") or bname.startswith("right_"))
                and any(tok in bname for tok in _hand_tokens_rb)
            )
            if is_sharpa or is_finger:
                state.hand_geom_ids.add(gid)

        # Table geom + workspace bounds.
        state.table_gid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
        if state.table_gid < 0:
            state.table_gid = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
        recompute_workspace_bounds()
        state.init_qpos = state.data.qpos.copy()

        # Mocap info.
        state.mocap_info = []
        state.mocap_id_to_name = {}
        state.hand_mocap_mapping = {"left": {}, "right": {}}
        for i in range(state.model.nbody):
            mid = state.model.body_mocapid[i]
            if mid < 0:
                continue
            pos_yup = R_conv @ state.data.mocap_pos[mid]
            mj_quat = state.data.mocap_quat[mid]
            mat_zup = Rotation.from_quat(
                [mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]]
            ).as_matrix()
            q_yup = Rotation.from_matrix(R_conv @ mat_zup @ R_conv.T).as_quat()
            name = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"mocap_{mid}"
            state.mocap_id_to_name[int(mid)] = name
            for side in ("left", "right"):
                pref = f"{side}-"
                if name.startswith(pref):
                    jn = name[len(pref):]
                    if jn in HAND_JOINT_NAMES:
                        state.hand_mocap_mapping[side][jn] = int(mid)
            state.mocap_info.append({
                "mocap_id": int(mid), "body_id": int(i), "name": name,
                "position": pos_yup.tolist(), "quaternion": q_yup.tolist(),
            })

        # Arm + gripper actuator IDs.
        state.leader_arm_actuator_ids = {
            "left": find_yam_arm_actuator_ids("left"),
            "right": find_yam_arm_actuator_ids("right"),
        }
        state.arm_actuator_gains = {
            aid: (state.model.actuator_gainprm[aid].copy(),
                  state.model.actuator_biasprm[aid].copy())
            for act_ids in state.leader_arm_actuator_ids.values()
            for aid in act_ids
        }

        # Wrist weld equality IDs.
        state.wrist_weld_eq_ids = {}
        state.wrist_mocap_ids = {}
        state.wrist_palm_site_ids = {}
        for mocap_name, site_name in SHARPA_WRIST_MOCAP_SITE_PAIRS:
            side = mocap_name.split("-")[0]
            mocap_site_name = f"{mocap_name}-site-mocap"
            palm_site_id = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            body_id = mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
            if body_id >= 0 and int(state.model.body_mocapid[body_id]) >= 0:
                state.wrist_mocap_ids[side] = int(state.model.body_mocapid[body_id])
            if palm_site_id >= 0:
                state.wrist_palm_site_ids[side] = palm_site_id
            for eq_id in range(state.model.neq):
                if state.model.eq_type[eq_id] != mujoco.mjtEq.mjEQ_WELD:
                    continue
                if state.model.eq_objtype[eq_id] != mujoco.mjtObj.mjOBJ_SITE:
                    continue
                s1 = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_SITE,
                                         state.model.eq_obj1id[eq_id])
                s2 = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_SITE,
                                         state.model.eq_obj2id[eq_id])
                if {s1, s2} == {mocap_site_name, site_name}:
                    state.wrist_weld_eq_ids[side] = eq_id
                    break
        if wrist_hybrid_weld_mode:
            for eq_id in state.wrist_weld_eq_ids.values():
                state.model.eq_active0[eq_id] = 0
                state.data.eq_active[eq_id] = 0

        # Gripper actuator IDs.
        state.gripper_actuator_ids = {"left": None, "right": None}
        for i in range(state.model.nu):
            aname = mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            al = aname.lower()
            if "grip" not in al and "finger" not in al:
                continue
            cr = state.model.actuator_ctrlrange[i]
            if "left" in al:
                state.gripper_actuator_ids["left"] = (i, float(cr[0]), float(cr[1]))
            elif "right" in al:
                state.gripper_actuator_ids["right"] = (i, float(cr[0]), float(cr[1]))
            elif state.gripper_actuator_ids["right"] is None:
                state.gripper_actuator_ids["right"] = (i, float(cr[0]), float(cr[1]))

        # Snapshot model + body names.
        state.snapshot_model = mujoco.MjModel.from_xml_path(str(new_scene.xml_path.resolve()))
        hide_arm_geoms(state.snapshot_model)
        state.body_names = [
            mujoco.mj_id2name(state.model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
            for bid in range(state.model.nbody)
        ]

        # Refresh keyframe snapshot for the new model size (qpos length may differ).
        state.keyframe_qpos = state.data.qpos.copy()
        state.keyframe_qvel = state.data.qvel.copy()
        if state.model.nmocap > 0:
            state.keyframe_mocap_pos = state.data.mocap_pos.copy()
            state.keyframe_mocap_quat = state.data.mocap_quat.copy()
        else:
            state.keyframe_mocap_pos = None
            state.keyframe_mocap_quat = None

        # The rebuild re-rolled the variant pools (bottle count + ids). Push the
        # fresh selection into the recorder so the next episode's saved
        # streamer_config matches the scene it recorded against, instead of
        # inheriting the first build's stale chosen_variants.
        session.update_variant_pools(
            (new_scene.streamer_config.get("domain_randomization") or {})
            .get("variant_pools")
        )
        # The rebuild also generated a fresh scene XML; point the recorder at it
        # so the next episode saves its own XML rather than the first build's.
        session.update_scene_xml(new_scene.xml_path)

        # Bump session_id so the frontend's status poll sees a new session
        # and refetches all GLB assets / mocap / body info.
        state.session_id = uuid.uuid4().hex

    def reset_scene(forced_mode: str | None = None):
        """Reset simulation to keyframe with optional randomization.

        If ``forced_mode`` is given, the DR samples that ``start_mode`` from
        its config (must be one of the configured modes) — this is a genuine
        task advance, so it also starts the resume cooldown. A bare reset
        (``forced_mode=None``, i.e. a revert) does not.

        Otherwise the DR samples by the per-mode probabilities in its config.

        New tasks/ registry: when the active task has a registered TaskSpec,
        we delegate to its bespoke ``reset(state.model, state.data, rng)`` instead of
        the JSON-driven DR. This lets each task own its initial state.
        """
        tracking_active[0] = False
        tracking_paused[0] = False
        if forced_mode is not None:
            begin_resume_cooldown()
        # Notify the headset that the scene will be unresponsive during the
        # reset (scatter + settle physics can take several seconds for the
        # dishrack scene). cleared in the `finally` of _reset_scene_inner.
        broadcast_loading(True, "Resetting scene")
        try:
            _reset_scene_inner(forced_mode)
        finally:
            broadcast_loading(False)

    def _reset_scene_inner(forced_mode: str | None):
        # Per-task reset (new path) takes precedence when available. When a
        # registry task is current, IT is the authority for the scene — we do
        # NOT fall through to the scene-DR / keyframe path on failure, because
        # that produces a scene unrelated to the current task (e.g. the
        # previous task's geometry left half-overwritten). Instead: retry the
        # task's own reset once with a fresh RNG draw (transient settle/NaN
        # blowups usually clear on a re-roll), and if it still fails, restore
        # the keyframe as a known-safe state and log the full traceback.
        if task_manager is not None:
            spec = task_manager.current_spec
            if spec is not None:
                for attempt in (1, 2):
                    try:
                        spec.reset(state.model, state.data, np.random.default_rng())
                        align_wrist_mocaps_to_palms()
                        mujoco.mj_forward(state.model, state.data)
                        recompute_workspace_bounds()
                        sync_task_manager_to_start_mode()
                        refresh_body_info_poses()
                        save_episode_initial_state()
                        last_saved_state[0] = capture_sim_state()
                        clear_tracking_state()
                        print(f"Scene reset via TaskSpec.reset for {spec.id}"
                              + (f" (attempt {attempt})" if attempt > 1 else ""))
                        return
                    except Exception:
                        print(f"TaskSpec {spec.id} reset raised "
                              f"(attempt {attempt}/2):\n{traceback.format_exc()}")
                # Both attempts failed: restore the keyframe so the scene is in
                # a known state rather than a half-applied chimera, then bail.
                print(f"TaskSpec {spec.id} reset failed twice; "
                      f"restoring keyframe.")
                state.data.qpos[:] = state.keyframe_qpos
                state.data.qvel[:] = state.keyframe_qvel
                state.data.ctrl[:] = 0.0
                if state.keyframe_mocap_pos is not None:
                    state.data.mocap_pos[:] = state.keyframe_mocap_pos
                if state.keyframe_mocap_quat is not None:
                    state.data.mocap_quat[:] = state.keyframe_mocap_quat
                align_wrist_mocaps_to_palms()
                mujoco.mj_forward(state.model, state.data)
                recompute_workspace_bounds()
                sync_task_manager_to_start_mode()
                refresh_body_info_poses()
                save_episode_initial_state()
                last_saved_state[0] = capture_sim_state()
                clear_tracking_state()
                return
        if state.domain_randomizer.enabled:
            # If the DR config specifies a quiescence check we let the
            # randomizer retry until the scene is quiet (or the attempt budget
            # is hit). Otherwise this is just a randomize() + settle().
            resampling = state.domain_randomizer.config.get("resampling") or {}
            attempts_used = state.domain_randomizer.randomize_and_settle(
                max_attempts=int(resampling.get("max_attempts", 1)),
                check_quiescence=resampling.get("check_quiescence"),
                forced_mode=forced_mode,
            )
            align_wrist_mocaps_to_palms()
            if attempts_used > 1:
                print(f"Reset used {attempts_used} attempts to reach quiescence")
        else:
            state.data.qpos[:] = state.keyframe_qpos
            state.data.qvel[:] = state.keyframe_qvel
            state.data.ctrl[:] = 0.0
            if state.keyframe_mocap_pos is not None:
                state.data.mocap_pos[:] = state.keyframe_mocap_pos
            if state.keyframe_mocap_quat is not None:
                state.data.mocap_quat[:] = state.keyframe_mocap_quat
            align_wrist_mocaps_to_palms()
            mujoco.mj_forward(state.model, state.data)
            state.domain_randomizer.settle()
        recompute_workspace_bounds()
        sync_task_manager_to_start_mode()
        refresh_body_info_poses()
        save_episode_initial_state()
        last_saved_state[0] = capture_sim_state()
        clear_tracking_state()
        if state.domain_randomizer.enabled:
            print(
                "Scene reset; domain randomization applied "
                f"(mode={state.domain_randomizer.active_start_mode()}, {len(state.domain_randomizer.sample_log)} samples)"
            )
        else:
            print("Scene reset")

    def set_keyframe():
        """Capture current sim state as the new reset keyframe."""
        state.keyframe_qpos = state.data.qpos.copy()
        state.keyframe_qvel = state.data.qvel.copy()
        if state.model.nmocap > 0:
            state.keyframe_mocap_pos = state.data.mocap_pos.copy()
            state.keyframe_mocap_quat = state.data.mocap_quat.copy()
        current_ctrl = state.data.ctrl.copy()
        state.data.ctrl[:] = 0.0
        try:
            save_episode_initial_state()
            # Save persistent keyframe file (reusable across sessions)
            save_keyframe_file(keyframe_file_path, state.model, state.data)
        finally:
            state.data.ctrl[:] = current_ctrl
        print("Keyframe set from current state")

    def save_recording_episode(
        task_info: dict | None,
        *,
        trim_to_last_marker: bool,
        clip_head: bool = True,
        finalize: bool = True,
    ) -> Path | None:
        flush_hybrid_arm_ctrl_labels()
        before_seconds = float(session.total_recorded_seconds)
        result = session.save_episode(
            task_info=task_info,
            trim_to_last_marker=trim_to_last_marker,
            clip_head=clip_head,
            finalize=finalize,
        )
        if result is not None:
            delta_seconds = max(0.0, float(session.total_recorded_seconds) - before_seconds)
            try:
                state_store.record_episode_metric(
                    args.builder.scene_type,
                    minutes=delta_seconds / 60.0,
                    episodes=1,
                )
            except Exception as exc:
                print(f"Failed to record episode metric: {exc}")
        return result

    def revert_to_last_save_and_pause() -> None:
        was_active = tracking_active[0]
        task_info = task_manager.task_info if task_manager else None
        session.discard_to_checkpoint(task_info=task_info)
        restore_sim_state(last_saved_state[0])
        refresh_body_info_poses()
        clear_tracking_state()
        record_delay_until[0] = 0.0
        tracking_active[0] = was_active
        tracking_paused[0] = was_active
        save_episode_initial_state()
        print("Reverted to last save" + (" and paused" if was_active else ""))

    frame_width = 11

    def build_frame_buffer():
        """Pack streamed body transforms into a flat float32 buffer (Y-up)."""
        buf = np.zeros(len(state.stream_body_ids) * frame_width, dtype=np.float32)
        for i, bid in enumerate(state.stream_body_ids):
            pos_yup, q = get_body_pose_yup(bid)
            offset = i * frame_width
            buf[offset] = float(bid)
            buf[offset + 1:offset + 4] = pos_yup
            buf[offset + 4:offset + 8] = q
            buf[offset + 8:offset + 11] = get_body_scale(bid)
        return buf.tobytes()

    # Shared state for physics thread
    frame_bytes = [build_frame_buffer()]

    def physics_loop():
        physics_rate = float(args.physics_rate)
        command_rate = float(args.control_rate)
        record_rate = float(args.record_rate or args.control_rate)
        physics_dt = 1.0 / physics_rate
        command_interval_steps = max(1, round(physics_rate / command_rate))
        record_interval_steps = max(1, round(physics_rate / record_rate))
        stream_interval_steps = max(1, round(physics_rate / float(args.stream_rate)))
        if not np.isclose(command_interval_steps * command_rate, physics_rate, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"physics_rate ({physics_rate}) must be an integer multiple of control_rate ({command_rate})"
            )
        if not np.isclose(record_interval_steps * record_rate, physics_rate, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"physics_rate ({physics_rate}) must be an integer multiple of record_rate ({record_rate})"
            )
        R_inv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # Y-up to Z-up

        print(
            "Loop rates: "
            f"physics={physics_rate:.1f}Hz, command={command_rate:.1f}Hz, "
            f"record={record_rate:.1f}Hz, command_interval={command_interval_steps} steps, "
            f"record_interval={record_interval_steps} steps, stream_interval={stream_interval_steps} steps "
            f"(timestep={state.model.opt.timestep})"
        )

        sim_step = 0

        # Latency probe: rolling samples of worker recv() wait times.
        latency_samples: dict[str, list[float]] = {}
        latency_last_print = time.perf_counter()
        latency_print_interval = 5.0

        def poll_ik_results() -> None:
            for side, pool in ik_workers.items():
                for worker in pool["workers"]:
                    if not worker.get("in_flight") or not worker["conn"].poll():
                        continue
                    result = worker["conn"].recv()
                    worker["in_flight"] = False
                    seq = int(worker["seq"])
                    total_ms = (time.perf_counter() - worker["send_ts"]) * 1000.0
                    latency_samples.setdefault(f"ik-{side}-total", []).append(total_ms)
                    latency_samples.setdefault(f"mink-{side}-solve", []).append(float(result.get("solve_ms", total_ms)))
                    if seq <= pool["latest_result_seq"]:
                        continue
                    cmd = np.asarray(result["joint_positions"], dtype=float)
                    pool["latest_result"] = cmd
                    pool["latest_result_seq"] = seq
                    arm_solution = np.full(ARM_DIAGNOSTIC_JOINTS, np.nan, dtype=float)
                    n_arm = min(
                        ARM_DIAGNOSTIC_JOINTS,
                        len(ik_diagnostic_arm_actuator_ids.get(side, [])),
                        cmd.shape[0],
                    )
                    if n_arm > 0:
                        arm_solution[:n_arm] = cmd[:n_arm]
                    latest_arm_ik_solution_qpos[side] = arm_solution
                    hand_solution = np.full(HAND_DIAGNOSTIC_JOINTS, np.nan, dtype=float)
                    if pool.get("joint_mode"):
                        hand_start = len(ik_diagnostic_arm_actuator_ids.get(side, []))
                        n_hand = min(
                            HAND_DIAGNOSTIC_JOINTS,
                            len(ik_diagnostic_hand_actuator_ids.get(side, [])),
                            max(0, cmd.shape[0] - hand_start),
                        )
                        if n_hand > 0:
                            hand_solution[:n_hand] = cmd[hand_start:hand_start + n_hand]
                    latest_hand_ik_solution_qpos[side] = hand_solution

        def apply_latest_worker_results() -> None:
            for side, pool in ik_workers.items():
                if pool["latest_result"] is None or pool["latest_result_seq"] <= pool["last_applied_seq"]:
                    continue
                cmd = pool["latest_result"].copy()
                limits = ik_arm_ctrl_limits.get(side)
                if limits is not None:
                    cmd = np.clip(cmd, limits[0], limits[1])
                state.data.ctrl[ik_arm_actuator_ids[side]] = cmd
                pool["last_applied_seq"] = pool["latest_result_seq"]

        def dispatch_ik_targets() -> None:
            for side, pool in ik_workers.items():
                seq = int(pool["target_seq"])
                if seq <= pool["dispatched_seq"]:
                    continue
                worker = next((w for w in pool["workers"] if not w.get("in_flight")), None)
                if worker is None:
                    continue
                act_ids = ik_arm_actuator_ids[side]
                targets = list(pool["target_map"].values())
                if not targets:
                    continue
                msg = {
                    "cmd": "solve",
                    "targets": targets,
                    "seed_q": state.data.ctrl[act_ids].copy().tolist(),
                }
                worker["conn"].send(msg)
                worker["in_flight"] = True
                worker["seq"] = seq
                worker["send_ts"] = time.perf_counter()
                pool["dispatched_seq"] = seq

        _last_iter_ts = time.perf_counter()
        _last_cmd_label = "<idle>"
        # Stage timer: each iter records timestamps as it crosses stage
        # boundaries; when an iter is slow, dump the deltas so we can see
        # WHICH stage (lock-acquire, command handling, mj_step, record_frame,
        # etc.) ate the time.
        _stage_marks: list[tuple[str, float]] = []
        def _mark_stage(label: str) -> None:
            _stage_marks.append((label, time.perf_counter()))
        while True:
            t0 = time.perf_counter()
            _stage_marks.clear()
            _stage_marks.append(("iter_start", t0))
            # Iteration-gap watchdog: log if more than 250ms elapsed since the
            # previous loop iteration completed, so a freeze leaves a marker
            # naming the last command processed.
            _gap_ms = (t0 - _last_iter_ts) * 1000.0
            if _gap_ms > 250.0:
                print(f"[physics_loop] slow iter gap={_gap_ms:.0f}ms after cmd={_last_cmd_label!r} "
                      f"(this iter will report its own per-stage breakdown)")
            _last_iter_ts = t0

            with lock:
                _mark_stage("after_lock_acquire")
                command_due = sim_step % command_interval_steps == 0
                record_due_after_step = (sim_step + 1) % record_interval_steps == 0
                stream_due_after_step = (sim_step + 1) % stream_interval_steps == 0

                def queue_ik_target(
                    side: str,
                    frame_name: str,
                    pos_zup: np.ndarray,
                    quat_wxyz: list[float],
                    position_weight: float,
                    orientation_weight: float,
                ) -> None:
                    worker = ik_workers[side]
                    worker["target_map"][frame_name] = {
                        "frame_name": frame_name,
                        "position": pos_zup.tolist(),
                        "quaternion": list(quat_wxyz),
                        "position_weight": float(position_weight),
                        "orientation_weight": float(orientation_weight),
                    }
                    worker["target_seq"] += 1

                def apply_mocap_update(mid: int, pos_zup: np.ndarray, q_zup: np.ndarray) -> None:
                    quat_wxyz = [q_zup[3], q_zup[0], q_zup[1], q_zup[2]]
                    mocap_name = state.mocap_id_to_name.get(mid, "")

                    # Clamp wrist mocaps into the per-hand workspace cuboid.
                    if mocap_name in ("right-wrist", "left-wrist"):
                        side = mocap_name.split("-")[0]
                        bounds = state.workspace_bounds.get(side)
                        if bounds is not None:
                            pos_zup = np.clip(pos_zup, bounds[0], bounds[1])

                    if mocap_name in ik_wrist_names:
                        side = mocap_name.split("-")[0]
                        if side in ik_workers:
                            worker = ik_workers[side]
                            if not ik_calibrated[side]:
                                init_pos = state.data.mocap_pos[mid]
                                if np.allclose(pos_zup, init_pos, atol=0.01):
                                    return
                                ik_calibrated[side] = True
                                print(f"IK tracking started ({side}): hand={pos_zup.round(3)}")

                            state.data.mocap_pos[mid] = pos_zup
                            state.data.mocap_quat[mid] = quat_wxyz

                            if worker.get("joint_mode"):
                                frame_name = worker["mocap_to_site"].get(mocap_name)
                                if frame_name:
                                    latest_arm_ik_target_pos[side] = pos_zup.copy()
                                    latest_arm_ik_target_quat[side] = np.asarray(quat_wxyz, dtype=float)
                                    queue_ik_target(
                                        side,
                                        frame_name,
                                        pos_zup,
                                        quat_wxyz,
                                        HAND_IK_PALM_POS_WEIGHT,
                                        float(args.ik_ori_weight),
                                    )
                        return

                    if mocap_name:
                        side = mocap_name.split("-")[0]
                        worker = ik_workers.get(side)
                        if worker is not None and worker.get("joint_mode"):
                            state.data.mocap_pos[mid] = pos_zup
                            state.data.mocap_quat[mid] = quat_wxyz
                            frame_name = worker["mocap_to_site"].get(mocap_name)
                            if frame_name and ik_calibrated.get(side, False):
                                queue_ik_target(
                                    side,
                                    frame_name,
                                    pos_zup,
                                    quat_wxyz,
                                    HAND_IK_FINGER_POS_WEIGHT,
                                    HAND_IK_FINGER_ORI_WEIGHT,
                                )
                            return

                    state.data.mocap_pos[mid] = pos_zup
                    state.data.mocap_quat[mid] = quat_wxyz

                poll_ik_results()

                if tracking_active[0] and not tracking_paused[0]:
                    latest_mocap_by_id = {}
                    for update in pending_mocap:
                        latest_mocap_by_id[update["mocap_id"]] = update
                    pending_mocap.clear()

                    process_ts = time.perf_counter()
                    for update in latest_mocap_by_id.values():
                        recv_ts = update.get("_server_recv_ts")
                        if recv_ts is not None:
                            latency_samples.setdefault("mocap-recv-delay", []).append(
                                (process_ts - float(recv_ts)) * 1000.0
                            )
                        mid = update["mocap_id"]
                        if not (0 <= mid < state.model.nmocap):
                            continue
                        position = update.get("position")
                        quaternion = update.get("quaternion")
                        if position is None or quaternion is None:
                            continue
                        if any(v is None for v in position) or any(v is None for v in quaternion):
                            continue
                        pos_zup = R_inv @ np.array(position, dtype=float)
                        mat_yup = Rotation.from_quat(quaternion).as_matrix()
                        mat_zup = R_inv @ mat_yup @ R_inv.T
                        q_zup = Rotation.from_matrix(mat_zup).as_quat()
                        apply_mocap_update(mid, pos_zup, q_zup)

                    dispatch_ik_targets()
                else:
                    pending_mocap.clear()

                if command_due:
                    apply_latest_worker_results()
                    update_hybrid_wrist_modes()

                    trig = pending_triggers[-1] if pending_triggers else None
                    pending_triggers.clear()
                    if trig and tracking_active[0] and not tracking_paused[0]:
                        for side in ["left", "right"]:
                            info = state.gripper_actuator_ids.get(side)
                            if info:
                                aid, lo, hi = info
                                val = 1.0 - trig.get(side, 0)
                                state.data.ctrl[aid] = lo + val * (hi - lo)

                    _mark_stage("before_commands")
                    while pending_commands:
                        cmd = pending_commands.pop(0)
                        _last_cmd_label = cmd
                        _cmd_t0 = time.perf_counter()
                        if cmd == "save":
                            task_info = task_manager.task_info if task_manager else None
                            save_recording_episode(task_info=task_info, trim_to_last_marker=True)
                            # Pick next task + its allowed mode atomically so
                            # the reset uses the right mode for the upcoming
                            # task.
                            chosen_mode = advance_task_and_pick_mode()
                            reset_scene(forced_mode=chosen_mode)
                            record_delay_until[0] = 0.0
                            refresh_snapshot_state()
                        elif cmd == "save_checkpoint":
                            if playground_mode[0]:
                                # Playground: snapshot state only so short-C can revert here.
                                last_saved_state[0] = capture_sim_state()
                                print("Playground: revert anchor set")
                                pending_broadcasts.append({
                                    "type": "checkpoint_result",
                                    "accepted": True,
                                })
                            elif tracking_active[0] and not tracking_paused[0]:
                                frame_idx = session.mark()
                                if frame_idx is None:
                                    print("Checkpoint ignored; no recorded frame yet")
                                    continue
                                task_info = task_manager.task_info if task_manager else None
                                saved_path = session.save_checkpoint(task_info=task_info)
                                if saved_path is not None:
                                    last_saved_state[0] = capture_sim_state()
                                    save_episode_initial_state()
                                    print(f"Saved checkpoint: {saved_path}")
                                    pending_broadcasts.append({
                                        "type": "checkpoint_result",
                                        "accepted": True,
                                    })
                            else:
                                print("Checkpoint ignored; tracking is not active")
                        elif cmd == "discard":
                            if playground_mode[0]:
                                # Playground: revert sim state only, no session machinery.
                                restore_sim_state(last_saved_state[0])
                                refresh_body_info_poses()
                                clear_tracking_state()
                                print("Playground: reverted to last anchor")
                            else:
                                revert_to_last_save_and_pause()
                        elif cmd == "reset":
                            # Autosave: capture whatever progress was made
                            # this episode before we throw the scene away, so
                            # a hard reset never silently drops work.
                            if (
                                tracking_active[0]
                                and not playground_mode[0]
                                and session.recording
                                and session.frame_count > 0
                            ):
                                try:
                                    session.mark()
                                    task_info = task_manager.task_info if task_manager else None
                                    saved_path = session.save_checkpoint(task_info=task_info)
                                    if saved_path is not None:
                                        print(f"Auto-saved checkpoint on reset: {saved_path}")
                                except Exception as exc:
                                    print(f"Auto-checkpoint on reset failed: {exc}")
                            reset_task_info = task_manager.task_info if task_manager else None
                            session.discard_to_checkpoint(task_info=reset_task_info)
                            session.finalize_checkpoint_episode()
                            chosen_mode = advance_task_and_pick_mode()
                            reset_scene(forced_mode=chosen_mode)
                            record_delay_until[0] = 0.0
                            refresh_snapshot_state()
                            # Stay paused after a hard reset so the close-up
                            # instruction panel can be read; tap B to start
                            # recording the new task.
                            tracking_active[0] = True
                            tracking_paused[0] = True
                        elif cmd in ("start", "toggle_pause"):
                            toggle_tracking_pause()
                        elif cmd == "pause":
                            pause_tracking()
                        elif cmd == "resume":
                            resume_tracking()
                        elif cmd == "exit_playground":
                            exit_playground()
                        elif cmd == "set_keyframe":
                            set_keyframe()
                        elif cmd in ("raise_table", "lower_table"):
                            step = (
                                TABLE_HEIGHT_STEP
                                if cmd == "raise_table"
                                else -TABLE_HEIGHT_STEP
                            )
                            table_height_offset[0] = clamp_height_offset(
                                table_height_offset[0] + step
                            )
                            save_table_height_offset(table_height_offset[0])
                            print(
                                f"Table height offset: {table_height_offset[0]:+.3f} m"
                            )
                        elif cmd == "next_task":
                            if task_manager:
                                task_manager.next_task()
                                refresh_snapshot_state()
                        elif cmd == "prev_task":
                            if task_manager:
                                task_manager.prev_task()
                                refresh_snapshot_state()
                        elif cmd == "repeat_task":
                            # Post-task modal: A press. Reset the same task
                            # (rerunning DR) without advancing task_manager.
                            # For scenes whose variant count is randomized at
                            # build time (e.g. hang_mugs picks 1-4 mugs), the
                            # only way to actually re-roll is a rebuild.
                            awaiting_post_task_choice[0] = False
                            broadcast_loading(True, "Resetting task")
                            try:
                                if builder_config.scene_type in REBUILD_ON_RESET_SCENES:
                                    rebuild_scene()
                                reset_scene()
                            finally:
                                broadcast_loading(False)
                            refresh_snapshot_state()
                            record_delay_until[0] = 0.0
                            begin_resume_cooldown()
                            tracking_paused[0] = False
                            print("Repeating same task with fresh randomization.")
                        elif cmd == "switch_task":
                            # Post-task modal: C press. Advance to the next
                            # task in the epoch and reset.
                            awaiting_post_task_choice[0] = False
                            chosen_mode = advance_task_and_pick_mode()
                            broadcast_loading(True, "Loading next task")
                            try:
                                if builder_config.scene_type in REBUILD_ON_RESET_SCENES:
                                    rebuild_scene()
                                reset_scene(forced_mode=chosen_mode)
                            finally:
                                broadcast_loading(False)
                            refresh_snapshot_state()
                            record_delay_until[0] = 0.0
                            tracking_paused[0] = False
                            print("Switched to next task.")
                        elif cmd in ("mark_task_complete", "mark_task_skipped"):
                            # A-hold finalizes the task as complete, C-hold as
                            # skipped/incomplete. Both SAVE the recording (the
                            # only difference is the `incomplete` flag in the
                            # episode metadata — nothing is discarded).
                            # Default: enter post-task modal so the operator
                            # picks A=repeat / C=switch.
                            # Dishrack: custom manager suppresses the modal and
                            # forces the rack→plate cycle; no operator choice.
                            skipped = cmd == "mark_task_skipped"
                            task_info = task_manager.task_info if task_manager else None
                            if task_info is None:
                                task_info = {}
                            saved_task_info = (
                                {**task_info, "incomplete": True}
                                if skipped else task_info
                            )
                            try:
                                save_recording_episode(
                                    task_info=saved_task_info,
                                    trim_to_last_marker=True)
                            except Exception as exc:
                                print(f"{cmd} save failed: {exc}")
                            result = "skipped" if skipped else "complete"
                            pending_broadcasts.append({
                                "type": "task_finished",
                                "result": result,
                            })
                            # Ask the task manager what to do next. If it
                            # returns a directive (forced-cycle managers
                            # like dishrack), apply it and skip the modal.
                            # If it returns None (default base manager),
                            # show the post-task modal.
                            directive = (task_manager.on_task_marked(result, state.model, state.data)
                                          if task_manager is not None else None)
                            if directive is not None:
                                if directive.redo_variants:
                                    broadcast_loading(True, "Rebuilding scene")
                                    try:
                                        rebuild_scene()
                                        reset_scene()
                                    finally:
                                        broadcast_loading(False)
                                elif directive.restore_snapshot and task_manager.snapshot is not None:
                                    broadcast_loading(True, "Loading next task")
                                    try:
                                        from mujoco_vr_teleop.dishrack_task_manager import restore_snapshot
                                        restore_snapshot(state.model, state.data, task_manager.snapshot)
                                        refresh_body_info_poses()
                                        clear_tracking_state()
                                        save_episode_initial_state()
                                        last_saved_state[0] = capture_sim_state()
                                    finally:
                                        broadcast_loading(False)
                                # else: keep current state (e.g. #1 complete).
                                # The task manager has advanced its pointer; rebind
                                # the snapshot runtime so captures land in the new
                                # task's directory rather than the previous one's.
                                refresh_snapshot_state()
                                record_delay_until[0] = 0.0
                                begin_resume_cooldown()
                                # Pause after a forced-cycle transition so the
                                # operator can read the new task instruction
                                # and orient before the next task starts.
                                tracking_paused[0] = True
                                awaiting_post_task_choice[0] = False
                                print(f"Forced cycle ({task_manager.scene}): "
                                      f"{result} → {directive.next_task_id} "
                                      f"(redo_variants={directive.redo_variants}, "
                                      f"restore_snapshot={directive.restore_snapshot})")
                            else:
                                awaiting_post_task_choice[0] = True
                                tracking_paused[0] = True
                                print(f"Task {result}; awaiting choice (A=repeat, C=switch)")
                        elif cmd == "snapshot":
                            rt = current_runtime[0]
                            if rt is None:
                                print("snapshot: no active task runtime")
                            else:
                                out_path = rt.next_snapshot_path()
                                try:
                                    from mujoco_vr_teleop.snapshot_runtime import render_snapshot
                                    render_snapshot(state.snapshot_model, state.data, out_path)
                                    count = len(rt.existing_snapshots())
                                    print(f"snapshot {count}: saved {out_path}")
                                    pending_broadcasts.append({
                                        "type": "snapshot_saved",
                                        "count": count,
                                    })
                                except Exception as exc:
                                    print(f"snapshot failed: {exc}")
                        _cmd_dt = (time.perf_counter() - _cmd_t0) * 1000.0
                        if _cmd_dt > 100.0:
                            print(f"[physics_loop] cmd {cmd!r} took {_cmd_dt:.0f}ms")
                    _mark_stage("after_commands")

                if tracking_active[0] and not tracking_paused[0]:
                    step_t0 = time.perf_counter()
                    native_stepper.step(state.model, state.data, 1)
                    _mark_stage("after_mj_step")
                    latency_samples.setdefault("sim-step", []).append(
                        (time.perf_counter() - step_t0) * 1000.0
                    )
                    sim_step += 1

                # In pedal mode the reset delay is visual only; otherwise it auto-starts recording.
                if not tracking_paused[0] and record_delay_until[0] > 0 and time.time() >= record_delay_until[0]:
                    record_delay_until[0] = 0.0
                    if not session.recording:
                        session.start_recording()

                if tracking_active[0] and not tracking_paused[0] and record_due_after_step:
                    frame_extra = arm_ik_diagnostics_frame()
                    frame_extra.update(wrist_hybrid_diagnostics_frame())
                    was_empty = session.frame_count == 0
                    session.record_frame(
                        state.model,
                        state.data,
                        sim_step,
                        extra=frame_extra,
                    )
                    # Frame 0 acts as an implicit checkpoint: same effect as
                    # pressing A immediately on start. Subsequent A presses
                    # add more checkpoints on top.
                    if was_empty and session.frame_count > 0:
                        session.mark()
                        task_info = task_manager.task_info if task_manager else None
                        session.save_checkpoint(task_info=task_info)
                    if wrist_hybrid_weld_mode:
                        label_frame_idx = session.frame_count - 1 - int(args.wrist_hybrid_label_horizon)
                        if label_frame_idx >= 0:
                            write_hybrid_arm_ctrl_label(label_frame_idx)
                    _mark_stage("after_record_frame")

                if stream_due_after_step or not tracking_active[0] or tracking_paused[0]:
                    frame_bytes[0] = build_frame_buffer()
                    _mark_stage("after_build_frame_buffer")

            elapsed = time.perf_counter() - t0
            if elapsed * 1000.0 > 250.0:
                # Slow iteration: dump per-stage deltas so we can see where time went.
                parts = []
                prev_t = _stage_marks[0][1]
                for label, ts in _stage_marks[1:]:
                    parts.append(f"{label}=+{(ts - prev_t) * 1000.0:.0f}ms")
                    prev_t = ts
                total_ms = (time.perf_counter() - t0) * 1000.0
                print(f"[physics_loop] SLOW ITER total={total_ms:.0f}ms  "
                      f"last_cmd={_last_cmd_label!r}  stages: " + ", ".join(parts))
            remaining = physics_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

            # Periodic latency summary (only active if at least one worker reports).
            if latency_samples and (time.perf_counter() - latency_last_print) >= latency_print_interval:
                parts = []
                for label in sorted(latency_samples):
                    samples = latency_samples[label]
                    if not samples:
                        continue
                    arr = np.asarray(samples, dtype=float)
                    parts.append(
                        f"{label}: n={len(arr)} p50={np.percentile(arr, 50):.1f}ms "
                        f"p95={np.percentile(arr, 95):.1f}ms max={arr.max():.1f}ms"
                    )
                if parts:
                    print(f"[diagnostics] sim_step={sim_step} | " + " | ".join(parts))
                latency_samples.clear()
                latency_last_print = time.perf_counter()

    # Boot into playground, paused: tap B to start forwarding mocap, hold B
    # 1.5s to exit playground and start recording.
    tracking_active[0] = True
    tracking_paused[0] = True

    # Start physics in background thread
    physics_thread = threading.Thread(target=physics_loop, daemon=True)
    physics_thread.start()

    # Global server hotkeys for foot-pedal style control
    pressed_server_keys: set[str] = set()
    pressed_server_key_times: dict[str, float] = {}
    # A real pedal hold lasts at most a few seconds. If a key has been "held"
    # far longer than that, its release event was almost certainly lost (Quest
    # USB hiccup), which would otherwise leave the key stuck in
    # pressed_server_keys forever and silently drop every future press of it.
    STUCK_KEY_TIMEOUT_S = 10.0

    def _evict_if_stuck(ch: str) -> None:
        pressed_at = pressed_server_key_times.get(ch)
        if pressed_at is None or (time.time() - pressed_at) < STUCK_KEY_TIMEOUT_S:
            return
        print(f"Server key {ch!r} stuck for "
              f"{time.time() - pressed_at:.0f}s; clearing stale press state")
        pressed_server_keys.discard(ch)
        pressed_server_key_times.pop(ch, None)
    # Monotonic press counter per pedal. The frontend uses the delta to detect
    # press events between status polls (so a tap shorter than the poll
    # interval still lights the pedal).
    pedal_press_counts: dict[str, int] = {"a": 0, "b": 0, "c": 0}

    def exit_playground():
        if not playground_mode[0]:
            return
        playground_mode[0] = False
        # Snap the robot + scene back to the keyframe so the user starts the
        # real episode from a clean state instead of wherever they left the
        # arm during practice. Leave tracking paused so the close-up
        # instruction panel can be read; the user holds A again to start
        # recording.
        reset_scene()
        tracking_active[0] = True
        tracking_paused[0] = True
        record_delay_until[0] = 0.0
        begin_resume_cooldown()
        print("Playground exit; awaiting hold-A to start recording")

    def _server_key_name(key):
        ch = getattr(key, "char", None)
        if ch is None:
            return None
        return ch.lower()

    # Long-press threshold. No action ever fires on a timer — press just
    # records the press time, and _on_server_key_release decides tap vs hold
    # from how long the key was actually held. This makes a long-press a
    # single, unambiguous event (no mid-hold firing, no double-fire, no
    # auto-advancing past a modal that opened mid-hold).
    LONG_PRESS_S = 1.5

    def _on_server_key_press(key):
        ch = _server_key_name(key)
        if ch is None:
            return
        # If this key is somehow still "pressed", its release was likely lost;
        # evict the stale state so this genuine press isn't dropped.
        _evict_if_stuck(ch)
        if ch in pressed_server_keys:
            return
        pressed_server_keys.add(ch)
        pressed_server_key_times[ch] = time.time()
        if ch in pedal_press_counts:
            pedal_press_counts[ch] += 1
        print(f"Server key {ch!r} pressed")

    def _on_server_key_release(key):
        ch = _server_key_name(key)
        if ch is None:
            return
        held_seconds = time.time() - pressed_server_key_times.pop(ch, time.time())
        was_height_adjust = ch in ("a", "c") and playground_mode[0]
        pressed_server_keys.discard(ch)
        is_hold = held_seconds >= LONG_PRESS_S

        # --- Playground: A/C adjust table height (driven by the height-adjust
        # loop while held; on release just persist). B tap = toggle_pause,
        # B hold = exit playground. ---
        if playground_mode[0]:
            if ch in ("a", "c"):
                try:
                    save_table_height_offset(table_height_offset[0])
                    print(f"Server key {ch!r} released ({held_seconds:.1f}s) -> "
                          f"persisted table height {table_height_offset[0]:+.3f} m")
                except Exception as exc:
                    print(f"Failed to persist table height offset: {exc}")
                return
            if ch == "b":
                if is_hold:
                    pending_commands.append("exit_playground")
                    print(f"Server key 'b' ({held_seconds:.1f}s) -> exit_playground")
                else:
                    pending_commands.append("toggle_pause")
                    print(f"Server key 'b' ({held_seconds:.1f}s) -> toggle_pause")
                return
            return

        # --- Post-task modal: A repeats the task, C advances to the next.
        # Tap or hold both count — the operator already finalized the task. ---
        if awaiting_post_task_choice[0]:
            if ch == "a":
                pending_commands.append("repeat_task")
                print(f"Server key 'a' ({held_seconds:.1f}s) -> repeat_task")
            elif ch == "c":
                pending_commands.append("switch_task")
                print(f"Server key 'c' ({held_seconds:.1f}s) -> switch_task")
            else:
                print(f"Server key {ch!r} ignored (awaiting post-task choice)")
            return

        # --- Normal recording controls (collector + choreographer) ---
        #   A tap  = save checkpoint (+ snapshot in choreographer)
        #   A hold = mark task complete
        #   B tap  = pause/resume tracking
        #   C tap  = revert to last save
        #   C hold = mark task skipped (incomplete)
        if ch == "a":
            if is_hold:
                pending_commands.append("mark_task_complete")
                print(f"Server key 'a' ({held_seconds:.1f}s) -> mark_task_complete")
            else:
                if args.mode == "choreographer":
                    pending_commands.append("snapshot")
                pending_commands.append("save_checkpoint")
                print(f"Server key 'a' ({held_seconds:.1f}s) -> save_checkpoint"
                      + (" (+snapshot)" if args.mode == "choreographer" else ""))
            return
        if ch == "b":
            pending_commands.append("toggle_pause")
            print(f"Server key 'b' ({held_seconds:.1f}s) -> toggle_pause")
            return
        if ch == "c":
            if is_hold:
                pending_commands.append("mark_task_skipped")
                print(f"Server key 'c' ({held_seconds:.1f}s) -> mark_task_skipped")
            else:
                pending_commands.append("discard")
                print(f"Server key 'c' ({held_seconds:.1f}s) -> discard")
            return

    from pynput import keyboard as pynput_keyboard

    server_key_listener = pynput_keyboard.Listener(
        on_press=_on_server_key_press,
        on_release=_on_server_key_release,
    )
    server_key_listener.daemon = True
    server_key_listener.start()
    print(
        "Server hotkeys: A=save checkpoint, B=pause/resume "
        "(hold 1.5s to continue/exit playground), C=discard/revert "
        "(hold 1.5s=reset)"
    )
    print(
        "In playground: hold A=lower table, hold C=raise table "
        f"(persists in {state_store.state_dir()})"
    )
    print("Started in playground mode. Hold 'b' for 1.5s to load a task.")

    # Background thread: while in playground and A or C is held, drift the
    # table height offset at 5 cm/sec. Persistence happens on key release.
    HEIGHT_ADJUST_RATE = 0.05  # m/sec
    HEIGHT_ADJUST_TICK = 0.05  # seconds

    def _height_adjust_loop():
        # In playground, hold A to lower the table, hold C to raise it. B is
        # reserved for the long-press continue/exit. When B is held, suppress
        # height adjustment so an incidentally-brushed A or C doesn't move
        # the table mid-gesture.
        #
        # This 20 Hz loop also sweeps for stuck keys: if a release event was
        # lost, _on_server_key_press only recovers on the *next* press of that
        # key — the sweep clears it even if the operator never touches it
        # again (which would otherwise keep a pedal "held" indefinitely).
        while True:
            time.sleep(HEIGHT_ADJUST_TICK)
            for ch_ in list(pressed_server_keys):
                _evict_if_stuck(ch_)
            if not playground_mode[0]:
                continue
            if "b" in pressed_server_keys:
                continue
            a_held = "a" in pressed_server_keys
            c_held = "c" in pressed_server_keys
            if not a_held and not c_held:
                continue
            delta = 0.0
            if a_held:
                delta -= HEIGHT_ADJUST_RATE * HEIGHT_ADJUST_TICK
            if c_held:
                delta += HEIGHT_ADJUST_RATE * HEIGHT_ADJUST_TICK
            table_height_offset[0] = clamp_height_offset(
                table_height_offset[0] + delta
            )

    height_adjust_thread = threading.Thread(
        target=_height_adjust_loop, daemon=True, name="height-adjust"
    )
    height_adjust_thread.start()

    frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    frontend_index = frontend_dist / "index.html"
    frontend_assets = frontend_dist / "assets"
    if not frontend_index.exists():
        raise SystemExit(
            "Frontend build not found. Run `npm install --prefix frontend && npm run build --prefix frontend`."
        )

    frontend_asset_files = [p for p in frontend_assets.glob("*") if p.is_file()]
    if not frontend_asset_files:
        raise SystemExit(
            "Frontend assets not found. Run `npm install --prefix frontend && npm run build --prefix frontend`."
        )

    # --- aiohttp web server ---
    app_web = web.Application()
    ws_clients: list[web.WebSocketResponse] = []
    active_input_ws: list[web.WebSocketResponse | None] = [None]
    ignored_input_ws_ids: set[int] = set()
    frontend_version = str(int(max(
        frontend_index.stat().st_mtime,
        *(path.stat().st_mtime for path in frontend_asset_files),
    )))

    async def handle_index(request):
        html = frontend_index.read_text()
        html = html.replace("/assets/main.css", f"/assets/main.css?v={frontend_version}")
        html = html.replace("/assets/main.js", f"/assets/main.js?v={frontend_version}")
        return web.Response(
            text=html,
            content_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    async def handle_client_config(request):
        return web.json_response({
            "handSkeleton": bool(args.hand_skeleton),
            "handScale": hand_scale,
            "scenePos": [float(x) for x in args.scene_pos],
            "vrPos": [float(x) for x in args.vr_pos],
            "vrTarget": [float(x) for x in args.vr_target],
            "tableHeightOffset": float(table_height_offset[0]),
        })

    async def handle_bodies(request):
        return web.json_response({
            str(k): {
                "file": v["file"],
                "is_fixed": bool(v["is_fixed"]),
                "name": v["name"],
                "position": v["position"],
                "quaternion": v["quaternion"],
                "scale": v.get("scale", 1.0),
                "opacity": v.get("opacity", 1.0),
            }
            for k, v in state.body_info.items()
        })

    async def handle_mocap(request):
        return web.json_response(state.mocap_info)

    async def handle_status(request):
        task_info = task_manager.task_info if task_manager else {}
        recorded_seconds = float(session.total_recorded_seconds)
        # Pedal state: how long each of A/B/C has been held (seconds) and
        # which long-press window each is in. The frontend uses this to draw
        # the pedal-fill animation during long-press.
        now_ts = time.time()
        pedal_state: dict[str, dict] = {}
        for letter in ("a", "b", "c"):
            held = letter in pressed_server_keys
            started = pressed_server_key_times.get(letter, now_ts) if held else now_ts
            pedal_state[letter] = {
                "held": held,
                "held_seconds": max(0.0, now_ts - started) if held else 0.0,
                "long_press_seconds": 1.5 if letter in ("b", "c") else 0.0,
                "press_count": pedal_press_counts[letter],
            }
        # State: "playground" before user enters collection, "delay" during
        # post-reset pause, "recording" while capturing, "idle" otherwise.
        if playground_mode[0]:
            ui_state = "playground"
        elif tracking_paused[0]:
            ui_state = "paused"
        elif record_delay_until[0] > 0 and time.time() < record_delay_until[0]:
            ui_state = "delay"
        elif session.recording:
            ui_state = "recording"
        else:
            ui_state = "idle"
        # Per-pedal labels for the HUD: the action each pedal performs in the
        # current state. Single source of truth for the on-screen hint so the
        # frontend doesn't drift from the actual key handlers. "Tap" vs "Hold"
        # is decided on key release from how long the key was held. A newline
        # splits the tap action (line 1) from the hold action (line 2); the
        # frontend renders each on its own line.
        b_pause_resume = "Pause" if ui_state == "recording" else "Resume"
        if ui_state == "playground":
            pedal_labels = {
                "a": "Lower table",
                "b": f"{b_pause_resume}\nHold: load task",
                "c": "Raise table",
            }
        elif awaiting_post_task_choice[0]:
            # Post-task modal: pick what to do with the next episode.
            pedal_labels = {
                "a": "Repeat task",
                "b": "—",
                "c": "Next task",
            }
        else:
            # Recording controls, shared by collector + choreographer.
            a_tap = "Snapshot + save" if args.mode == "choreographer" else "Save"
            pedal_labels = {
                "a": f"{a_tap}\nHold: task complete",
                "b": b_pause_resume,
                "c": "Revert\nHold: task skipped",
            }
        rt = current_runtime[0]
        snapshot_payload = rt.status_payload() if rt is not None else None

        return web.json_response({
            "session_id": state.session_id,
            "recording": session.recording,
            "tracking_active": tracking_active[0],
            "tracking_paused": tracking_paused[0],
            "playground": playground_mode[0],
            "state": ui_state,
            "frame_count": session.frame_count,
            "marker_count": session.total_marker_count,
            "seconds_since_marker": session.seconds_since_marker,
            "recorded_seconds": recorded_seconds,
            "recorded_minutes": recorded_seconds / 60.0,
            "episode": session.episode_counter,
            "task": task_info,
            "snapshots": snapshot_payload,
            "mode": args.mode,
            "awaiting_post_task_choice": bool(awaiting_post_task_choice[0]),
            "table_height_offset": float(table_height_offset[0]),
            "pedals": pedal_state,
            "pedal_labels": pedal_labels,
        })

    async def handle_mesh(request):
        filename = request.match_info["filename"]
        path = mesh_dir / filename
        if not path.exists():
            return web.Response(status=404)
        return web.FileResponse(path, headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Content-Type": "model/gltf-binary",
        })

    async def handle_snapshot(request):
        scene = request.match_info["scene"]
        task = request.match_info["task"]
        name = Path(request.match_info["name"]).name
        path = task_scenes_root / scene / task / name
        if not path.exists():
            return web.Response(status=404)
        return web.FileResponse(path, headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Content-Type": "image/png",
        })

    async def handle_asset(request):
        filename = Path(request.match_info["filename"]).name
        path = frontend_assets / filename
        if not path.exists():
            return web.Response(status=404)
        content_type = "application/javascript" if path.suffix == ".js" else "text/css"
        return web.FileResponse(path, headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Content-Type": content_type,
        })

    def accepts_vr_input(ws: web.WebSocketResponse) -> bool:
        active = active_input_ws[0]
        if active is None or active.closed:
            active_input_ws[0] = ws
            ignored_input_ws_ids.clear()
            if len(ws_clients) > 1:
                print("VR input source selected; mocap from secondary clients will be ignored")
            return True
        if active is ws:
            return True

        ws_id = id(ws)
        if ws_id not in ignored_input_ws_ids:
            ignored_input_ws_ids.add(ws_id)
            print("Ignoring mocap/trigger input from secondary VR client")
        return False

    ALLOWED_COMMANDS = (
        "reset",
        "start",
        "save",
        "save_checkpoint",
        "discard",
        "toggle_pause",
        # State-explicit alternatives to toggle_pause. Voice control uses
        # these so e.g. saying "resume" while already recording is a no-op
        # instead of accidentally pausing.
        "pause",
        "resume",
        "next_task",
        "prev_task",
        "set_keyframe",
        "exit_playground",
        "raise_table",
        "lower_table",
        # Choreographer + collector commands.
        "snapshot",
        "mark_task_complete",
        "mark_task_skipped",
        "repeat_task",
        "switch_task",
    )

    async def handle_command(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        cmd = body.get("command") if isinstance(body, dict) else None
        if cmd not in ALLOWED_COMMANDS:
            return web.json_response(
                {"error": "unknown_command", "command": cmd},
                status=400,
            )
        pending_commands.append(cmd)
        return web.json_response({"ok": True, "command": cmd})

    async def handle_ws(request):
        # NOTE: no aiohttp heartbeat/receive_timeout — the Quest browser does
        # not reliably auto-pong over the USB-tethered link, so a heartbeat
        # just churns disconnects on healthy clients. (A half-dead socket that
        # silently stalls is a separate, rarer problem; revisit with an
        # app-level staleness check rather than the transport heartbeat.)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients.append(ws)
        print(f"VR client connected ({len(ws_clients)} total)")

        last_recv_ts = time.perf_counter()
        msg_count = 0
        try:
            async for msg in ws:
                now = time.perf_counter()
                gap = (now - last_recv_ts) * 1000.0
                if gap > 500.0:
                    print(f"[ws_recv] gap={gap:.0f}ms between client→server messages (after {msg_count} msgs)")
                last_recv_ts = now
                msg_count += 1
                if msg.type == web.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    ptype = payload.get("type")
                    if ptype == "mocap":
                        if accepts_vr_input(ws):
                            payload["_server_recv_ts"] = time.perf_counter()
                            pending_mocap.append(payload)
                    elif ptype == "trigger":
                        if accepts_vr_input(ws):
                            pending_triggers.append(payload)
                    elif ptype == "event_marker":
                        session.add_event_marker(payload.get("name", "unknown"))
                    else:
                        print(f"[ws_recv] unknown msg type: {ptype!r}")
                elif msg.type in (web.WSMsgType.PING, web.WSMsgType.PONG):
                    pass
                else:
                    print(f"[ws_recv] unexpected ws msg type: {msg.type}")
        except Exception as exc:
            # receive_timeout / heartbeat failure / transport error — log and
            # fall through to cleanup so a reconnecting client can take over.
            print(f"VR client websocket ended: {type(exc).__name__}: {exc}  (after {msg_count} msgs)")
        finally:
            if active_input_ws[0] is ws:
                active_input_ws[0] = None
                ignored_input_ws_ids.clear()
            if ws in ws_clients:
                ws_clients.remove(ws)
            print(f"VR client disconnected ({len(ws_clients)} remaining)")

        return ws

    async def stream_loop():
        """Broadcast transforms to all connected WebSocket clients."""
        dt = 1.0 / args.stream_rate
        _stream_last_send_ts = time.perf_counter()
        while True:
            if ws_clients:
                # Drain control broadcasts FIRST and without the lock so
                # messages like {type: loading, active: true} can be sent
                # even while the physics thread is mid-rebuild holding the
                # lock for ~1s.
                _t_iter = time.perf_counter()
                _iter_gap_ms = (_t_iter - _stream_last_send_ts) * 1000.0
                # Track WHERE in this iteration time is being spent.
                broadcasts = pending_broadcasts[:]
                pending_broadcasts.clear()
                _t_bcast = time.perf_counter()
                bcast_per_client_ms: list[float] = []
                for ws in list(ws_clients):
                    if not broadcasts:
                        continue
                    _t_one = time.perf_counter()
                    for msg in broadcasts:
                        try:
                            await asyncio.wait_for(ws.send_json(msg), timeout=0.1)
                        except asyncio.TimeoutError:
                            print(f"[stream_loop] broadcast send_json TIMEOUT  msg.type={msg.get('type')!r}")
                        except Exception as exc:
                            print(f"[stream_loop] broadcast send_json error: {type(exc).__name__}: {exc}")
                    bcast_per_client_ms.append((time.perf_counter() - _t_one) * 1000.0)
                _bcast_ms = (time.perf_counter() - _t_bcast) * 1000.0
                # Frame data needs the lock; this may briefly block during a
                # rebuild but that's fine (we just skip a frame).
                _t_lock = time.perf_counter()
                got_lock = lock.acquire(timeout=0.1)
                _lock_wait_ms = (time.perf_counter() - _t_lock) * 1000.0
                if got_lock:
                    try:
                        data_bytes = frame_bytes[0]
                        _frame_len = len(data_bytes) if data_bytes is not None else 0
                    finally:
                        lock.release()
                    _t_send = time.perf_counter()
                    # Send to all clients concurrently with a per-send timeout.
                    # A stuck client (full TCP buffer, paused tab) used to block
                    # the for-loop on its `await`, freezing every other client
                    # for as long as the slow one took to recover.
                    send_durations: dict[int, float] = {}
                    send_outcomes: dict[int, str] = {}
                    async def _send_one(w):
                        _ts = time.perf_counter()
                        try:
                            # No timeout: a slow client (~hundreds of ms of
                            # WiFi/OS jitter, or a server-side 500ms snapshot
                            # render that just blocked the lock) should not get
                            # disconnected — the disconnect cascades into a
                            # full asset reload that takes many seconds. Just
                            # let send_bytes await as long as it needs.
                            await w.send_bytes(data_bytes)
                            send_outcomes[id(w)] = "ok"
                        except Exception as exc:
                            send_outcomes[id(w)] = f"err:{type(exc).__name__}"
                            print(f"[stream_loop] send_bytes error: {type(exc).__name__}: {exc}")
                        send_durations[id(w)] = (time.perf_counter() - _ts) * 1000.0
                    clients_snapshot = list(ws_clients)
                    await asyncio.gather(*(_send_one(w) for w in clients_snapshot))
                    _send_ms = (time.perf_counter() - _t_send) * 1000.0
                    _stream_last_send_ts = _t_send
                    if (_iter_gap_ms > 250.0 or _send_ms > 100.0 or _bcast_ms > 100.0
                            or _lock_wait_ms > 25.0 or any(d > 25 for d in send_durations.values())):
                        per_client = ", ".join(
                            f"{send_durations[id(c)]:.0f}ms[{send_outcomes[id(c)]}]"
                            for c in clients_snapshot
                        )
                        print(f"[stream_loop] gap={_iter_gap_ms:.0f}ms  "
                              f"bcast={_bcast_ms:.0f}ms (n={len(broadcasts)})  "
                              f"lock_wait={_lock_wait_ms:.0f}ms  "
                              f"send_total={_send_ms:.0f}ms  "
                              f"frame_len={_frame_len}  "
                              f"clients={len(clients_snapshot)} [{per_client}]")
                else:
                    print(f"[stream_loop] failed to acquire lock within 100ms (lock_wait={_lock_wait_ms:.0f}ms) — "
                          f"physics thread is blocked on something")
            await asyncio.sleep(dt)

    async def on_startup(app):
        asyncio.create_task(stream_loop())

    app_web.router.add_get("/", handle_index)
    app_web.router.add_get("/api/bodies", handle_bodies)
    app_web.router.add_get("/api/client-config", handle_client_config)
    app_web.router.add_get("/api/mocap", handle_mocap)
    app_web.router.add_get("/api/status", handle_status)
    app_web.router.add_post("/api/command", handle_command)
    app_web.router.add_get("/meshes/{filename}", handle_mesh)
    app_web.router.add_get(
        "/snapshots/{scene}/{task}/{name}", handle_snapshot)
    app_web.router.add_get("/assets/{filename}", handle_asset)
    app_web.router.add_get("/ws", handle_ws)

    app_web.on_startup.append(on_startup)

    async def on_cleanup(app):
        for pool in ik_workers.values():
            for worker in pool["workers"]:
                try:
                    worker["conn"].send({"cmd": "close"})
                except Exception:
                    pass
                worker["proc"].join(timeout=0.5)

    app_web.on_cleanup.append(on_cleanup)

    print(f"\nVR Streaming Server: http://0.0.0.0:{args.port}")
    print(f"Open in VR headset browser to connect")

    # Run the aiohttp server on its own thread with a fresh asyncio loop so the
    # main thread is free to host the Textual TUI (which needs to own stdin).
    web_loop_holder: dict = {}
    web_thread_ready = threading.Event()

    def _run_web():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        web_loop_holder["loop"] = loop
        runner = web.AppRunner(app_web)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, host="0.0.0.0", port=args.port)
        loop.run_until_complete(site.start())
        web_loop_holder["runner"] = runner
        web_thread_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(runner.cleanup())
            loop.close()

    web_thread = threading.Thread(target=_run_web, daemon=True, name="aiohttp")
    web_thread.start()
    web_thread_ready.wait(timeout=10.0)

    # Tee stdout / stderr into <session_dir>/server.log and server.stderr.log
    # once the session directory exists. The supervisor TUI tails these files
    # for its log pane.
    def _attach_log_files(dir_path: Path) -> None:
        try:
            stdout_tee.set_log_file(dir_path / "server.log")
            stderr_tee.set_log_file(dir_path / "server.stderr.log")
        except Exception as exc:
            print(f"Failed to attach log files to {dir_path}: {exc}")

    session.on_dir_created.append(_attach_log_files)

    # Install a SIGTERM handler that converts the signal into KeyboardInterrupt
    # so the join() below unblocks. (Default SIGTERM kills without cleanup.)
    import signal as _signal

    def _sigterm_to_interrupt(signum, _frame):
        print(f"\nReceived signal {signum}; shutting down...")
        raise KeyboardInterrupt()

    try:
        _signal.signal(_signal.SIGTERM, _sigterm_to_interrupt)
    except (ValueError, AttributeError):
        pass  # only main-thread can install handlers; ignore if not possible

    # Block on the aiohttp thread; SIGTERM/SIGINT bubbles up via KeyboardInterrupt.
    try:
        web_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        # If the streamer is dying mid-recording, save what we have with an
        # `interrupted: True` flag in the task metadata so dataset users can
        # filter these out (or treat them differently).
        try:
            if session.recording and session.frame_count > 0:
                task_info = task_manager.task_info if task_manager else {}
                task_info = {**(task_info or {}), "interrupted": True}
                save_recording_episode(
                    task_info=task_info, trim_to_last_marker=False)
                print("Flushed in-flight recording with interrupted=True")
        except Exception as exc:
            print(f"Could not flush in-flight recording on shutdown: {exc}")
        loop = web_loop_holder.get("loop")
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        web_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
