from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import attrs
import mujoco
import numpy as np


SceneType = Literal[
    "jenga",
    "bottles_in_bin",
    "cleanup",
    "nut_and_bolt",
    "cup_stack",
    "cup_ball",
    "cups_and_mugs",
    "dishrack",
    "hang_mugs",
    "shapes",
    "spell_and_stow",
    "tea_time",
]
ArmType = Literal["yam", "yam_pro", "yam_ultra"]
PhysicalArmType = Literal["yam", "yam_pro", "yam_ultra"]
EndEffector = Literal["sharpa"]
ActuatorMode = Literal["position", "torque"]
FingerControl = Literal["ik"]
WristControl = Literal["ik", "hybrid_weld"]

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
GENERATED = EXAMPLES / "generated"
BASE_SCENES = EXAMPLES / "base_scenes"
TASK_SCENES = EXAMPLES / "task_scenes"
ASSETS = EXAMPLES / "assets"
YAM_VARIANTS = ASSETS / "yam_variants"
END_EFFECTORS = ROOT / "mujoco_vr_teleop" / "end_effectors"

BASE_SCENE_XML = BASE_SCENES / "workcell.xml"

# Side channel for variant_pools.attach_variant_pools: it returns metadata
# describing the slot/variant layout, and build_scene reads it back here after
# the spec has been composed. MjSpec doesn't accept arbitrary attributes, so a
# module-level dict is the simplest carrier. Always overwritten per compose call.
_LAST_VARIANT_POOL_METADATA: dict = {}
# Also stash the pinned selected_variants for the most-recently-built scene so
# task helpers (which load the DR JSON from disk, where pins don't live) can
# honor them. None when no pin was requested.
_LAST_SELECTED_VARIANTS: dict[str, int] | None = None


def _stable_short_hash(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:10]


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def _task_scenes() -> tuple[SceneType, ...]:
    # Single source of truth for the SceneType literal members (kept in sync
    # with the literal definition above).
    return (
        "jenga",
        "bottles_in_bin",
        "cleanup",
        "nut_and_bolt",
        "cup_stack",
        "cup_ball",
        "cups_and_mugs",
        "dishrack",
        "hang_mugs",
        "shapes",
        "spell_and_stow",
        "tea_time",
    )


TASK_SOURCE_XML: dict[SceneType, Path] = {
    name: TASK_SCENES / f"{name}.xml" for name in _task_scenes()
}

ARM_XML = {
    "yam": YAM_VARIANTS / "prebuilt_yam" / "bare.xml",
    "yam_pro": YAM_VARIANTS / "prebuilt_yam" / "bare.xml",
    "yam_ultra": YAM_VARIANTS / "prebuilt_yam" / "bare.xml",
}
ARM_XML_SAFE = {
    "yam": YAM_VARIANTS / "prebuilt_yam" / "bare_safe.xml",
    "yam_pro": YAM_VARIANTS / "prebuilt_yam" / "bare_safe.xml",
    "yam_ultra": YAM_VARIANTS / "prebuilt_yam" / "bare_safe.xml",
}

def sharpa_xml(side: str, quality: str = "standard") -> Path:
    """Path to the Sharpa hand XML for the requested quality.

    Three versions are checked in under mujoco_vr_teleop/end_effectors/:
      * ``sharpa_{side}_standard.xml`` — small STL meshes, fast GLB export.
      * ``sharpa_{side}_good.xml``     — USDZ-derived OBJ meshes decimated
                                          ~50%; mid-weight middle ground.
      * ``sharpa_{side}_high.xml``     — high-poly OBJ meshes; nicer in VR
                                          but much slower to export GLBs.
    """
    if quality not in ("standard", "good", "high"):
        raise ValueError(
            "sharpa_visual_mesh_quality must be 'standard', 'good', or 'high', "
            f"got {quality!r}"
        )
    return END_EFFECTORS / f"sharpa_{side}_{quality}.xml"


SHARPA_MOUNT = {
    "yam": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    "yam_pro": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    "yam_ultra": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
}
SHARPA_WRIST_WELD_SOLREF = (0.03, 1.0)
SHARPA_WRIST_WELD_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
ARM_JOINT_RANGES = {
    1: (-2.61799, 3.05433),
    2: (0.0, 3.66519),
    3: (0.0, 3.66519),
    4: (-1.5708, 1.5708),
    5: (-1.5708, 1.5708),
    6: (-2.0944, 2.0944),
}
# Tighter, "safe" ranges used when BuildSceneConfig.safe_joint_ranges=True.
# Keeps the arm from reaching behind the table or wrist overextension.
# J3 / J4 limits derived from the right-arm pose at the last saved checkpoint.
# NOTE: when changing J2/J3 here, also update _YAM_SAFE_JOINT_LIMITS in
# i2rt_official/i2rt/robots/get_robot.py.
ARM_JOINT_RANGES_SAFE = {
    **ARM_JOINT_RANGES,
    1: (-0.7854, 1.0472),  # -45° to +60°
    2: (0.0, 3.14159),     # 0° to 180°
    3: (0.0, 2.35619),     # 0° to 135°
    4: (-1.5708, 1.2229),  # -90° to +70° (checkpoint right-arm-joint4)
}
ARM_SIDES = {
    "right": ("right-arm", "right-arm-", -0.31),
    "left": ("left-arm", "left-arm-", 0.31),
}

MocapSpec = tuple[str, tuple[float, float, float], tuple[float, float, float, float]]

SHARPA_MOCAPS: tuple[MocapSpec, ...] = (
    ("right-wrist", (0.23, -0.15, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-thumb-tip", (0.32, -0.28, 1.017), (0.2706, 0.6533, 0.2706, -0.6533)),
    ("right-thumb-phalanx-distal", (0.301, -0.259, 1.016), (0.2706, 0.6533, 0.2706, -0.6533)),
    ("right-thumb-phalanx-proximal", (0.274, -0.231, 1.016), (0.2706, 0.6533, 0.2706, -0.6533)),
    ("right-index-finger-tip", (0.41, -0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-index-finger-phalanx-distal", (0.383, -0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-index-finger-phalanx-intermediate", (0.3515, -0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-index-finger-phalanx-proximal", (0.305, -0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-middle-finger-tip", (0.415, -0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-middle-finger-phalanx-distal", (0.387, -0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-middle-finger-phalanx-intermediate", (0.3555, -0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-middle-finger-phalanx-proximal", (0.309, -0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-ring-finger-tip", (0.409, -0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-ring-finger-phalanx-distal", (0.381, -0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-ring-finger-phalanx-intermediate", (0.35, -0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-ring-finger-phalanx-proximal", (0.303, -0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-pinky-finger-tip", (0.403, -0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-pinky-finger-phalanx-distal", (0.3755, -0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-pinky-finger-phalanx-intermediate", (0.3435, -0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("right-pinky-finger-phalanx-proximal", (0.2965, -0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-wrist", (-0.19, 0.15, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-thumb-tip", (-0.1, 0.28, 1.017), (0.6533, -0.2706, -0.6533, -0.2706)),
    ("left-thumb-phalanx-distal", (-0.12, 0.253, 1.016), (0.2706, 0.6533, 0.2706, -0.6533)),
    ("left-thumb-phalanx-proximal", (-0.147, 0.225, 1.016), (0.2706, 0.6533, 0.2706, -0.6533)),
    ("left-index-finger-tip", (-0.01, 0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-index-finger-phalanx-distal", (-0.037, 0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-index-finger-phalanx-intermediate", (-0.0685, 0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-index-finger-phalanx-proximal", (-0.115, 0.18, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-middle-finger-tip", (-0.005, 0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-middle-finger-phalanx-distal", (-0.033, 0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-middle-finger-phalanx-intermediate", (-0.0645, 0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-middle-finger-phalanx-proximal", (-0.111, 0.16, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-ring-finger-tip", (-0.011, 0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-ring-finger-phalanx-distal", (-0.039, 0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-ring-finger-phalanx-intermediate", (-0.07, 0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-ring-finger-phalanx-proximal", (-0.117, 0.14, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-pinky-finger-tip", (-0.017, 0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-pinky-finger-phalanx-distal", (-0.0445, 0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-pinky-finger-phalanx-intermediate", (-0.0765, 0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
    ("left-pinky-finger-phalanx-proximal", (-0.1235, 0.12, 1.0), (0.0, 0.7071, 0.7071, 0.0)),
)

SHARPA_WRIST_MOCAP_SITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("right-wrist", "sharpa_palm_site"),
    ("left-wrist", "left_sharpa_palm_site"),
)


@attrs.define(frozen=True)
class BuildSceneConfig:
    scene_type: SceneType
    arm_type: ArmType = "yam_ultra"
    end_effector: EndEffector = "sharpa"
    actuator_mode: ActuatorMode = "position"
    hand_scale: float = 0.9
    ik_ori_weight: float = 5.0
    finger_control: FingerControl = "ik"
    wrist_control: WristControl = "ik"
    walls: bool = True
    wall_opacity: float = 0.2
    safe_joint_ranges: bool = False
    # Vertical offset (meters) of the play-table top relative to where the
    # arms are mounted. The table and everything resting on it (task subtree)
    # shift by this amount; the arms stay put. Negative lowers the table.
    # Default 0.02921 m = 1.15 in, matching the physical workcell shim.
    table_height_offset: float = 0.02921
    # Pin specific variant selections per pool. When set, the runtime DR
    # `variant_select` op selects exactly the listed id for each pool name
    # instead of sampling from the pool. Used by replay to reconstruct an
    # episode's exact scene; left None at recording time so DR samples freely.
    # Maps pool name (e.g. "dishrack_plate_0") -> variant id (e.g. 7).
    selected_variants: dict[str, int] | None = None
    # Sharpa hand visual mesh quality. "standard" uses the small original
    # STL meshes (~thousands of verts per link, fast GLB export).
    # "good" uses the USDZ-derived OBJs decimated ~50%
    # (examples/assets/sharpa_hand/usd/obj_good/); a mid-weight middle ground.
    # "high" uses the full-res OBJs under examples/assets/sharpa_hand/usd/obj/
    # (~30k verts on the palm; nicer-looking but slow GLB export).
    sharpa_visual_mesh_quality: str = "standard"

    @classmethod
    def from_dict(cls, data: dict) -> "BuildSceneConfig":
        fields = attrs.fields_dict(cls)
        return cls(**{key: value for key, value in data.items() if key in fields})


@attrs.define(frozen=True)
class GeneratedScene:
    model: mujoco.MjModel
    xml_path: Path
    config_path: Path
    streamer_config: dict
    base_xml: Path
    source_xml: Path


def bare_arm(
    arm_type: PhysicalArmType,
    arm_gravcomp: float,
    safe_joint_ranges: bool = False,
) -> mujoco.MjSpec:
    source = ARM_XML_SAFE[arm_type] if safe_joint_ranges else ARM_XML[arm_type]
    arm = mujoco.MjSpec.from_file(str(source))
    for name in ("link_1", "link_2", "link_3", "link_4", "link_5", "link_6"):
        arm.body(name).gravcomp = arm_gravcomp
    return arm


def sharpa_hand(side: str, end_effector_gravcomp: float,
                  quality: str = "standard") -> mujoco.MjSpec:
    hand = mujoco.MjSpec.from_file(str(sharpa_xml(side, quality)))
    hand.default.name = f"{side}_sharpa"
    for body in hand.bodies:
        if body.name != "world":
            body.gravcomp = end_effector_gravcomp
    for actuator in hand.actuators:
        joint = hand.joint(actuator.target)
        actuator.ctrllimited = True
        actuator.ctrlrange = joint.range
    return hand


def add_position_actuator(
    spec: mujoco.MjSpec,
    name: str,
    target: str,
    kp: float,
    kv: float,
    force: tuple[float, float] | None = None,
    ctrlrange: tuple[float, float] | None = None,
) -> None:
    spec.add_actuator(
        name=name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=target,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[kp, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -kp, -kv, 0, 0, 0, 0, 0, 0, 0],
        forcelimited=force is not None,
        forcerange=force,
        ctrllimited=ctrlrange is not None,
        ctrlrange=ctrlrange,
        inheritrange=1.0 if ctrlrange is None else 0.0,
    )


def add_torque_actuator(
    spec: mujoco.MjSpec,
    name: str,
    target: str,
    ctrlrange: tuple[float, float],
) -> None:
    spec.add_actuator(
        name=name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=target,
        gear=[1, 0, 0, 0, 0, 0],
        ctrllimited=True,
        ctrlrange=ctrlrange,
    )


def add_arm_actuator(
    spec: mujoco.MjSpec,
    actuator_mode: ActuatorMode,
    name: str,
    target: str,
    kp: float,
    kv: float,
    force: tuple[float, float],
    ctrlrange: tuple[float, float],
) -> None:
    if actuator_mode == "position":
        add_position_actuator(
            spec,
            name=name,
            target=target,
            kp=kp,
            kv=kv,
            force=force,
            ctrlrange=ctrlrange,
        )
    else:
        add_torque_actuator(spec, name=name, target=target, ctrlrange=force)


def add_mocap(spec: mujoco.MjSpec, name: str, pos: tuple[float, float, float], quat: tuple[float, float, float, float]) -> None:
    body = spec.worldbody.add_body(name=name, mocap=True, pos=pos, quat=quat)
    body.add_site(
        name=f"{name}-site-mocap",
        size=(0.002, 0.002, 0.002),
        rgba=(0.658, 0.411, 0.75, 1),
        group=5,
    )


def _set_cup_sliding_friction(spec: mujoco.MjSpec, sliding: float) -> None:
    """Set the sliding-friction component of every cup collision geom.

    Cup collision geoms live under ``cup_slot_<letter>__*`` bodies in group 3.
    Only the first (sliding) friction component is changed; torsional and
    rolling are left as-is. Used to make the cup_ball shuffle task's cups
    slide cleanly on the table.
    """
    for body in spec.bodies:
        if "cup_slot_" not in (body.name or ""):
            continue
        for geom in body.geoms:
            if int(geom.group) != 3:
                continue
            fr = list(geom.friction)
            fr[0] = float(sliding)
            geom.friction = fr


def add_cage_walls(spec: mujoco.MjSpec, opacity: float = 0.2) -> None:
    alpha = float(max(0.0, min(1.0, opacity)))
    walls = (
        ("back_wall", (0.005, 0.61, 1.0), (1.2144, 0.0, 1.0)),
        ("left_wall", (0.4572, 0.005, 1.0), (0.7522, 0.61, 1.0)),
        ("right_wall", (0.4572, 0.005, 1.0), (0.7522, -0.61, 1.0)),
    )
    for name, size, pos in walls:
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            group=2,
            rgba=(0.88, 0.88, 0.88, alpha),
        )


def enable_sleeping_islands(spec: mujoco.MjSpec) -> None:
    spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)


def stabilize_contacts(spec: mujoco.MjSpec, scene_type: str | None = None) -> None:
    # Elliptic friction cone settles small stacked objects far more stably
    # than the default pyramidal cone (which jitters light boxes). The noslip
    # solver pass removes residual contact-patch micro-sliding, so low-friction
    # blocks can stack without vibrating.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 1


def add_yam_mount_frame(
    spec: mujoco.MjSpec,
    root_name: str,
    pos: tuple[float, float, float],
) -> mujoco.MjsFrame:
    return spec.worldbody.add_frame(name=f"{root_name}_frame", pos=pos)


def add_robot(
    spec: mujoco.MjSpec,
    arm_type: PhysicalArmType,
    end_effector: EndEffector,
    actuator_mode: ActuatorMode,
    arm_gravcomp: float,
    end_effector_gravcomp: float,
    safe_joint_ranges: bool = False,
    sharpa_visual_mesh_quality: str = "standard",
) -> None:
    joint_ranges = ARM_JOINT_RANGES_SAFE if safe_joint_ranges else ARM_JOINT_RANGES
    arms = (
        ("left", *ARM_SIDES["left"]),
        ("right", *ARM_SIDES["right"]),
    )
    for side, root_name, prefix, y in arms:
        arm = bare_arm(arm_type, arm_gravcomp, safe_joint_ranges=safe_joint_ranges)
        frame = add_yam_mount_frame(spec, root_name, (0.2525, y, 0.76))
        spec.attach(
            arm,
            prefix=prefix,
            frame=frame,
        )
        spec.body(f"{prefix}arm_model").name = root_name

    for side, _, prefix, _ in arms:
        for joint in range(1, 7):
            strong_joint = joint <= 3
            wrist_yaw_joint = joint == 4
            joint_name = f"{prefix}joint{joint}"
            add_arm_actuator(
                spec,
                actuator_mode,
                name=joint_name,
                target=joint_name,
                kp=40.0 if strong_joint else 20.0 if wrist_yaw_joint else 10.0,
                kv=2.5 if strong_joint else 0.5 if wrist_yaw_joint else 1.0,
                force=(-28.0, 28.0) if strong_joint else (-10.0, 10.0),
                ctrlrange=joint_ranges[joint],
            )

    for side, (_, prefix, _) in ARM_SIDES.items():
        hand_pos, hand_quat = SHARPA_MOUNT[arm_type]
        hand_frame = spec.body(f"{prefix}link_6").add_frame(
            name=f"{side}_sharpa_frame",
            pos=hand_pos,
            quat=hand_quat,
        )
        hand = sharpa_hand(side, end_effector_gravcomp,
                             quality=sharpa_visual_mesh_quality)
        spec.attach(hand, prefix="", frame=hand_frame)

    for mocap in SHARPA_MOCAPS:
        add_mocap(spec, *mocap)


def build_joint_ik_scene(
    arm_type: PhysicalArmType,
    side: Literal["right", "left"],
    xml_path: Path,
    gravcomp: float = 0.0,
) -> None:
    root_name, prefix, y = ARM_SIDES[side]
    spec = mujoco.MjSpec()
    spec.compiler.meshdir = "../assets/"
    spec.compiler.texturedir = "../assets/"

    arm = bare_arm(arm_type, arm_gravcomp=0.0)
    frame = add_yam_mount_frame(spec, root_name, (0.2525, y, 0.76))
    spec.attach(arm, prefix=prefix, frame=frame)
    spec.body(f"{prefix}arm_model").name = root_name

    hand_pos, hand_quat = SHARPA_MOUNT[arm_type]
    hand_frame = spec.body(f"{prefix}link_6").add_frame(
        name=f"{side}_sharpa_frame",
        pos=hand_pos,
        quat=hand_quat,
    )
    spec.attach(sharpa_hand(side, end_effector_gravcomp=gravcomp), prefix="", frame=hand_frame)

    _atomic_write_text(xml_path, spec.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(xml_path))


def add_sharpa_wrist_mocap_welds(spec: mujoco.MjSpec) -> None:
    for mocap_name, site_name in SHARPA_WRIST_MOCAP_SITE_PAIRS:
        spec.add_equality(
            name=f"{mocap_name.replace('-', '_')}_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{mocap_name}-site-mocap",
            name2=site_name,
            solref=SHARPA_WRIST_WELD_SOLREF,
            solimp=SHARPA_WRIST_WELD_SOLIMP,
        )


def align_wrist_mocaps_to_sharpa_palms(spec: mujoco.MjSpec) -> None:
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for mocap_name, site_name in SHARPA_WRIST_MOCAP_SITE_PAIRS:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            continue
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, data.site_xmat[sid].reshape(9))
        body = spec.body(mocap_name)
        body.pos = data.site_xpos[sid].copy()
        body.quat = quat


def domain_randomization_config(config: BuildSceneConfig) -> dict:
    # Domain randomization is done entirely in Python — each scene's task
    # `reset` (e.g. scatter_dishrack, scatter_bottles_in_bin) owns scaling
    # and placement. There are no DR JSON files; the streamer's JSON-driven
    # DR path stays disabled.
    return {"enabled": False, "tree": {"actions": [], "children": []}}


def validate_config(config: BuildSceneConfig) -> None:
    if config.wrist_control == "hybrid_weld":
        if config.arm_type not in ARM_XML or config.end_effector != "sharpa":
            raise ValueError("wrist weld control requires a physical YAM arm with a Sharpa end effector")


def streamer_config(config: BuildSceneConfig) -> dict:
    validate_config(config)
    return {
        "ik_mode": True,
        "ik_ori_weight": config.ik_ori_weight,
        "ik_smoothing": 0.95,
        "domain_randomization": domain_randomization_config(config),
    }


def prepare_generated_dir() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)


def gravcomp_label(value: float) -> str:
    text = f"{value:.1f}" if value.is_integer() else f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def compose_scene_spec(
    config: BuildSceneConfig, *, source_xml_override: Path | None = None
) -> mujoco.MjSpec:
    validate_config(config)
    # source_xml_override lets a caller substitute the task XML (e.g. replaying
    # an old recording against a legacy scene revision); the rest of the build
    # — robot, walls, variant pools — is identical.
    source_xml = source_xml_override or TASK_SOURCE_XML[config.scene_type]
    spec = mujoco.MjSpec.from_file(str(BASE_SCENE_XML))
    task = mujoco.MjSpec.from_file(str(source_xml))
    task.default.name = f"{config.scene_type}_task"
    dz = float(config.table_height_offset)
    task_frame = spec.worldbody.add_frame(
        name=f"{config.scene_type}_task_frame",
        pos=[0.0, 0.0, dz],
    )
    spec.attach(task, prefix="", frame=task_frame)
    if dz != 0.0:
        for geom_name in ("table_visual", "table_plane"):
            geom = spec.geom(geom_name)
            new_pos = np.asarray(geom.pos, dtype=float).copy()
            new_pos[2] += dz
            geom.pos = new_pos
    # Preload variant pools (e.g. dish-rack + plate variants) as slot bodies.
    # The runtime `variant_select` DR op picks which slot is active each reset.
    from mujoco_vr_teleop.variant_pools import attach_chosen_variants

    chosen_variants = attach_chosen_variants(
        spec, config.scene_type,
        selected_variants=config.selected_variants,
    )
    _LAST_VARIANT_POOL_METADATA.clear()
    # Stash the chosen variants in the side channel so task helpers and
    # the recorder can read them back without rebuilding the scene.
    _LAST_VARIANT_POOL_METADATA["chosen_variants"] = dict(chosen_variants)
    # The cup scenes also have non-deterministic build inputs (cup family,
    # task, count, ball presence, cup id, scale) that chosen_variants alone
    # cannot reproduce — stash the resolved per-scene build dict so replay can
    # pin it exactly. See variant_pools.
    if config.scene_type == "cup_stack":
        from mujoco_vr_teleop.variant_pools import CUP_STACK_BUILD
        _LAST_VARIANT_POOL_METADATA["cup_build"] = dict(CUP_STACK_BUILD)
    elif config.scene_type == "cup_ball":
        from mujoco_vr_teleop.variant_pools import CUP_BALL_BUILD
        _LAST_VARIANT_POOL_METADATA["cup_build"] = dict(CUP_BALL_BUILD)
        # The shuffle task wants low-friction cups so they slide cleanly on the
        # table when the operator shuffles them. Drop the sliding-friction
        # component of every cup collision geom for this build only.
        if CUP_BALL_BUILD.get("task") == "shuffle_ball":
            _set_cup_sliding_friction(spec, 0.2)
    global _LAST_SELECTED_VARIANTS
    _LAST_SELECTED_VARIANTS = dict(chosen_variants)
    enable_sleeping_islands(spec)
    stabilize_contacts(spec, config.scene_type)
    if config.walls:
        add_cage_walls(spec, opacity=config.wall_opacity)
    add_robot(
        spec,
        config.arm_type,
        config.end_effector,
        config.actuator_mode,
        arm_gravcomp=1.0,
        end_effector_gravcomp=1.0,
        safe_joint_ranges=config.safe_joint_ranges,
        sharpa_visual_mesh_quality=config.sharpa_visual_mesh_quality,
    )
    if config.wrist_control == "hybrid_weld":
        add_sharpa_wrist_mocap_welds(spec)
        align_wrist_mocaps_to_sharpa_palms(spec)
    _dedupe_skybox(spec)
    return spec


def _dedupe_skybox(spec: mujoco.MjSpec) -> None:
    """Drop all but the first skybox texture.

    The base workcell and every task XML each declare a ``type="skybox"``
    texture; attaching them yields several. MuJoCo supports only one skybox —
    extras make the renderer fall back to the magenta missing-texture colour.
    Keep the first, delete the rest.
    """
    skyboxes = [
        tex for tex in spec.textures
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX
    ]
    for tex in skyboxes[1:]:
        spec.delete(tex)


def _scene_output_paths(
    config: BuildSceneConfig, scene_label_suffix: str = ""
) -> tuple[Path, Path]:
    """Resolve the generated XML path and its .config.json sidecar path.

    ``scene_label_suffix`` is appended to the scene-type token (e.g. "legacy")
    so an alternate build does not overwrite the standard scene's XML.
    """
    gravcomp = f"ag{gravcomp_label(1.0)}_eg{gravcomp_label(1.0)}"
    output_end_effector = config.end_effector
    scene_tokens = [config.scene_type + scene_label_suffix]
    if config.selected_variants:
        scene_tokens.append(f"variants{_stable_short_hash(config.selected_variants)}")
    if config.walls:
        scene_tokens.append("walls")
    if config.wrist_control == "hybrid_weld":
        scene_tokens.append("hybridweld")
    if (
        config.arm_type in ARM_XML
        and config.end_effector == "sharpa"
        and config.finger_control == "ik"
        and config.wrist_control in ("ik", "hybrid_weld")
    ):
        scene_tokens.append("ikboth")
    # Tag non-default sharpa visual mesh quality so good/high builds don't
    # overwrite the standard-quality scene file (and vice versa).
    if config.end_effector == "sharpa" and config.sharpa_visual_mesh_quality != "standard":
        scene_tokens.append(f"{config.sharpa_visual_mesh_quality}mesh")
    scene_label = "_".join(scene_tokens)
    xml_path = (
        GENERATED
        / f"{config.arm_type}_{scene_label}_{output_end_effector}_{config.actuator_mode}_{gravcomp}.xml"
    )
    return xml_path, xml_path.with_suffix(".config.json")


def _finalize_scene(
    config: BuildSceneConfig,
    spec: mujoco.MjSpec,
    source_xml: Path,
    scene_label_suffix: str = "",
) -> GeneratedScene:
    """Compile a composed spec, write its XML + config sidecar, return the scene.

    Shared tail of build_scene and build_scene_from_recording: everything from a
    ready-to-compile MjSpec onward. ``scene_label_suffix`` disambiguates the
    output filename for alternate builds (e.g. a legacy scene revision).
    """
    xml_path, config_path = _scene_output_paths(config, scene_label_suffix)
    model = spec.compile()
    _atomic_write_text(xml_path, spec.to_xml(), encoding="utf-8")

    config_data = streamer_config(config)
    # Surface variant-pool metadata (slot body names per pool) onto the DR
    # config so the runtime `variant_select` op can resolve pools at reset
    # time. selected_variants pins specific ids during replay; absent during
    # recording, which lets DR sample freely.
    variant_pool_metadata = dict(_LAST_VARIANT_POOL_METADATA)
    if variant_pool_metadata:
        dr_section = config_data.setdefault("domain_randomization", {})
        dr_section["variant_pools"] = variant_pool_metadata
        if config.selected_variants:
            dr_section["selected_variants"] = dict(config.selected_variants)
    if (
        config.arm_type in ARM_XML
        and config.end_effector == "sharpa"
        and config.finger_control == "ik"
        and config.wrist_control in ("ik", "hybrid_weld")
    ):
        for side in ("right", "left"):
            ik_xml_path = xml_path.with_name(f"{xml_path.stem}_{side}_ik.xml")
            build_joint_ik_scene(config.arm_type, side, ik_xml_path)
            config_data[f"joint_ik_xml_{side}"] = str(ik_xml_path.relative_to(ROOT))

    _atomic_write_text(
        config_path,
        json.dumps(config_data, indent=2) + "\n",
        encoding="utf-8",
    )

    return GeneratedScene(
        model=model,
        xml_path=xml_path,
        config_path=config_path,
        streamer_config=config_data,
        base_xml=BASE_SCENE_XML,
        source_xml=source_xml,
    )


def build_scene(config: BuildSceneConfig) -> GeneratedScene:
    """Build a scene from a config: compose the spec, compile, write XML."""
    prepare_generated_dir()
    spec = compose_scene_spec(config)
    return _finalize_scene(config, spec, TASK_SOURCE_XML[config.scene_type])


# DR ops that resize geometry. Every other op (offset, freejoint_pose,
# scatter, drops, grid_place) only moves things — replay reproduces those from
# the recorded trajectory's qpos, so they need no scene-build handling.
_GEOMETRY_DR_OPS = ("scale", "size_offset")


def recorded_dr_samples(root) -> list[dict]:
    """The recorded domain-randomization sample log, or [] if none.

    Tolerates recordings with no DR section and older recordings whose log
    shape differs — callers filter to the ops they understand.
    """
    init = root["initial_state"] if "initial_state" in root else None
    if init is None:
        return []
    dr = init.attrs.get("domain_randomization")
    if not isinstance(dr, dict):
        return []
    samples = dr.get("samples")
    return list(samples) if isinstance(samples, list) else []


def apply_recorded_dr_to_spec(spec: mujoco.MjSpec, root) -> None:
    """Re-apply a recording's geometry randomization onto a composed spec.

    The DR ``scale`` and ``size_offset`` ops resize bodies; both log their
    per-body multiplier keyed by body name. Re-applying those multipliers to
    the spec's geoms (and the meshes they reference) reproduces the recorded
    geometry — by name, so it is immune to geoms being added/removed/reordered
    since the recording. Friction is never randomized; pose ops are reproduced
    from the trajectory, not here.

    Unknown or pose-only ops are ignored. Bodies absent from the current scene
    (renamed/removed since the recording) are skipped.
    """
    for sample in recorded_dr_samples(root):
        if not isinstance(sample, dict) or sample.get("op") not in _GEOMETRY_DR_OPS:
            continue
        value = sample.get("value")
        if not isinstance(value, dict):
            continue
        # scale logs `factors_by_body`; size_offset logs `factor_by_body`.
        factors = value.get("factors_by_body") or value.get("factor_by_body") or {}
        scales_meshes = sample.get("op") == "scale"
        scaled_meshes: set[str] = set()
        for body_name, factor in factors.items():
            try:
                body = spec.body(body_name)
            except (KeyError, ValueError):
                continue  # body no longer in the scene — skip
            if body is None:
                continue
            factor_arr = np.asarray(factor, dtype=float)
            for geom in body.geoms:
                geom.size = np.asarray(geom.size, dtype=float) * factor_arr
                # A mesh geom's vertices are scaled too. Meshes can be shared
                # across geoms, so scale each referenced mesh only once.
                if not scales_meshes:
                    continue
                mesh_name = geom.meshname
                if not mesh_name or mesh_name in scaled_meshes:
                    continue
                try:
                    mesh = spec.mesh(mesh_name)
                except (KeyError, ValueError):
                    continue
                mesh.scale = np.asarray(mesh.scale, dtype=float) * factor_arr
                scaled_meshes.add(mesh_name)


# Legacy task-scene revisions kept only for replaying old recordings whose
# joint count no longer matches the current scene. Keyed by (scene_type,
# recorded qpos width) -> committed legacy XML. spell_and_stow changed shape
# twice — nq 164 (15 blocks A-O, May 13) and nq 171 (May 15) — before the
# current nq 155 build; each legacy XML reproduces one revision so pre-change
# recordings still load.
_LEGACY_TASK_XML: dict[tuple[str, int], Path] = {}


def _legacy_source_xml(scene_type: str, root) -> tuple[Path, str] | None:
    """The legacy task XML for a recording, if its qpos width needs one.

    Returns (xml_path, scene_label_suffix) or None when the current scene
    already matches (the common case). The suffix carries the recorded qpos
    width so different legacy revisions write distinct generated XMLs.
    """
    try:
        recorded_nq = int(root["trajectory"]["qpos"].shape[1])
    except (KeyError, AttributeError, IndexError):
        return None
    xml = _LEGACY_TASK_XML.get((scene_type, recorded_nq))
    if xml is None:
        return None
    return xml, f"_legacy_nq{recorded_nq}"


def build_scene_from_recording(
    root, *, overrides: dict | None = None
) -> GeneratedScene:
    """Rebuild the scene a recording was captured in.

    Reconstructs the BuildSceneConfig from the recording, pins the recorded
    variant-pool selections, applies the recorded geometry randomization to the
    spec (apply_recorded_dr_to_spec), and finalizes — so the emitted XML and
    compiled model match the recorded scene. ``overrides`` evolves the config
    before composing.

    Recordings of a task scene that has since changed joint count are rebuilt
    against a committed legacy XML (see _LEGACY_TASK_XML) so they still load.
    """
    from mujoco_vr_teleop import variant_pools as _vp

    builder = root.attrs["streamer_config"]["config"]["builder"]
    config = BuildSceneConfig.from_dict(builder)

    # Sharpa hand mesh quality default. Recordings may have been captured with
    # ``standard`` (small STL meshes, fast GLB export); for replay rendering we
    # prefer ``good`` (USDZ-derived OBJs decimated ~50%) — the visual quality
    # difference is noticeable in mp4s + batch-render output without paying the
    # full-res cost. Callers can opt out via overrides={"sharpa_visual_mesh_quality":...}.
    if (overrides is None or "sharpa_visual_mesh_quality" not in overrides) and \
            config.end_effector == "sharpa":
        config = attrs.evolve(config, sharpa_visual_mesh_quality="good")

    variant_pools_meta = (
        root.attrs["streamer_config"]["config"]
        .get("domain_randomization", {})
        .get("variant_pools", {})
    )
    chosen = variant_pools_meta.get("chosen_variants")
    cup_build = variant_pools_meta.get("cup_build")
    is_cup_scene = config.scene_type in ("cup_stack", "cup_ball")
    if chosen:
        config = attrs.evolve(config, selected_variants=dict(chosen))
    if overrides:
        config = attrs.evolve(config, **overrides)

    legacy = _legacy_source_xml(config.scene_type, root)
    legacy_xml = legacy[0] if legacy else None
    legacy_suffix = legacy[1] if legacy else ""
    source_xml = legacy_xml or TASK_SOURCE_XML[config.scene_type]

    prepare_generated_dir()
    _vp.set_pinned_pools(chosen or None)
    if is_cup_scene:
        _vp.set_pinned_cup_build(config.scene_type, cup_build or None)
    try:
        spec = compose_scene_spec(config, source_xml_override=legacy_xml)
        apply_recorded_dr_to_spec(spec, root)
        return _finalize_scene(
            config, spec, source_xml, scene_label_suffix=legacy_suffix
        )
    finally:
        _vp.set_pinned_pools(None)
        if is_cup_scene:
            _vp.set_pinned_cup_build(config.scene_type, None)
