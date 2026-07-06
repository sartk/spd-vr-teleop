"""Per-scene variant pools.

Two API surfaces:

  1. Compile-time picker (``attach_chosen_variants``). Used by the production
     scene builder: for each scene, pick ONE variant per pool (sampled or
     pinned via ``selected_variants``) and attach that variant's MJCF as a
     subtree of the scene spec. Each attached variant carries its own
     ``<asset>``, ``<material>``, ``<texture>`` blocks so the streamer/VR
     gets full materials with no extra work.

  2. Bench-time thin builder (``attach_thin_pool_for_variants``). Used by the
     mjwarp K-sweep bench: given a set of variant tuples observed across
     recorded episodes, build a thin scene with ALL observed variants'
     meshes embedded and one slot body per pool sized to the max-mesh count.
     Per-world ``geom_dataid`` / ``body_*`` widening then drives which
     variant each world simulates.

Production scene NEVER uses (2). Bench/training NEVER uses (1) at step time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR_DEFAULT = REPO_ROOT / "examples" / "assets"

# Variant id pools. Excluded entries:
#   plate_11 (820k visual verts — too heavy)
#   plate_5, plate_16 (non-flat / tilted refquat — grippers penetrate them)
#   plate_1 (worst visual-vs-collision discrepancy: 34mm hausdorff, won't
#     fit the rack properly)
#   dish_rack_0 (uses mesh-collider decomposition; the others use box
#     primitives; mixing geom types within one slot doesn't work for the
#     thin bench pattern. The streamer compile path doesn't care.)
DISHRACK_PLATE_IDS = [0, 2, 3, 4, 7, 8, 9, 14, 15, 17, 18]
DISHRACK_RACK_IDS = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12]
# Excluded: mug_1 (brightness=0.08) and mug_12 (brightness=0.07) — their
# baseColor textures are near-black so they render as featureless dark blobs
# even under bright ambient lighting.
DISHRACK_MUG_IDS = [i for i in range(19) if i not in (1, 12)]
MUG_TREE_IDS = [0]


@dataclass(frozen=True)
class PoolDef:
    """Declarative description of one pool slot in a scene.

    ``slot_body`` is the body name installed at compile time (single body
    per pool; no preloading). ``replaces_body`` names a placeholder body in
    the source XML that this pool replaces (legacy ``dishrack``,
    ``plate_0``, etc.).
    """
    name: str
    slot_body: str
    variant_root: Path
    variant_dir_template: str   # e.g. "dish_rack_{vid}"
    ids: list[int]
    home_pos: tuple[float, float, float]
    replaces_body: str | None = None


DISHRACK_POOLS: list[PoolDef] = [
    PoolDef(
        name="dishrack_rack",
        slot_body="rack_slot",
        variant_root=ASSETS_DIR_DEFAULT / "task_dishrack" / "dish_rack",
        variant_dir_template="dish_rack_{vid}",
        ids=DISHRACK_RACK_IDS,
        home_pos=(0.62, -0.24, 0.85),
        replaces_body="dishrack",
    ),
    *[
        PoolDef(
            name=f"dishrack_plate_{i}",
            slot_body=f"plate_slot_{letter}",
            variant_root=ASSETS_DIR_DEFAULT / "task_dishrack" / "plate",
            variant_dir_template="plate_{vid}",
            ids=DISHRACK_PLATE_IDS,
            home_pos=home,
            replaces_body=replaces,
        )
        for i, (letter, home, replaces) in enumerate([
            ("a", (0.55, 0.36, 0.85), "plate_0"),
            ("b", (0.55, 0.48, 0.85), "plate_1"),
            ("c", (0.82, 0.36, 0.85), None),
            ("d", (0.82, 0.48, 0.85), None),
        ])
    ],
    *[
        PoolDef(
            name=f"dishrack_mug_{i}",
            slot_body=f"mug_slot_{letter}",
            variant_root=ASSETS_DIR_DEFAULT / "task_dishrack" / "mug",
            variant_dir_template="mug_{vid}",
            ids=DISHRACK_MUG_IDS,
            home_pos=home,
            replaces_body=replaces,
        )
        for i, (letter, home, replaces) in enumerate([
            ("a", (0.47, 0.10, 0.85), "mug_0"),
            ("b", (0.82, 0.10, 0.85), "mug_1"),
        ])
    ],
]


HANG_MUGS_POOLS: list[PoolDef] = [
    PoolDef(
        name="hang_mugs_mug_tree",
        slot_body="mug_tree_slot",
        variant_root=ASSETS_DIR_DEFAULT / "mug_tree",
        variant_dir_template="mug_tree_{vid}",
        ids=MUG_TREE_IDS,
        home_pos=(0.70, 0.00, 0.85),
    ),
    *[
        PoolDef(
            name=f"hang_mugs_mug_{i}",
            slot_body=f"mug_slot_{letter}",
            variant_root=ASSETS_DIR_DEFAULT / "task_dishrack" / "mug",
            variant_dir_template="mug_{vid}",
            ids=DISHRACK_MUG_IDS,
            home_pos=home,
        )
        for i, (letter, home) in enumerate([
            ("a", (0.60, -0.20, 0.85)),
            ("b", (0.60, 0.20, 0.85)),
            ("c", (0.75, -0.20, 0.85)),
            ("d", (0.75, 0.20, 0.85)),
        ])
    ],
]


# Full pool of mug slots that hang_mugs MAY include. Each rebuild picks a
# random subset of size 1..MAX_HANG_MUGS_COUNT (driven by `pools_for("hang_mugs")`).
HANG_MUGS_MUG_POOLS: list[PoolDef] = [p for p in HANG_MUGS_POOLS
                                        if p.name.startswith("hang_mugs_mug_")
                                        and p.name != "hang_mugs_mug_tree"]
HANG_MUGS_TREE_POOL: PoolDef = next(p for p in HANG_MUGS_POOLS
                                     if p.name == "hang_mugs_mug_tree")
MAX_HANG_MUGS_COUNT = len(HANG_MUGS_MUG_POOLS)


# Bottles in bin: one garbage_can (single asset, no variants — vid 0 always)
# + up to MAX_BOTTLES_IN_BIN bottle slots. pools_for() picks 3..MAX per build.
BOTTLES_IN_BIN_BOTTLE_IDS = list(range(19))  # bottle_0 .. bottle_18

BOTTLES_IN_BIN_BIN_POOL: PoolDef = PoolDef(
    name="bottles_in_bin_bin",
    slot_body="bin_slot",
    variant_root=ASSETS_DIR_DEFAULT / "task_bottles_in_bin",
    variant_dir_template="garbage_can",  # no {vid} — only one asset
    ids=[0],
    home_pos=(0.82, 0.00, 0.85),
)

# 6 bottle slots: pools_for picks 3..6 of these per build. Letters a..f.
BOTTLES_IN_BIN_BOTTLE_POOLS: list[PoolDef] = [
    PoolDef(
        name=f"bottles_in_bin_bottle_{i}",
        slot_body=f"bottle_slot_{letter}",
        variant_root=ASSETS_DIR_DEFAULT / "task_bottles_in_bin" / "bottle",
        variant_dir_template="bottle_{vid}",
        ids=BOTTLES_IN_BIN_BOTTLE_IDS,
        home_pos=home,
    )
    for i, (letter, home) in enumerate([
        ("a", (0.55, -0.30, 0.85)),
        ("b", (0.55, -0.10, 0.85)),
        ("c", (0.55, 0.10, 0.85)),
        ("d", (0.55, 0.30, 0.85)),
        ("e", (0.68, -0.20, 0.85)),
        ("f", (0.68, 0.20, 0.85)),
    ])
]
MAX_BOTTLES_IN_BIN = len(BOTTLES_IN_BIN_BOTTLE_POOLS)
MIN_BOTTLES_IN_BIN = 3
BOTTLES_IN_BIN_POOLS: list[PoolDef] = [
    BOTTLES_IN_BIN_BIN_POOL,
    *BOTTLES_IN_BIN_BOTTLE_POOLS,
]


# ---------------------------------------------------------------------------
# Cup scenes: two independent scene types, cup_stack and cup_ball. Each build
# is task-driven — the task decides the cup COUNT (3 or 6), whether a ball is
# present, and (for ball tasks) forces the opaque plastic family so the ball
# can be hidden. The cup TYPE is one variant shared by every cup in the build;
# a single per-build scale factor is applied uniformly to all cups.
#
# cup_stack only uses cup variants that nest cleanly (the restricted id sets
# below). cup_ball uses the full id sets.
# ---------------------------------------------------------------------------
PLASTIC_CUP_ROOT = ASSETS_DIR_DEFAULT / "task_cups" / "plastic_cup"
GLASS_CUP_ROOT = ASSETS_DIR_DEFAULT / "task_cups" / "glass_cup"
MAX_CUPS = 6

# Full id sets (task_cups/plastic_cup/cup_0..3, task_cups/glass_cup/glass_cup_0..23).
PLASTIC_CUP_IDS_ALL = [0, 1, 2, 3]
GLASS_CUP_IDS_ALL = list(range(24))

# cup_ball uses every cup variant.
CUP_BALL_PLASTIC_IDS = list(PLASTIC_CUP_IDS_ALL)
CUP_BALL_GLASS_IDS = list(GLASS_CUP_IDS_ALL)

# cup_stack uses only cups that nest cleanly for stacking (plastic_cup_3 and
# most glass cups flare or are bulbous and do not stack reliably).
CUP_STACK_PLASTIC_IDS = [0, 1, 2]
CUP_STACK_GLASS_IDS = [7, 9, 10, 11, 20]

# --- cup_stack tasks ---
# Per-task build profile: (cup_count, with_ball, forced_family|None).
CUP_STACK_TASK_PROFILE: dict[str, tuple[int, bool, str | None]] = {
    "stack_two_threes": (6, False, None),
    "unstack":          (6, False, None),
    "pyramid":          (6, False, None),
}

# --- cup_ball tasks ---
CUP_BALL_TASK_PROFILE: dict[str, tuple[int, bool, str | None]] = {
    "shuffle_ball":     (3, True,  "plastic"),
    "pong":             (6, True,  None),
    "playground":       (6, True,  None),
}

# Per-scene build state, mutated by pools_for() each build; scene_resets and
# the streamer read it back to learn the current build's cup count / ball /
# family / cup id / scale. ``cup_id`` is the single variant id shared by every
# cup slot; ``scale`` is the single factor applied uniformly to all cups.
CUP_STACK_BUILD: dict = {"task": None, "family": None, "with_ball": False,
                         "n": MAX_CUPS, "cup_id": None, "scale": 1.0}
CUP_BALL_BUILD: dict = {"task": None, "family": None, "with_ball": False,
                        "n": MAX_CUPS, "cup_id": None, "scale": 1.0}

# Uniform cup-scale randomization range, applied per build (one factor for the
# whole build, shared by every cup).
CUP_SCALE_RANGE = (0.9, 1.1)

# Per-scene config: (task profile, build-state dict, plastic ids, glass ids).
_CUP_SCENE_CFG: dict[str, tuple] = {
    "cup_stack": (CUP_STACK_TASK_PROFILE, CUP_STACK_BUILD,
                  CUP_STACK_PLASTIC_IDS, CUP_STACK_GLASS_IDS),
    "cup_ball":  (CUP_BALL_TASK_PROFILE, CUP_BALL_BUILD,
                  CUP_BALL_PLASTIC_IDS, CUP_BALL_GLASS_IDS),
}

# Set by the task manager before a (re)build so pools_for knows which task's
# profile to build. Keyed by scene type. Absent => pools_for samples a task.
_CUP_TASK_OVERRIDE: dict[str, str | None] = {}

# Set by replay before a (re)build so pools_for reproduces the recorded pool
# subset exactly (count of bodies) instead of re-rolling. None => random.
# Holds the recorded ``chosen_variants`` dict ({pool_name: variant_id}); only
# its keys (= which pools existed) are used here.
_PINNED_POOL_NAMES: set[str] | None = None

# Set by replay before a cup-scene (re)build, keyed by scene type. The cup
# scenes have non-deterministic build inputs NOT recoverable from
# chosen_variants alone — cup ``family`` (plastic/glass id ranges overlap, so a
# variant id does not identify the family) and, transitively, ``with_ball``,
# ``n``, ``scale``. Replay pins the whole recorded build dict here so pools_for
# reproduces the build with zero rng calls. Absent => normal sampling.
_PINNED_CUP_BUILD: dict[str, dict] = {}


def set_pinned_pools(chosen_variants: dict[str, int] | None) -> None:
    """Tell the next ``pools_for(...)`` to reproduce exactly these pools.

    ``chosen_variants`` is the dict recorded into ``streamer_config`` at record
    time. Pass None to restore random subset sampling.
    """
    global _PINNED_POOL_NAMES
    _PINNED_POOL_NAMES = set(chosen_variants) if chosen_variants else None


def set_pinned_cup_build(scene_type: str, cup_build: dict | None) -> None:
    """Tell the next ``pools_for(scene_type, ...)`` to reproduce this build.

    ``cup_build`` is the recorded build dict ({task, family, with_ball, n,
    cup_id, scale}) captured at record time into ``streamer_config``. Pass None
    to restore normal sampling. ``scene_type`` is ``cup_stack`` or ``cup_ball``.
    """
    if scene_type not in _CUP_SCENE_CFG:
        raise ValueError(f"set_pinned_cup_build: unknown scene {scene_type!r}")
    if cup_build:
        _PINNED_CUP_BUILD[scene_type] = dict(cup_build)
    else:
        _PINNED_CUP_BUILD.pop(scene_type, None)


def set_cup_task(scene_type: str, task_short_name: str | None) -> None:
    """Tell the next ``pools_for(scene_type, ...)`` which task to build for.

    ``task_short_name`` is the part after ``<scene>/`` (e.g. ``"pong"``), or
    None to let pools_for sample a task at random. ``scene_type`` is
    ``cup_stack`` or ``cup_ball``.
    """
    if scene_type not in _CUP_SCENE_CFG:
        raise ValueError(f"set_cup_task: unknown scene {scene_type!r}")
    profile = _CUP_SCENE_CFG[scene_type][0]
    if task_short_name is not None and task_short_name not in profile:
        raise ValueError(f"set_cup_task: unknown task {task_short_name!r} for "
                          f"{scene_type!r} (have: {sorted(profile)})")
    if task_short_name is None:
        _CUP_TASK_OVERRIDE.pop(scene_type, None)
    else:
        _CUP_TASK_OVERRIDE[scene_type] = task_short_name


def _cup_pools(scene_type: str, family: str, n: int,
               cup_id: int) -> list[PoolDef]:
    """``n`` cup slots, every slot the same ``cup_id`` of ``family``.

    A cup build uses one cup type throughout — every slot's pool is pinned to
    the single ``cup_id`` so all cups in the scene are identical. ``name`` is
    prefixed with the scene type so chosen_variants keys are scene-unique.
    """
    if family == "plastic":
        root, tmpl = PLASTIC_CUP_ROOT, "cup_{vid}"
    elif family == "glass":
        root, tmpl = GLASS_CUP_ROOT, "glass_cup_{vid}"
    else:
        raise ValueError(f"_cup_pools: unknown family {family!r}")
    letters = "abcdef"[:n]
    return [
        PoolDef(
            name=f"{scene_type}_cup_{i}",
            slot_body=f"cup_slot_{ltr}",
            variant_root=root,
            variant_dir_template=tmpl,
            ids=[int(cup_id)],
            home_pos=(0.70, -0.30 + 0.12 * i, 0.95),
        )
        for i, ltr in enumerate(letters)
    ]


def cup_ids_for(scene_type: str, family: str) -> list[int]:
    """Valid cup variant ids for a scene + family ("plastic" or "glass")."""
    if scene_type not in _CUP_SCENE_CFG:
        raise ValueError(f"cup_ids_for: unknown scene {scene_type!r}")
    _, _, plastic_ids, glass_ids = _CUP_SCENE_CFG[scene_type]
    if family == "plastic":
        return plastic_ids
    if family == "glass":
        return glass_ids
    raise ValueError(f"cup_ids_for: unknown family {family!r}")


CUP_STACK_POOLS: list[PoolDef] = _cup_pools(
    "cup_stack", "plastic", MAX_CUPS, CUP_STACK_PLASTIC_IDS[0])
CUP_BALL_POOLS: list[PoolDef] = _cup_pools(
    "cup_ball", "plastic", MAX_CUPS, CUP_BALL_PLASTIC_IDS[0])


SCENE_POOLS: dict[str, list[PoolDef]] = {
    "dishrack": DISHRACK_POOLS,
    # Initialized to the full set; pools_for() mutates this in-place each
    # build so downstream consumers (scene_resets._slot, reset iteration)
    # see the current build's actual pool subset.
    "hang_mugs": list(HANG_MUGS_POOLS),
    "bottles_in_bin": list(BOTTLES_IN_BIN_POOLS),
    # Initialized to valid defaults; pools_for() rebuilds these per task.
    "cup_stack": list(CUP_STACK_POOLS),
    "cup_ball": list(CUP_BALL_POOLS),
}


def pools_for(scene_type: str, rng: np.random.Generator | None = None) -> list[PoolDef]:
    """Return the pools for ``scene_type`` for the upcoming build.

    For most scenes this is just the static SCENE_POOLS entry. For scenes
    that randomize the *count* of bodies per build (currently hang_mugs and
    bottles_in_bin), this picks a random subset and writes it back into
    SCENE_POOLS so downstream readers (scene_resets._slot, reset iteration)
    see the current build's actual pool subset.
    """
    if rng is None:
        rng = np.random.default_rng()
    if scene_type == "hang_mugs":
        if _PINNED_POOL_NAMES is not None:
            n_active = sum(1 for p in HANG_MUGS_MUG_POOLS
                           if p.name in _PINNED_POOL_NAMES)
        else:
            n_active = int(rng.integers(1, MAX_HANG_MUGS_COUNT + 1))
        active_subset = [HANG_MUGS_TREE_POOL, *HANG_MUGS_MUG_POOLS[:n_active]]
        SCENE_POOLS["hang_mugs"] = active_subset
        return active_subset
    if scene_type == "bottles_in_bin":
        if _PINNED_POOL_NAMES is not None:
            n_active = sum(1 for p in BOTTLES_IN_BIN_BOTTLE_POOLS
                           if p.name in _PINNED_POOL_NAMES)
        else:
            n_active = int(rng.integers(MIN_BOTTLES_IN_BIN, MAX_BOTTLES_IN_BIN + 1))
        active_subset = [BOTTLES_IN_BIN_BIN_POOL,
                         *BOTTLES_IN_BIN_BOTTLE_POOLS[:n_active]]
        SCENE_POOLS["bottles_in_bin"] = active_subset
        return active_subset
    if scene_type in _CUP_SCENE_CFG:
        # The task drives the build: cup count, ball presence, forced family.
        # Three input sources, in priority order:
        #   1. _PINNED_CUP_BUILD[scene] — replay: reproduce the recorded build
        #      EXACTLY, reading task/family/with_ball/n/cup_id/scale with zero
        #      rng calls.
        #   2. _CUP_TASK_OVERRIDE[scene] — task manager picked the task; family,
        #      cup id and scale are still sampled (family forced for ball tasks).
        #   3. neither — bare build / test: sample the task too.
        profile, build_state, _, _ = _CUP_SCENE_CFG[scene_type]
        pinned = _PINNED_CUP_BUILD.get(scene_type)
        if pinned is not None:
            task = pinned["task"]
            family = pinned["family"]
            with_ball = bool(pinned["with_ball"])
            n = int(pinned["n"])
            cup_id = int(pinned["cup_id"])
            scale = float(pinned["scale"])
        else:
            override = _CUP_TASK_OVERRIDE.get(scene_type)
            if override is not None:
                task = override
            else:
                task = str(rng.choice(sorted(profile)))
            n, with_ball, forced_family = profile[task]
            family = forced_family or ("plastic" if rng.random() < 0.5 else "glass")
            # One cup type per build: pick a single id every slot will share.
            cup_id = int(rng.choice(cup_ids_for(scene_type, family)))
            # One uniform scale factor for the whole build, shared by all cups.
            scale = float(rng.uniform(*CUP_SCALE_RANGE))
        build_state.update(task=task, family=family, with_ball=with_ball,
                            n=n, cup_id=cup_id, scale=scale)
        active_subset = _cup_pools(scene_type, family, n, cup_id)
        SCENE_POOLS[scene_type] = active_subset
        return active_subset
    return SCENE_POOLS.get(scene_type, [])


# ---------------------------------------------------------------------------
# (1) Compile-time picker for the production scene
# ---------------------------------------------------------------------------

def _delete_body_if_present(spec: mujoco.MjSpec, body_name: str) -> bool:
    try:
        body = spec.body(body_name)
    except (KeyError, ValueError):
        return False
    if body is None:
        return False
    spec.delete(body)
    return True


def _resolve_variant_mesh_paths(child: mujoco.MjSpec, variant_xml: Path) -> None:
    """Mutate ``child`` so every mesh and texture file is an absolute path,
    honoring the variant XML's own ``meshdir`` / ``texturedir``. Without this,
    the merged spec's ``to_xml()`` produces bare basenames that the streamer
    fails to re-load.
    """
    child_meshdir = child.compiler.meshdir or ""
    child_texdir = child.compiler.texturedir or child_meshdir
    base = variant_xml.parent
    mesh_base = (base / child_meshdir).resolve()
    tex_base = (base / child_texdir).resolve()
    for mesh in list(child.meshes):
        if mesh.file:
            mesh.file = str((mesh_base / mesh.file).resolve())
    for tex in list(child.textures):
        if tex.file:
            tex.file = str((tex_base / tex.file).resolve())


def attach_chosen_variants(spec: mujoco.MjSpec, scene_type: str,
                            selected_variants: dict[str, int] | None = None,
                            rng: np.random.Generator | None = None) -> dict[str, int]:
    """Attach ONE chosen variant per pool to ``spec`` as a subtree.

    For each pool registered for ``scene_type``:
      1. Pick a variant id (pinned via ``selected_variants[pool.name]`` or
         randomly sampled from ``pool.ids``).
      2. Remove the placeholder body named ``pool.replaces_body`` if present.
      3. Load the variant's ``model.xml`` as an MjSpec, rewrite mesh/texture
         file paths to absolute, strip any inner freejoint, then attach
         under a fresh slot body at ``pool.home_pos`` with its own
         freejoint.

    Returns ``{pool_name: variant_id}`` — the actual selections — so the
    caller can record them (e.g. into ``streamer_config`` for replay).
    """
    rng = rng if rng is not None else np.random.default_rng()
    pools = pools_for(scene_type, rng)
    if not pools:
        return {}
    selected_variants = selected_variants or {}

    chosen: dict[str, int] = {}
    for pool in pools:
        if pool.name in selected_variants:
            vid = int(selected_variants[pool.name])
            if vid not in pool.ids:
                raise ValueError(
                    f"attach_chosen_variants: pinned id {vid} not in pool "
                    f"{pool.name!r} (have: {pool.ids})"
                )
        else:
            vid = int(rng.choice(pool.ids))
        chosen[pool.name] = vid

        if pool.replaces_body:
            _delete_body_if_present(spec, pool.replaces_body)

        variant_xml = pool.variant_root / pool.variant_dir_template.format(vid=vid) / "model.xml"
        child = mujoco.MjSpec.from_file(str(variant_xml))
        _resolve_variant_mesh_paths(child, variant_xml)

        # Variants may carry their own freejoint (e.g. plate_0); the slot
        # owns the freejoint instead.
        for jnt in list(child.joints):
            if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
                child.delete(jnt)

        slot = spec.worldbody.add_body(
            name=pool.slot_body,
            pos=list(pool.home_pos),
        )
        slot.add_freejoint(name=f"{pool.slot_body}_joint")
        attach_frame = slot.add_frame(name=f"{pool.slot_body}_frame", pos=[0.0, 0.0, 0.0])
        spec.attach(child, prefix=f"{pool.slot_body}__", frame=attach_frame)

    # The cup_ball scene's ball tasks need a ball. It is a plain sphere (no
    # mesh, no variant pool); inject it here so it shares the build and is
    # parked / placed by the scene reset. Radius 0.02 m = 40 mm regulation
    # ping-pong; density ~120 kg/m^3 ≈ 2.6 g, close to a real ball.
    if scene_type == "cup_ball" and CUP_BALL_BUILD.get("with_ball"):
        ball = spec.worldbody.add_body(name="ball", pos=[0.70, 0.0, 0.95])
        ball.add_freejoint(name="ball_joint")
        ball.add_geom(
            name="ball_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0.0, 0.0],
            density=120.0,
            friction=[0.95, 0.3, 0.1],
            solimp=[0.998, 0.998, 0.001, 0.5, 2.0],
            solref=[0.001, 1.0],
            rgba=[1.0, 0.5, 0.0, 1.0],
        )

    return chosen


# ---------------------------------------------------------------------------
# (2) Bench-time thin pool builder
# ---------------------------------------------------------------------------
#
# Lives in scratch/variant_swap/thin_dishrack.py and friends, NOT here. The
# production import path only sees `attach_chosen_variants` above. Keeping
# the bench scaffolding out of mujoco_vr_teleop/ means no preload/slot
# machinery runs at streamer startup.
