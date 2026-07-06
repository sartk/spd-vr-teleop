"""Cross-scene helpers shared by every tasks/<scene>/ subpackage.

Only put things here that are scene-agnostic (settle checks, pose math).
Scene-specific stuff (block names, tower geometry) belongs in
``tasks/<scene>/common.py``.
"""

from __future__ import annotations

import numpy as np

from mujoco_vr_teleop.tasks.base import RewardContext


def settled(ctx: RewardContext, body: str, lin_thresh: float = 0.02,
            ang_thresh: float = 0.5) -> bool:
    """True if `body` has both linear and angular speed below thresholds."""
    return (
        float(np.linalg.norm(ctx.body_lin_vel(body))) < lin_thresh
        and float(np.linalg.norm(ctx.body_ang_vel(body))) < ang_thresh
    )


def settled_for(ctx: RewardContext, body: str, *, key: str, seconds: float = 0.5,
                lin_thresh: float = 0.02, ang_thresh: float = 0.5) -> bool:
    """Sticky 'settled for at least N seconds' check.

    Counts consecutive ticks where the body has been still. Once the threshold
    is reached, returns True; if the body moves, the counter resets. State
    lives in ``ctx.scratch[key]`` so each call site needs a unique key.
    """
    counters: dict[str, int] = ctx.scratch.setdefault("__settle_counters__", {})
    if settled(ctx, body, lin_thresh, ang_thresh):
        counters[key] = counters.get(key, 0) + 1
    else:
        counters[key] = 0
    return counters[key] * ctx.physics_dt >= seconds


def lifted_ever(ctx: RewardContext, body: str, *, table_z: float, lift_above: float = 0.05,
                key: str | None = None) -> bool:
    """Sticky: True once `body` has been seen above `table_z + lift_above` at
    any point in this task. Stored in ``ctx.scratch`` under ``key`` (defaults
    to the body name)."""
    flags: set[str] = ctx.scratch.setdefault("__lifted_blocks__", set())
    label = key or body
    if label in flags:
        return True
    if float(ctx.body_pos(body)[2]) >= table_z + lift_above:
        flags.add(label)
        return True
    return False


def long_axis_world(ctx: RewardContext, body: str) -> np.ndarray:
    """Unit vector along the body's local +z axis, expressed in world frame.

    Most rectangular bodies in this codebase have their longest extent along
    local z. For other bodies, write a body-specific helper.
    """
    w, x, y, z = ctx.body_quat(body)
    return np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])


def make_task(*, task_id: str, title: str, instruction: str, reset,
              difficulty: str = "hard", skill: str = "", template: str = "",
              target_duration_s: float = 300.0, probability: float = 0.1,
              randomize_overrides: dict[str, bool] | None = None) -> "TaskSpec":
    """Build a TaskSpec."""
    from mujoco_vr_teleop.tasks.base import TaskSpec
    return TaskSpec(
        id=task_id, title=title, instruction=instruction,
        difficulty=difficulty, skill=skill, template=template,
        target_duration_s=target_duration_s, probability=probability,
        reset=reset,
        randomize_overrides=randomize_overrides,
    )


# ---------------------------------------------------------------------------
# Reset bridge: invoke the existing scene-level DR JSON's start_mode.
# ---------------------------------------------------------------------------

# DomainRandomizer.create() snapshots the model state at construction as the
# "baseline" to restore between resets. When tasks call apply_dr_start_mode
# repeatedly, each call would otherwise snapshot the model in its
# already-randomized state, breaking subsequent resets (parked variants stay
# parked because contype=0 is in the new baseline). Cache the FIRST DR per
# model id so the baseline is captured once, from the clean post-compile state.
_TASK_DR_CACHE: dict[int, "object"] = {}


def apply_dr_start_mode(model, data, scene: str, mode_name: str) -> None:
    """Apply the named start_mode from examples/task_scenes/<scene>.domain_randomization.json
    (or .assembly.domain_randomization.json for jenga). Reuses the existing
    DomainRandomizer machinery rather than reimplementing reset semantics.
    """
    import json
    from pathlib import Path
    import mujoco

    from mujoco_vr_teleop.domain_randomization import DomainRandomizer

    repo = Path(__file__).resolve().parents[2]
    candidates = [
        repo / "examples" / "task_scenes" / f"{scene}.domain_randomization.json",
        repo / "examples" / "task_scenes" / f"{scene}.assembly.domain_randomization.json",
    ]
    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"No domain randomization JSON found for scene {scene!r}")
    with cfg_path.open() as f:
        config = json.load(f)
    start_modes = config.get("start_modes", {})
    if mode_name not in start_modes:
        raise KeyError(f"start_mode {mode_name!r} not in {cfg_path.name}; "
                       f"available: {sorted(start_modes)}")
    tree = start_modes[mode_name].get("tree", {})
    settle = start_modes[mode_name].get("settle", {}).get("seconds", 0.0)

    # Reuse a cached DomainRandomizer for this model so its baseline state
    # (captured at first construction) survives across calls.
    cache_key = id(model)
    dr = _TASK_DR_CACHE.get(cache_key)
    if dr is None or dr.data is not data:
        from mujoco_vr_teleop import scene_builder as _scene_builder
        dr_config: dict = {"enabled": True, "tree": tree, "scene_type": scene}
        pool_metadata = dict(_scene_builder._LAST_VARIANT_POOL_METADATA)
        if pool_metadata:
            dr_config["variant_pools"] = pool_metadata
        # If the most-recently-built scene pinned variants (e.g. replay), pass
        # those pins through so variant_select uses them instead of sampling.
        pinned = _scene_builder._LAST_SELECTED_VARIANTS
        if pinned:
            dr_config["selected_variants"] = dict(pinned)
        dr = DomainRandomizer.create(model, data, dr_config)
        _TASK_DR_CACHE[cache_key] = dr
    dr.restore_baseline()
    dr.sample_log.clear()

    # Apply the global tree first (e.g. rack variant_select), then the
    # start-mode tree (plate variants + smalls placement). Mirrors what
    # DomainRandomizer.randomize() does.
    global_tree = config.get("tree")
    if isinstance(global_tree, dict):
        dr._apply_node(global_tree)
    dr._apply_node(tree)
    if settle > 0:
        for _ in range(round(float(settle) / model.opt.timestep)):
            mujoco.mj_step(model, data)
