from __future__ import annotations

import os
import shutil
from pathlib import Path

import mujoco

from mujoco_vr_teleop.scene_builder import BuildSceneConfig


def resolve_episode_zarr_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or path.suffix == ".zarr":
        return path
    named = path / f"{path.name}.zarr"
    if named.exists():
        return named
    zarrs = sorted(path.glob("*.zarr"))
    if len(zarrs) == 1:
        return zarrs[0]
    raise FileNotFoundError(f"could not find a single episode zarr in {path}")


def builder_config_from_recording(root) -> BuildSceneConfig:
    builder = root.attrs["streamer_config"]["config"]["builder"]
    assert isinstance(builder, dict)
    return BuildSceneConfig.from_dict(builder)


def streamer_config_from_recording(root) -> dict:
    config = root.attrs["streamer_config"]["config"]
    assert isinstance(config, dict)
    return config


def _verify_recording_replayable(root) -> None:
    """Raise if a recording lacks the variant metadata needed to reproduce it.

    Pool-ful scenes need ``chosen_variants``; cup scenes additionally need
    ``cup_build``. Without these the rebuild would re-roll a different scene.
    """
    from mujoco_vr_teleop import variant_pools as _vp

    scene_type = builder_config_from_recording(root).scene_type
    variant_pools_meta = (
        streamer_config_from_recording(root)
        .get("domain_randomization", {})
        .get("variant_pools", {})
    )
    chosen = variant_pools_meta.get("chosen_variants")
    cup_build = variant_pools_meta.get("cup_build")
    if _vp.SCENE_POOLS.get(scene_type) and not chosen:
        raise ValueError(
            f"Recording of scene type {scene_type!r} is "
            "missing variant_pools.chosen_variants under streamer_config. "
            "Without it the randomized scene cannot be reproduced and "
            "replay would re-roll a different scene. Re-record against "
            "current tooling."
        )
    if scene_type in ("cup_stack", "cup_ball") and not cup_build:
        raise ValueError(
            f"Recording of scene type {scene_type!r} is missing "
            "variant_pools.cup_build under streamer_config — it predates "
            "cup_build capture and is not replayable. Re-record against "
            "current tooling."
        )


def _recording_scene_xml_is_replay_model(root) -> bool:
    return bool(
        root.attrs.get("scene_xml_is_replay_model", False)
        or root.attrs.get("scene_xml_kind") == "replay_model"
    )


def load_model_from_recording(
    root,
    *,
    skip_fixed_camera_update: bool = False,
    replay_memory_bytes: int | None = None,
    recorded_xml_path: str | Path | None = None,
    scene_overrides: dict | None = None,
) -> tuple[mujoco.MjModel, Path]:
    # Fast path: the recording references a pre-built replay XML next to it
    # (DR multipliers already baked into the spec). Load it directly instead
    # of re-running the full scene-builder pipeline.
    if recorded_xml_path is not None and _recording_scene_xml_is_replay_model(root):
        scene_xml_path = Path(recorded_xml_path).expanduser().resolve()
        if replay_memory_bytes is not None and replay_memory_bytes > 0:
            spec = mujoco.MjSpec.from_file(str(scene_xml_path))
            spec.memory = int(replay_memory_bytes)
            model = spec.compile()
        else:
            model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        model.opt.timestep = 1.0 / float(streamer_config_from_recording(root)["physics_rate"])
        _verify_recorded_variant_paths(root)
        return model, scene_xml_path

    # Rebuild the scene the recording was captured in via the scene builder's
    # build_scene_from_recording: it re-composes from current base/task XML,
    # pins the recorded variant pools, and re-applies the recorded geometry
    # randomization to the spec by body name (so added/removed geoms never
    # break it). scene_overrides evolves the config first (e.g. a table
    # texture for an mp4 render).
    from mujoco_vr_teleop.scene_builder import build_scene_from_recording

    _verify_recording_replayable(root)
    scene = build_scene_from_recording(root, overrides=scene_overrides)
    scene_xml_path = scene.xml_path.resolve()
    # Also stash a copy of the rebuilt XML alongside the recording, so anything
    # pointed at that directory (e.g. batch_render --xml-path) can find the
    # exact same DR-applied scene without knowing about examples/generated/.
    _save_xml_next_to_recording(root, scene_xml_path)
    model = scene.model
    if replay_memory_bytes is not None and replay_memory_bytes > 0:
        spec = mujoco.MjSpec.from_file(str(scene_xml_path))
        spec.memory = int(replay_memory_bytes)
        model = spec.compile()
    model.opt.timestep = 1.0 / float(streamer_config_from_recording(root)["physics_rate"])
    _verify_recorded_variant_paths(root)
    return model, scene_xml_path


def _save_xml_next_to_recording(root, scene_xml_path: Path) -> None:
    """Copy the rebuilt scene XML into the recording's directory.

    Best-effort: if the recording is on a read-only store, or its location
    can't be inferred from the zarr (remote/in-memory stores), this silently
    does nothing — replay still works against the canonical examples/generated
    copy returned in ``scene_xml_path``.
    """
    store_path = getattr(getattr(root, "store", None), "path", None)
    if not store_path:
        return
    try:
        recording_dir = Path(store_path).resolve().parent
    except (OSError, ValueError):
        return
    if not recording_dir.is_dir():
        return
    dest = recording_dir / scene_xml_path.name
    try:
        if dest.exists() and dest.samefile(scene_xml_path):
            return
        shutil.copyfile(scene_xml_path, dest)
    except (OSError, PermissionError):
        # Read-only mount, race with a concurrent replay, etc. — leave it.
        pass


def _verify_recorded_variant_paths(root) -> None:
    """Strict-mode check: every mesh path the recording resolved must still
    exist on disk. Raises ``ValueError`` listing missing assets if not.
    """
    if "initial_state" not in root:
        return
    paths_attr = root["initial_state"].attrs.get("selected_variant_paths")
    if not paths_attr:
        return
    missing = [
        (pool, path) for pool, path in dict(paths_attr).items()
        if not os.path.exists(str(path))
    ]
    if missing:
        details = "\n".join(f"  {pool}: {path}" for pool, path in missing)
        raise ValueError(
            "Recorded variant mesh paths missing on disk:\n"
            f"{details}\n"
            "Replay requires all variant assets to be present at the same "
            "paths the recording resolved. Restore them or rerun the "
            "recording against the current asset layout."
        )


# Recorded scene geometry is reconstructed by the scene builder
# (build_scene_from_recording / apply_recorded_dr_to_spec): it re-applies the
# recorded domain-randomization size multipliers onto the spec, keyed by body
# name. That replaces the old positional model_snapshot array copy, which broke
# whenever geom count/order changed (e.g. a changed drawer count).
