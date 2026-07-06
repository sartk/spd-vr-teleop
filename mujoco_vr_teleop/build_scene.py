from __future__ import annotations

from pathlib import Path

import tyro

from mujoco_vr_teleop.scene_builder import (
    BuildSceneConfig,
    build_scene,
)


def build_scene_file(config: BuildSceneConfig) -> Path:
    return build_scene(config).xml_path


def main() -> None:
    print(build_scene_file(tyro.cli(BuildSceneConfig)))


if __name__ == "__main__":
    main()
