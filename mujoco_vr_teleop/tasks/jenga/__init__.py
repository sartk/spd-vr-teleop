"""Per-task TaskSpecs for the jenga scene.

Each sibling module exposes ``TASK: TaskSpec`` with id ``jenga/<task>``.
"""

# Per-scene domain randomization defaults. Overridden per task via
# TaskSpec.randomize_overrides.
SCENE_RANDOMIZATION = {"size": True, "friction": True}
