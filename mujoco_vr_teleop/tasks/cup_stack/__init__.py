"""Per-task TaskSpecs for the cup_stack scene.

Three no-ball stacking tasks: stack_two_threes, unstack, pyramid. Each module
exposes ``TASK: TaskSpec`` with id ``cup_stack/<task>``.

The build is task-driven: variant_pools.CUP_STACK_TASK_PROFILE maps each task
to its cup count. CupStackTaskManager calls variant_pools.set_cup_task() before
each build. cup_stack only uses cup variants that nest cleanly for stacking.
"""

SCENE_RANDOMIZATION = {"size": True}
