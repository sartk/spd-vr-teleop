"""Per-task TaskSpecs for the cup_ball scene.

Three ball tasks: shuffle_ball, pong, playground. Each module exposes
``TASK: TaskSpec`` with id ``cup_ball/<task>``.

The build is task-driven: variant_pools.CUP_BALL_TASK_PROFILE maps each task to
its cup count, ball presence, and (for shuffle) the forced opaque-plastic
family. CupBallTaskManager calls variant_pools.set_cup_task() before each build.
"""

SCENE_RANDOMIZATION = {"size": True}
