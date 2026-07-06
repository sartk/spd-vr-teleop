"""Per-reset object-scale log.

Several scenes apply a single uniform scale factor to their task objects on
each reset (jenga blocks, spell_and_stow letter blocks + cabinet). The scaled
geometry is captured faithfully in the recording's ``model_snapshot``, but the
scalar factor itself is useful as explicit episode metadata — for filtering or
analysing recordings by object scale without reverse-deriving it from
``geom_size`` ratios.

The scale helpers call :func:`record` right after they apply a factor; the
streamer calls :func:`snapshot` when it saves an episode's initial state, and
:func:`reset` is not needed (each ``record`` overwrites the last value).
"""

from __future__ import annotations

_last_scale: dict[str, float] = {}


def record(scene: str, factor: float) -> None:
    """Note that ``scene``'s task objects were scaled by ``factor`` this reset."""
    _last_scale[str(scene)] = float(factor)


def snapshot() -> dict[str, float]:
    """The most recent scale factor recorded for each scene, for episode
    metadata. Empty if no scene applied a scale this run."""
    return dict(_last_scale)
