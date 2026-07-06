"""Persisted user-tunable state and per-scene daily metrics for the VR streamer.

Lives in `$HOME/.vr_streamer/`:

  table_height.json        — user's chosen vertical scene offset (meters).
  YYYYMMDD/<scene>.json    — per-day, per-scene recording totals
                             ({"minutes": float, "episodes": int}).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


DEFAULT_HEIGHT_OFFSET = 0.0
HEIGHT_BOUNDS = (-0.6, 0.5)  # meters; final scene lift relative to the default
TABLE_HEIGHT_FILE = "table_height.json"


def state_dir() -> Path:
    return Path.home() / ".vr_streamer"


def clamp_height_offset(value: float) -> float:
    lo, hi = HEIGHT_BOUNDS
    return max(lo, min(hi, float(value)))


def load_table_height_offset() -> float:
    path = state_dir() / TABLE_HEIGHT_FILE
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return DEFAULT_HEIGHT_OFFSET
    raw = data.get("offset", DEFAULT_HEIGHT_OFFSET)
    try:
        return clamp_height_offset(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_HEIGHT_OFFSET


def save_table_height_offset(value: float) -> None:
    clamped = clamp_height_offset(value)
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (TABLE_HEIGHT_FILE + ".tmp")
    tmp.write_text(json.dumps({"offset": clamped}))
    tmp.replace(d / TABLE_HEIGHT_FILE)


# --------------------------------------------------------------------------- #
# Recording metrics                                                           #
# --------------------------------------------------------------------------- #


def _safe_scene_filename(scene_type: str) -> str:
    # Filesystem-safe; scene_types are expected to be alphanumeric/underscore
    # but we strip anything else just in case.
    return "".join(c for c in scene_type if c.isalnum() or c in "_-") + ".json"


def record_episode_metric(scene_type: str, minutes: float, episodes: int = 1) -> None:
    """Append `minutes` and `episodes` to today's per-scene metric file."""
    if not scene_type:
        return
    today = _dt.date.today().strftime("%Y%m%d")
    day_dir = state_dir() / today
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / _safe_scene_filename(scene_type)
    existing: dict = {}
    try:
        existing = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    total_minutes = float(existing.get("minutes", 0.0)) + max(0.0, float(minutes))
    total_episodes = int(existing.get("episodes", 0)) + max(0, int(episodes))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"minutes": total_minutes, "episodes": total_episodes}))
    tmp.replace(path)


def load_scene_metrics(scene_type: str, days: int = 7) -> list[tuple[_dt.date, float, int]]:
    """Return `(date, minutes, episodes)` for the last `days` days, oldest first.

    Days with no recording return zeros. Missing/corrupt files count as zero.
    """
    if days < 1:
        return []
    today = _dt.date.today()
    fname = _safe_scene_filename(scene_type)
    out: list[tuple[_dt.date, float, int]] = []
    for offset in range(days - 1, -1, -1):
        day = today - _dt.timedelta(days=offset)
        path = state_dir() / day.strftime("%Y%m%d") / fname
        minutes = 0.0
        episodes = 0
        try:
            data = json.loads(path.read_text())
            minutes = float(data.get("minutes", 0.0))
            episodes = int(data.get("episodes", 0))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
        out.append((day, minutes, episodes))
    return out


def format_minutes(minutes: float) -> str:
    """Format e.g. 73.4 -> '1h 13m'."""
    total_minutes = max(0, int(round(minutes)))
    h, m = divmod(total_minutes, 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"
