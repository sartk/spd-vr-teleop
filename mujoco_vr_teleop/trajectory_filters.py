"""Reusable trajectory filtering utilities.

The functions here intentionally avoid viewer-specific code. They operate on a
MuJoCo model plus recorded arrays and return masks/segments that callers can
visualize, write to a manifest, or use for dataset clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Sequence
from pathlib import Path

import mujoco
import numpy as np

CONTACT_FILTER_CACHE_VERSION = 1
GROUND_CONTACT_NAME_TOKENS = ("ground", "floor", "table", "wall", "workcell", "plane")


@dataclass(frozen=True)
class FrameSpan:
    """Half-open frame span [start, end)."""

    start: int
    end: int
    duration_s: float


@dataclass(frozen=True)
class HandObjectContactFilterResult:
    """Per-frame hand/object contact result and cut proposal."""

    contact_mask: np.ndarray
    cut_mask: np.ndarray
    no_contact_spans: tuple[FrameSpan, ...]
    cut_spans: tuple[FrameSpan, ...]
    threshold_s: float
    pre_contact_buffer_s: float
    post_contact_buffer_s: float
    frame_dt_s: float
    hand_geom_ids: tuple[int, ...]
    object_geom_ids: tuple[int, ...]
    exclude_ground_contact: bool = False

    @property
    def contact_buffer_s(self) -> float:
        if abs(self.pre_contact_buffer_s - self.post_contact_buffer_s) < 1e-12:
            return float(self.pre_contact_buffer_s)
        return float(max(self.pre_contact_buffer_s, self.post_contact_buffer_s))

    @property
    def cut_frame_count(self) -> int:
        return int(np.count_nonzero(self.cut_mask))

    @property
    def cut_duration_s(self) -> float:
        return float(self.cut_frame_count * self.frame_dt_s)

    @property
    def longest_no_contact_s(self) -> float:
        if not self.no_contact_spans:
            return 0.0
        return float(max(span.duration_s for span in self.no_contact_spans))


@dataclass(frozen=True)
class WristJumpClipDecision:
    """Whole-clip rejection decision for wrist target discontinuities."""

    reject: bool
    reason: str
    max_jump_m: float
    first_trigger_frame0: int | None
    first_trigger_frame1: int | None
    first_trigger_jump_m: float
    first_trigger_side: str
    first_trigger_in_contact: bool
    nonfinite_count: int
    transition_count: int


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _is_ground_or_workcell_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    body_id = int(model.geom_bodyid[geom_id])
    text = f"{_geom_name(model, geom_id)} {_body_name(model, body_id)}".lower()
    return any(token in text for token in GROUND_CONTACT_NAME_TOKENS)


def _body_descendants(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    descendants: set[int] = set()
    for body_id in range(model.nbody):
        cur = body_id
        while cur > 0:
            if cur == root_body_id:
                descendants.add(body_id)
                break
            cur = int(model.body_parentid[cur])
    return descendants


def infer_hand_geom_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Infer collision geoms belonging to dexterous hands.

    This mirrors the online checkpoint blocker in ``vr_streamer``. It is broad
    enough for Sharpa-style hands while avoiding arm links.
    """

    hand_tokens = ("thumb", "index", "middle", "ring", "pinky", "palm")
    hand_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _body_name(model, body_id).lower()
        is_sharpa_mount = "sharpa" in body_name
        is_finger = (
            (body_name.startswith("left_") or body_name.startswith("right_"))
            and any(token in body_name for token in hand_tokens)
        )
        if is_sharpa_mount or is_finger:
            hand_geom_ids.add(geom_id)
    return tuple(sorted(hand_geom_ids))


def infer_robot_body_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Infer robot body ids so they can be excluded from manipulated objects."""

    robot_body_ids: set[int] = set()
    for root_name in ("left-arm", "right-arm"):
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        if root_id >= 0:
            robot_body_ids.update(_body_descendants(model, root_id))
    for body_id in range(model.nbody):
        body_name = _body_name(model, body_id).lower()
        if "sharpa" in body_name:
            robot_body_ids.add(body_id)
        if int(model.body_mocapid[body_id]) >= 0:
            robot_body_ids.add(body_id)
    return tuple(sorted(robot_body_ids))


def _body_has_dynamic_ancestor(
    model: mujoco.MjModel,
    body_id: int,
    excluded_body_ids: set[int],
) -> bool:
    cur = body_id
    while cur > 0 and cur not in excluded_body_ids:
        if int(model.body_dofnum[cur]) > 0:
            return True
        cur = int(model.body_parentid[cur])
    return False


def infer_object_geom_ids(
    model: mujoco.MjModel,
    *,
    hand_geom_ids: Sequence[int] | None = None,
    robot_body_ids: Sequence[int] | None = None,
    exclude_ground_contact: bool = False,
) -> tuple[int, ...]:
    """Infer non-robot dynamic geoms that can count as manipulated objects.

    This deliberately avoids task names such as ``jenga_*``. A geom counts as an
    object if it belongs to a non-robot body that has a dynamic ancestor. Static
    workcell geometry such as the table, walls, and floor is excluded.
    """

    hand_geoms = set(hand_geom_ids or infer_hand_geom_ids(model))
    robot_bodies = set(robot_body_ids or infer_robot_body_ids(model))
    object_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        if geom_id in hand_geoms:
            continue
        if exclude_ground_contact and _is_ground_or_workcell_geom(model, geom_id):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id in robot_bodies:
            continue
        if _body_has_dynamic_ancestor(model, body_id, robot_bodies):
            object_geom_ids.add(geom_id)
    return tuple(sorted(object_geom_ids))


def _median_frame_dt(frame_times: np.ndarray, fallback: float = 1.0 / 60.0) -> float:
    if frame_times.size >= 2:
        diffs = np.diff(frame_times.astype(float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            return float(np.median(diffs))
    return float(fallback)


def spans_from_mask(mask: np.ndarray, frame_times: np.ndarray | None = None) -> tuple[FrameSpan, ...]:
    """Convert a boolean mask into half-open spans."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if values.size == 0:
        return ()

    times = (
        np.arange(values.size, dtype=float) * (1.0 / 60.0)
        if frame_times is None
        else np.asarray(frame_times, dtype=float).reshape(-1)
    )
    if times.shape[0] != values.shape[0]:
        raise ValueError("frame_times length must match mask length")
    frame_dt = _median_frame_dt(times)

    spans: list[FrameSpan] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            duration = float((idx - start) * frame_dt)
            spans.append(FrameSpan(start=start, end=idx, duration_s=duration))
            start = None
    if start is not None:
        duration = float((values.size - start) * frame_dt)
        spans.append(FrameSpan(start=start, end=values.size, duration_s=duration))
    return tuple(spans)


def cut_mask_for_no_contact_spans(
    contact_mask: np.ndarray,
    frame_times: np.ndarray,
    *,
    threshold_s: float,
    contact_buffer_s: float = 0.0,
    pre_contact_buffer_s: float | None = None,
    post_contact_buffer_s: float | None = None,
) -> tuple[np.ndarray, tuple[FrameSpan, ...], tuple[FrameSpan, ...], float]:
    """Return buffered cuts inside no-contact spans longer than ``threshold_s``.

    Buffers preserve context around contact. For a middle no-contact span, the
    proposed cut starts ``post_contact_buffer_s`` after the previous contact and
    stops ``pre_contact_buffer_s`` before the next contact. Beginning/end spans
    only apply the buffer next to an actual contact.
    """

    if threshold_s <= 0.0:
        raise ValueError("threshold_s must be > 0")
    if contact_buffer_s < 0.0:
        raise ValueError("contact_buffer_s must be >= 0")
    pre_buffer_s = float(contact_buffer_s if pre_contact_buffer_s is None else pre_contact_buffer_s)
    post_buffer_s = float(contact_buffer_s if post_contact_buffer_s is None else post_contact_buffer_s)
    if pre_buffer_s < 0.0:
        raise ValueError("pre_contact_buffer_s must be >= 0")
    if post_buffer_s < 0.0:
        raise ValueError("post_contact_buffer_s must be >= 0")
    contact = np.asarray(contact_mask, dtype=bool).reshape(-1)
    times = np.asarray(frame_times, dtype=float).reshape(-1)
    if contact.shape[0] != times.shape[0]:
        raise ValueError("frame_times length must match contact_mask length")

    frame_dt = _median_frame_dt(times)
    pre_buffer_frames = int(round(pre_buffer_s / frame_dt)) if pre_buffer_s > 0.0 else 0
    post_buffer_frames = int(round(post_buffer_s / frame_dt)) if post_buffer_s > 0.0 else 0
    no_contact_spans = spans_from_mask(~contact, times)
    cut_spans_list: list[FrameSpan] = []
    cut_mask = np.zeros(contact.shape[0], dtype=bool)
    for span in no_contact_spans:
        has_prev_contact = span.start > 0 and bool(contact[span.start - 1])
        has_next_contact = span.end < contact.shape[0] and bool(contact[span.end])
        cut_start = span.start + (post_buffer_frames if has_prev_contact else 0)
        cut_end = span.end - (pre_buffer_frames if has_next_contact else 0)
        cut_start = min(max(span.start, cut_start), span.end)
        cut_end = min(max(cut_start, cut_end), span.end)
        cut_duration_s = float((cut_end - cut_start) * frame_dt)
        if cut_duration_s > threshold_s:
            cut_span = FrameSpan(
                start=cut_start,
                end=cut_end,
                duration_s=cut_duration_s,
            )
            cut_spans_list.append(cut_span)
            cut_mask[cut_start:cut_end] = True
    return cut_mask, no_contact_spans, tuple(cut_spans_list), frame_dt


def wrist_jump_clip_decision(
    wrist_positions,
    contact_mask: np.ndarray,
    span: FrameSpan,
    *,
    contact_jump_m: float = 0.10,
    severe_jump_m: float = 0.30,
    reject_nonfinite: bool = True,
) -> WristJumpClipDecision:
    """Return whether a kept clip should be rejected for wrist discontinuities.

    The final dataset filter rejects the whole clip if either wrist jumps at
    least ``severe_jump_m`` in one frame, regardless of contact, or jumps at
    least ``contact_jump_m`` while cached hand/object contact is active. A
    transition ``k -> k + 1`` counts as contact-active when either endpoint
    frame is in contact.
    """

    if contact_jump_m < 0.0:
        raise ValueError("contact_jump_m must be >= 0")
    if severe_jump_m < 0.0:
        raise ValueError("severe_jump_m must be >= 0")
    if span.end < span.start:
        raise ValueError("span end must be >= start")

    contact = np.asarray(contact_mask, dtype=bool).reshape(-1)
    if span.end > contact.shape[0]:
        raise ValueError("span extends past contact mask")

    frame_count = int(span.end - span.start)
    if frame_count <= 0:
        return WristJumpClipDecision(
            reject=False,
            reason="",
            max_jump_m=0.0,
            first_trigger_frame0=None,
            first_trigger_frame1=None,
            first_trigger_jump_m=0.0,
            first_trigger_side="",
            first_trigger_in_contact=False,
            nonfinite_count=0,
            transition_count=0,
        )

    positions = np.asarray(wrist_positions[span.start : span.end], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "wrist_positions must have shape (frames, wrists, 3); "
            f"got {positions.shape}"
        )
    if positions.shape[0] != frame_count:
        raise ValueError(
            f"wrist position slice length {positions.shape[0]} does not match span length {frame_count}"
        )

    if frame_count <= 1:
        nonfinite_count = int(positions.size - np.count_nonzero(np.isfinite(positions)))
        return WristJumpClipDecision(
            reject=bool(reject_nonfinite and nonfinite_count),
            reason="nonfinite_wrist" if reject_nonfinite and nonfinite_count else "",
            max_jump_m=0.0,
            first_trigger_frame0=span.start if reject_nonfinite and nonfinite_count else None,
            first_trigger_frame1=span.start if reject_nonfinite and nonfinite_count else None,
            first_trigger_jump_m=0.0,
            first_trigger_side="",
            first_trigger_in_contact=bool(contact[span.start]) if span.start < contact.size else False,
            nonfinite_count=nonfinite_count,
            transition_count=0,
        )

    delta = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    finite = np.isfinite(delta)
    nonfinite_count = int(delta.size - np.count_nonzero(finite))
    safe_delta = np.where(finite, delta, -np.inf)
    max_by_transition = np.max(safe_delta, axis=1)
    side_by_transition = np.argmax(safe_delta, axis=1)
    finite_max = max_by_transition[np.isfinite(max_by_transition)]
    max_jump_m = float(np.max(finite_max)) if finite_max.size else 0.0

    if reject_nonfinite and nonfinite_count:
        bad_transition = int(np.flatnonzero(~np.all(finite, axis=1))[0])
        return WristJumpClipDecision(
            reject=True,
            reason="nonfinite_wrist",
            max_jump_m=max_jump_m,
            first_trigger_frame0=int(span.start + bad_transition),
            first_trigger_frame1=int(span.start + bad_transition + 1),
            first_trigger_jump_m=0.0,
            first_trigger_side="",
            first_trigger_in_contact=bool(
                contact[span.start + bad_transition] or contact[span.start + bad_transition + 1]
            ),
            nonfinite_count=nonfinite_count,
            transition_count=int(delta.shape[0]),
        )

    transition_contact = contact[span.start : span.end - 1] | contact[span.start + 1 : span.end]
    severe_mask = max_by_transition >= float(severe_jump_m)
    contact_jump_mask = (max_by_transition >= float(contact_jump_m)) & transition_contact
    trigger_mask = severe_mask | contact_jump_mask
    if not np.any(trigger_mask):
        return WristJumpClipDecision(
            reject=False,
            reason="",
            max_jump_m=max_jump_m,
            first_trigger_frame0=None,
            first_trigger_frame1=None,
            first_trigger_jump_m=0.0,
            first_trigger_side="",
            first_trigger_in_contact=False,
            nonfinite_count=nonfinite_count,
            transition_count=int(delta.shape[0]),
        )

    first = int(np.flatnonzero(trigger_mask)[0])
    side = int(side_by_transition[first])
    return WristJumpClipDecision(
        reject=True,
        reason="severe_wrist_jump" if bool(severe_mask[first]) else "contact_wrist_jump",
        max_jump_m=max_jump_m,
        first_trigger_frame0=int(span.start + first),
        first_trigger_frame1=int(span.start + first + 1),
        first_trigger_jump_m=float(max_by_transition[first]),
        first_trigger_side="right" if side == 0 else "left" if side == 1 else str(side),
        first_trigger_in_contact=bool(transition_contact[first]),
        nonfinite_count=nonfinite_count,
        transition_count=int(delta.shape[0]),
    )


def compute_hand_object_contact_filter(
    model: mujoco.MjModel,
    qpos,
    frame_times,
    *,
    threshold_s: float = 4.0,
    contact_buffer_s: float = 0.0,
    pre_contact_buffer_s: float | None = None,
    post_contact_buffer_s: float | None = None,
    exclude_ground_contact: bool = False,
    qvel=None,
    progress: Callable[[int, int], None] | None = None,
) -> HandObjectContactFilterResult:
    """Compute proposed cut spans from recorded hand/object contacts.

    Parameters are array-like so callers can pass zarr arrays without loading
    the entire trajectory into memory.
    """

    if threshold_s <= 0.0:
        raise ValueError("threshold_s must be > 0")
    pre_buffer_s = float(contact_buffer_s if pre_contact_buffer_s is None else pre_contact_buffer_s)
    post_buffer_s = float(contact_buffer_s if post_contact_buffer_s is None else post_contact_buffer_s)

    qpos_arr = qpos
    frame_count = int(qpos_arr.shape[0])
    times = np.asarray(frame_times, dtype=float).reshape(-1)
    if times.shape[0] != frame_count:
        raise ValueError("frame_times length must match qpos frame count")

    hand_geom_ids = infer_hand_geom_ids(model)
    robot_body_ids = infer_robot_body_ids(model)
    object_geom_ids = infer_object_geom_ids(
        model,
        hand_geom_ids=hand_geom_ids,
        robot_body_ids=robot_body_ids,
        exclude_ground_contact=exclude_ground_contact,
    )
    hand_geoms = set(hand_geom_ids)
    object_geoms = set(object_geom_ids)
    if not hand_geoms:
        raise ValueError("could not infer hand geoms for contact filtering")
    if not object_geoms:
        raise ValueError("could not infer object geoms for contact filtering")

    data = mujoco.MjData(model)
    contact_mask = np.zeros(frame_count, dtype=bool)
    progress_stride = max(1, frame_count // 20)

    for frame_idx in range(frame_count):
        data.qpos[:] = np.asarray(qpos_arr[frame_idx], dtype=float)
        if qvel is not None:
            data.qvel[:] = np.asarray(qvel[frame_idx], dtype=float)
        else:
            data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 in hand_geoms and geom2 in object_geoms) or (
                geom2 in hand_geoms and geom1 in object_geoms
            ):
                contact_mask[frame_idx] = True
                break
        if progress is not None and (
            frame_idx == 0 or (frame_idx + 1) % progress_stride == 0 or frame_idx + 1 == frame_count
        ):
            progress(frame_idx + 1, frame_count)

    cut_mask, no_contact_spans, cut_spans, frame_dt = cut_mask_for_no_contact_spans(
        contact_mask,
        times,
        threshold_s=threshold_s,
        contact_buffer_s=contact_buffer_s,
        pre_contact_buffer_s=pre_buffer_s,
        post_contact_buffer_s=post_buffer_s,
    )
    return HandObjectContactFilterResult(
        contact_mask=contact_mask,
        cut_mask=cut_mask,
        no_contact_spans=no_contact_spans,
        cut_spans=cut_spans,
        threshold_s=float(threshold_s),
        pre_contact_buffer_s=pre_buffer_s,
        post_contact_buffer_s=post_buffer_s,
        frame_dt_s=frame_dt,
        hand_geom_ids=hand_geom_ids,
        object_geom_ids=object_geom_ids,
        exclude_ground_contact=bool(exclude_ground_contact),
    )


def _span_to_dict(span: FrameSpan) -> dict[str, float | int]:
    return {
        "start": int(span.start),
        "end": int(span.end),
        "duration_s": float(span.duration_s),
    }


def _span_from_dict(payload: dict[str, object]) -> FrameSpan:
    return FrameSpan(
        start=int(payload["start"]),
        end=int(payload["end"]),
        duration_s=float(payload["duration_s"]),
    )


def _mask_from_spans(spans: Sequence[FrameSpan], frame_count: int) -> np.ndarray:
    mask = np.zeros(int(frame_count), dtype=bool)
    for span in spans:
        start = min(max(0, int(span.start)), frame_count)
        end = min(max(start, int(span.end)), frame_count)
        mask[start:end] = True
    return mask


def contact_filter_cache_key(
    traj_path: str | Path,
    threshold_s: float,
    contact_buffer_s: float = 0.0,
    *,
    pre_contact_buffer_s: float | None = None,
    post_contact_buffer_s: float | None = None,
    exclude_ground_contact: bool = False,
) -> str:
    resolved = str(Path(traj_path).expanduser().resolve())
    payload = f"{resolved}\0{float(threshold_s):.9f}".encode("utf-8")
    if pre_contact_buffer_s is not None or post_contact_buffer_s is not None:
        pre_buffer_s = float(contact_buffer_s if pre_contact_buffer_s is None else pre_contact_buffer_s)
        post_buffer_s = float(contact_buffer_s if post_contact_buffer_s is None else post_contact_buffer_s)
        payload += f"\0pre={pre_buffer_s:.9f}\0post={post_buffer_s:.9f}".encode("utf-8")
    elif abs(float(contact_buffer_s)) > 1e-12:
        payload += f"\0{float(contact_buffer_s):.9f}".encode("utf-8")
    if exclude_ground_contact:
        payload += b"\0exclude_ground_contact=1"
    return hashlib.sha256(payload).hexdigest()[:24]


def contact_filter_cache_path(
    cache_dir: str | Path,
    traj_path: str | Path,
    threshold_s: float,
    contact_buffer_s: float = 0.0,
    *,
    pre_contact_buffer_s: float | None = None,
    post_contact_buffer_s: float | None = None,
    exclude_ground_contact: bool = False,
) -> Path:
    return (
        Path(cache_dir).expanduser().resolve()
        / "episodes"
        / f"{contact_filter_cache_key(traj_path, threshold_s, contact_buffer_s, pre_contact_buffer_s=pre_contact_buffer_s, post_contact_buffer_s=post_contact_buffer_s, exclude_ground_contact=exclude_ground_contact)}.json"
    )


def contact_filter_to_cache_dict(
    result: HandObjectContactFilterResult,
    *,
    traj_path: str | Path,
    frame_count: int,
    frame_times,
    method: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    times = np.asarray(frame_times, dtype=float).reshape(-1)
    contact_spans = spans_from_mask(result.contact_mask, times)
    payload: dict[str, object] = {
        "version": CONTACT_FILTER_CACHE_VERSION,
        "kind": "hand_object_no_contact_filter",
        "status": "ok",
        "method": method,
        "episode_path": str(traj_path),
        "resolved_episode_path": str(Path(traj_path).expanduser().resolve()),
        "threshold_s": float(result.threshold_s),
        "contact_buffer_s": float(result.contact_buffer_s),
        "pre_contact_buffer_s": float(result.pre_contact_buffer_s),
        "post_contact_buffer_s": float(result.post_contact_buffer_s),
        "frame_count": int(frame_count),
        "frame_dt_s": float(result.frame_dt_s),
        "contact_frame_count": int(np.count_nonzero(result.contact_mask)),
        "cut_frame_count": int(result.cut_frame_count),
        "cut_duration_s": float(result.cut_duration_s),
        "longest_no_contact_s": float(result.longest_no_contact_s),
        "hand_geom_ids": [int(value) for value in result.hand_geom_ids],
        "object_geom_ids": [int(value) for value in result.object_geom_ids],
        "exclude_ground_contact": bool(result.exclude_ground_contact),
        "contact_spans": [_span_to_dict(span) for span in contact_spans],
        "no_contact_spans": [_span_to_dict(span) for span in result.no_contact_spans],
        "cut_spans": [_span_to_dict(span) for span in result.cut_spans],
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def contact_filter_from_cache_dict(
    payload: dict[str, object],
    *,
    expected_frame_count: int | None = None,
    expected_threshold_s: float | None = None,
    expected_contact_buffer_s: float | None = None,
    expected_pre_contact_buffer_s: float | None = None,
    expected_post_contact_buffer_s: float | None = None,
    expected_exclude_ground_contact: bool | None = None,
) -> HandObjectContactFilterResult:
    if payload.get("kind") != "hand_object_no_contact_filter":
        raise ValueError("cache entry is not a hand/object no-contact filter")
    if payload.get("status") != "ok":
        raise ValueError(f"cache entry status is {payload.get('status')!r}")
    version = int(payload.get("version", 0))
    if version != CONTACT_FILTER_CACHE_VERSION:
        raise ValueError(f"unsupported contact filter cache version {version}")

    frame_count = int(payload["frame_count"])
    threshold_s = float(payload["threshold_s"])
    contact_buffer_s = float(payload.get("contact_buffer_s", 0.0))
    pre_contact_buffer_s = float(payload.get("pre_contact_buffer_s", contact_buffer_s))
    post_contact_buffer_s = float(payload.get("post_contact_buffer_s", contact_buffer_s))
    exclude_ground_contact = bool(payload.get("exclude_ground_contact", False))
    if expected_frame_count is not None and frame_count != int(expected_frame_count):
        raise ValueError(
            f"cache frame_count {frame_count} does not match expected {expected_frame_count}"
        )
    if expected_threshold_s is not None and abs(threshold_s - float(expected_threshold_s)) > 1e-6:
        raise ValueError(
            f"cache threshold {threshold_s} does not match expected {expected_threshold_s}"
        )
    if (
        expected_contact_buffer_s is not None
        and abs(contact_buffer_s - float(expected_contact_buffer_s)) > 1e-6
    ):
        raise ValueError(
            f"cache contact_buffer_s {contact_buffer_s} does not match expected "
            f"{expected_contact_buffer_s}"
        )
    if (
        expected_pre_contact_buffer_s is not None
        and abs(pre_contact_buffer_s - float(expected_pre_contact_buffer_s)) > 1e-6
    ):
        raise ValueError(
            f"cache pre_contact_buffer_s {pre_contact_buffer_s} does not match expected "
            f"{expected_pre_contact_buffer_s}"
        )
    if (
        expected_post_contact_buffer_s is not None
        and abs(post_contact_buffer_s - float(expected_post_contact_buffer_s)) > 1e-6
    ):
        raise ValueError(
            f"cache post_contact_buffer_s {post_contact_buffer_s} does not match expected "
            f"{expected_post_contact_buffer_s}"
        )
    if (
        expected_exclude_ground_contact is not None
        and exclude_ground_contact != bool(expected_exclude_ground_contact)
    ):
        raise ValueError(
            f"cache exclude_ground_contact {exclude_ground_contact} does not match expected "
            f"{bool(expected_exclude_ground_contact)}"
        )

    contact_spans = tuple(
        _span_from_dict(item) for item in payload.get("contact_spans", [])
    )
    no_contact_spans = tuple(
        _span_from_dict(item) for item in payload.get("no_contact_spans", [])
    )
    cut_spans = tuple(_span_from_dict(item) for item in payload.get("cut_spans", []))
    return HandObjectContactFilterResult(
        contact_mask=_mask_from_spans(contact_spans, frame_count),
        cut_mask=_mask_from_spans(cut_spans, frame_count),
        no_contact_spans=no_contact_spans,
        cut_spans=cut_spans,
        threshold_s=threshold_s,
        pre_contact_buffer_s=pre_contact_buffer_s,
        post_contact_buffer_s=post_contact_buffer_s,
        frame_dt_s=float(payload["frame_dt_s"]),
        hand_geom_ids=tuple(int(value) for value in payload.get("hand_geom_ids", [])),
        object_geom_ids=tuple(int(value) for value in payload.get("object_geom_ids", [])),
        exclude_ground_contact=exclude_ground_contact,
    )


def load_contact_filter_cache(
    cache_dir: str | Path,
    traj_path: str | Path,
    threshold_s: float,
    *,
    contact_buffer_s: float = 0.0,
    pre_contact_buffer_s: float | None = None,
    post_contact_buffer_s: float | None = None,
    exclude_ground_contact: bool = False,
    expected_frame_count: int | None = None,
) -> HandObjectContactFilterResult | None:
    path = contact_filter_cache_path(
        cache_dir,
        traj_path,
        threshold_s,
        contact_buffer_s,
        pre_contact_buffer_s=pre_contact_buffer_s,
        post_contact_buffer_s=post_contact_buffer_s,
        exclude_ground_contact=exclude_ground_contact,
    )
    if not path.exists():
        return None
    with path.open("r") as f:
        payload = json.load(f)
    return contact_filter_from_cache_dict(
        payload,
        expected_frame_count=expected_frame_count,
        expected_threshold_s=threshold_s,
        expected_contact_buffer_s=(
            contact_buffer_s
            if pre_contact_buffer_s is None and post_contact_buffer_s is None
            else None
        ),
        expected_pre_contact_buffer_s=pre_contact_buffer_s,
        expected_post_contact_buffer_s=post_contact_buffer_s,
        expected_exclude_ground_contact=exclude_ground_contact,
    )


def save_contact_filter_cache(
    cache_dir: str | Path,
    traj_path: str | Path,
    threshold_s: float,
    payload: dict[str, object],
    *,
    contact_buffer_s: float = 0.0,
    pre_contact_buffer_s: float | None = None,
    post_contact_buffer_s: float | None = None,
    exclude_ground_contact: bool = False,
) -> Path:
    path = contact_filter_cache_path(
        cache_dir,
        traj_path,
        threshold_s,
        contact_buffer_s,
        pre_contact_buffer_s=pre_contact_buffer_s,
        post_contact_buffer_s=post_contact_buffer_s,
        exclude_ground_contact=exclude_ground_contact,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)
    return path
