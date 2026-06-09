"""Cross-frame velocity for DetectedObjects.

Matches objects between two consecutive keyframes by ``instance_token``
(carried as ``DetectedObject.id``) and differences their ego-frame positions
in time. No nuScenes-devkit ``box_velocity`` — manual differencing only.

The velocity is expressed in the *current* ego frame: positive ``vx`` means
the object is moving away from the ego along its forward axis.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from src.object_list import DetectedObject, EgoState


def dt_seconds(prev_ts: int, curr_ts: int) -> float:
    """nuScenes timestamps are microseconds since the epoch."""
    return max(1e-6, (curr_ts - prev_ts) / 1_000_000.0)


def attach_velocity(
    prev_objects: Optional[Iterable[DetectedObject]],
    curr_objects: list[DetectedObject],
    dt_s: float,
) -> list[DetectedObject]:
    """Mutate ``curr_objects`` in place: set ``vx, vy`` from prev/curr deltas.

    Objects that did not appear in the previous frame keep ``vx = vy = None``.
    Returns the same list for chaining convenience.
    """
    if prev_objects is None or dt_s <= 0:
        return curr_objects
    prev_by_id = {o.id: o for o in prev_objects}
    for o in curr_objects:
        prev = prev_by_id.get(o.id)
        if prev is None:
            continue
        o.vx = (o.x - prev.x) / dt_s
        o.vy = (o.y - prev.y) / dt_s
    return curr_objects


def attach_ego_speed(
    prev_ego: Optional[EgoState],
    curr_ego: EgoState,
) -> EgoState:
    """Estimate ego speed by differencing global ego positions across frames."""
    if prev_ego is None:
        return curr_ego
    dt = dt_seconds(prev_ego.timestamp, curr_ego.timestamp)
    if dt <= 0:
        return curr_ego
    dx = curr_ego.x - prev_ego.x
    dy = curr_ego.y - prev_ego.y
    curr_ego.speed = math.hypot(dx, dy) / dt
    return curr_ego
