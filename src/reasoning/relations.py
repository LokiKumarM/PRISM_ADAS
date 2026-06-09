"""Layer A: spatial relations.

Pure functions over a ``DetectedObject`` and a rules dict. Each function is
small, deterministic, and side-effect free so it can be unit-tested in
isolation and composed by rules.py.

Ego frame: +x forward, +y left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.object_list import DetectedObject

EGO_LANE = "EGO_LANE"
ADJACENT_LANE = "ADJACENT_LANE"
OFF_ROAD = "OFF_ROAD"


def zone(obj: DetectedObject, rules: dict[str, Any]) -> str:
    half = float(rules["lane_half_width_m"])
    adj_max = float(rules["adjacent_lane_max_m"])
    ay = abs(obj.y)
    if ay <= half:
        return EGO_LANE
    if ay <= adj_max:
        return ADJACENT_LANE
    return OFF_ROAD


def is_ahead(obj: DetectedObject) -> bool:
    return obj.x > 0.0


def gap(obj: DetectedObject) -> float:
    """Forward distance from the ego (signed; positive = ahead)."""
    return obj.x


def closing_speed(obj: DetectedObject) -> float:
    """Rate at which the object is approaching ego along the forward axis.

    Positive means the gap is shrinking. ``None`` velocity (first appearance)
    is treated as zero closing.
    """
    if obj.vx is None:
        return 0.0
    return -float(obj.vx)


def ttc(obj: DetectedObject) -> float:
    """Time-to-collision in seconds, or +inf when not closing."""
    cs = closing_speed(obj)
    g = gap(obj)
    if cs <= 0.0 or g <= 0.0:
        return math.inf
    return g / cs


@dataclass(frozen=True)
class Relations:
    """Bundle of derived spatial facts for one object/frame pair."""
    zone: str
    ahead: bool
    gap: float
    closing_speed: float
    ttc: float


def relations_for(obj: DetectedObject, rules: dict[str, Any]) -> Relations:
    return Relations(
        zone=zone(obj, rules),
        ahead=is_ahead(obj),
        gap=gap(obj),
        closing_speed=closing_speed(obj),
        ttc=ttc(obj),
    )
