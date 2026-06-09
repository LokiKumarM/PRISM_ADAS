"""Layer B: situation rules R1..R5.

Each rule is a tiny pure function. Given a ``DetectedObject`` and its
``Relations`` plus the rules dict, it returns either a ``Finding`` or
``None``. The arbiter picks the winner across all findings.

R5 (CRUISE fallback) is special — it has no per-object body and is emitted
once per frame by ``apply_rules``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.object_list import DetectedObject
from src.reasoning.relations import (
    ADJACENT_LANE,
    EGO_LANE,
    Relations,
    relations_for,
)

# Action labels (string-typed so YAML / JSON / UI can pass them around).
ACTION_BRAKE = "BRAKE"
ACTION_STOP = "STOP"
ACTION_YIELD = "YIELD"
ACTION_FOLLOW = "FOLLOW"
ACTION_INHIBIT_LANE_CHANGE = "INHIBIT_LANE_CHANGE"
ACTION_CRUISE = "CRUISE"

BRAKE_CLASSES = {"VEHICLE", "LARGE_VEHICLE", "CYCLIST", "PEDESTRIAN"}
FOLLOW_CLASSES = {"VEHICLE", "LARGE_VEHICLE"}


@dataclass
class Finding:
    """One rule firing on one object (or fallback)."""
    action: str
    priority: int
    reason: str
    object_id: Optional[str] = None
    rule: str = ""  # rule id, e.g. "R1_BRAKE"


# ----------------------------- individual rules -----------------------------


def r1_brake(obj: DetectedObject, rel: Relations, rules: dict[str, Any]) -> Optional[Finding]:
    if rel.zone != EGO_LANE or not rel.ahead:
        return None
    if obj.cls not in BRAKE_CLASSES:
        return None
    if obj.state == "PARKED":
        return None
    brake_distance = float(rules["brake_distance_m"])
    brake_ttc = float(rules["brake_ttc_s"])
    fired_distance = rel.gap < brake_distance
    fired_ttc = rel.ttc < brake_ttc
    if not (fired_distance or fired_ttc):
        return None
    why = []
    if fired_distance:
        why.append(f"gap {rel.gap:.1f}m < {brake_distance:.1f}m")
    if fired_ttc:
        why.append(f"ttc {rel.ttc:.1f}s < {brake_ttc:.1f}s")
    reason = (
        f"BRAKE: {obj.cls} ({obj.state}) in ego lane "
        f"at {rel.gap:.1f}m ahead — " + ", ".join(why)
    )
    return Finding(
        action=ACTION_BRAKE,
        priority=100,
        reason=reason,
        object_id=obj.id,
        rule="R1_BRAKE",
    )


def r2_pedestrian(obj: DetectedObject, rel: Relations, rules: dict[str, Any]) -> Optional[Finding]:
    if obj.cls != "PEDESTRIAN":
        return None
    stop_r = float(rules["ped_stop_radius_m"])
    caution_r = float(rules["ped_caution_radius_m"])
    if obj.distance < stop_r:
        return Finding(
            action=ACTION_STOP,
            priority=95,
            reason=(
                f"STOP: pedestrian at {obj.distance:.1f}m "
                f"(< stop radius {stop_r:.1f}m), state={obj.state}"
            ),
            object_id=obj.id,
            rule="R2_PED_STOP",
        )
    if obj.distance < caution_r:
        return Finding(
            action=ACTION_YIELD,
            priority=80,
            reason=(
                f"YIELD: pedestrian at {obj.distance:.1f}m "
                f"(< caution radius {caution_r:.1f}m), state={obj.state}"
            ),
            object_id=obj.id,
            rule="R2_PED_YIELD",
        )
    return None


def r3_follow(obj: DetectedObject, rel: Relations, rules: dict[str, Any]) -> Optional[Finding]:
    if obj.cls not in FOLLOW_CLASSES:
        return None
    if rel.zone != EGO_LANE or not rel.ahead:
        return None
    if obj.state != "MOVING":
        return None
    follow_d = float(rules["follow_distance_m"])
    if rel.gap >= follow_d:
        return None
    return Finding(
        action=ACTION_FOLLOW,
        priority=50,
        reason=(
            f"FOLLOW: moving {obj.cls.lower()} ahead at {rel.gap:.1f}m "
            f"(< follow distance {follow_d:.1f}m)"
        ),
        object_id=obj.id,
        rule="R3_FOLLOW",
    )


def r4_inhibit_lane_change(
    obj: DetectedObject, rel: Relations, rules: dict[str, Any]
) -> Optional[Finding]:
    if rel.zone != ADJACENT_LANE:
        return None
    if obj.state != "MOVING":
        return None
    side = "left" if obj.y > 0 else "right"
    return Finding(
        action=ACTION_INHIBIT_LANE_CHANGE,
        priority=60,
        reason=(
            f"INHIBIT_LANE_CHANGE: moving {obj.cls.lower()} in {side} adjacent lane "
            f"at ({obj.x:+.1f}, {obj.y:+.1f})"
        ),
        object_id=obj.id,
        rule="R4_INHIBIT_LC",
    )


# Per-object rules in evaluation order (used as tie-breaker).
PER_OBJECT_RULES: list[Callable[[DetectedObject, Relations, dict[str, Any]], Optional[Finding]]] = [
    r1_brake,
    r2_pedestrian,
    r3_follow,
    r4_inhibit_lane_change,
]


def cruise_fallback() -> Finding:
    """R5 — emitted every frame; lowest priority."""
    return Finding(
        action=ACTION_CRUISE,
        priority=1,
        reason="CRUISE: no hazards or follow targets in the ego lane",
        object_id=None,
        rule="R5_CRUISE",
    )


def apply_rules(
    objects: list[DetectedObject], rules: dict[str, Any]
) -> list[Finding]:
    """Run R1..R4 on every object, then append R5 fallback."""
    findings: list[Finding] = []
    for obj in objects:
        rel = relations_for(obj, rules)
        for rule_fn in PER_OBJECT_RULES:
            f = rule_fn(obj, rel, rules)
            if f is not None:
                findings.append(f)
    findings.append(cruise_fallback())
    return findings
