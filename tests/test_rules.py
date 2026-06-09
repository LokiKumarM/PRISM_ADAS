"""Synthetic-input tests for the reasoning rules (R1..R5).

These tests construct ``DetectedObject`` instances directly; they do not
require the nuScenes JSON folder.
"""
from __future__ import annotations

from typing import Any

import pytest

from src.object_list import DetectedObject
from src.reasoning.relations import relations_for
from src.reasoning.rules import (
    ACTION_BRAKE,
    ACTION_CRUISE,
    ACTION_FOLLOW,
    ACTION_INHIBIT_LANE_CHANGE,
    ACTION_STOP,
    ACTION_YIELD,
    Finding,
    apply_rules,
    r1_brake,
    r2_pedestrian,
    r3_follow,
    r4_inhibit_lane_change,
)


RULES: dict[str, Any] = {
    "lane_half_width_m": 1.75,
    "adjacent_lane_max_m": 5.25,
    "follow_distance_m": 40.0,
    "brake_distance_m": 12.0,
    "brake_ttc_s": 2.5,
    "ped_caution_radius_m": 15.0,
    "ped_stop_radius_m": 8.0,
    "min_lidar_pts": 0,
    "min_visibility": 0,
}


def mk_obj(
    *,
    id_: str = "i",
    cls: str = "VEHICLE",
    state: str = "MOVING",
    x: float = 10.0,
    y: float = 0.0,
    vx: float | None = None,
    vy: float | None = None,
) -> DetectedObject:
    import math
    return DetectedObject(
        id=id_,
        raw_category="vehicle.car",
        cls=cls,
        state=state,
        x=x,
        y=y,
        distance=math.hypot(x, y),
        yaw=0.0,
        size=(2.0, 4.5, 1.7),
        vx=vx,
        vy=vy,
        num_lidar_pts=50,
        visibility=4,
    )


def rels(o):
    return relations_for(o, RULES)


# ----------------------------- R1 BRAKE -----------------------------------


def test_r1_brake_fires_on_close_gap():
    o = mk_obj(x=8.0, y=0.0)  # in lane, ahead, gap < 12
    f = r1_brake(o, rels(o), RULES)
    assert f is not None
    assert f.action == ACTION_BRAKE
    assert f.priority == 100
    assert "gap" in f.reason


def test_r1_brake_fires_on_low_ttc():
    # 20 m ahead but closing at 10 m/s → ttc = 2 s < 2.5 s
    o = mk_obj(x=20.0, y=0.0, vx=-10.0)
    f = r1_brake(o, rels(o), RULES)
    assert f is not None
    assert "ttc" in f.reason


def test_r1_brake_silent_on_parked_in_lane():
    o = mk_obj(x=8.0, y=0.0, state="PARKED")
    assert r1_brake(o, rels(o), RULES) is None


def test_r1_brake_silent_on_static_class():
    o = mk_obj(x=8.0, y=0.0, cls="STATIC", state="STANDING")
    assert r1_brake(o, rels(o), RULES) is None


def test_r1_brake_silent_behind_ego():
    o = mk_obj(x=-8.0, y=0.0)
    assert r1_brake(o, rels(o), RULES) is None


def test_r1_brake_silent_when_far_and_not_closing():
    o = mk_obj(x=30.0, y=0.0, vx=0.0)
    assert r1_brake(o, rels(o), RULES) is None


# --------------------------- R2 PEDESTRIAN --------------------------------


def test_r2_pedestrian_stop_radius():
    o = mk_obj(cls="PEDESTRIAN", state="MOVING", x=5.0, y=2.0)
    f = r2_pedestrian(o, rels(o), RULES)
    assert f is not None and f.action == ACTION_STOP and f.priority == 95


def test_r2_pedestrian_caution_radius():
    o = mk_obj(cls="PEDESTRIAN", state="STANDING", x=10.0, y=4.0)  # d≈10.77
    f = r2_pedestrian(o, rels(o), RULES)
    assert f is not None and f.action == ACTION_YIELD and f.priority == 80


def test_r2_pedestrian_far_away_silent():
    o = mk_obj(cls="PEDESTRIAN", x=20.0, y=0.0)
    assert r2_pedestrian(o, rels(o), RULES) is None


def test_r2_pedestrian_only_pedestrians():
    o = mk_obj(cls="VEHICLE", x=3.0, y=0.0)
    assert r2_pedestrian(o, rels(o), RULES) is None


# ----------------------------- R3 FOLLOW -----------------------------------


def test_r3_follow_in_lane_moving_vehicle():
    o = mk_obj(x=25.0, y=0.5, state="MOVING")
    f = r3_follow(o, rels(o), RULES)
    assert f is not None and f.action == ACTION_FOLLOW and f.priority == 50


def test_r3_follow_silent_stopped():
    o = mk_obj(x=25.0, y=0.0, state="STOPPED")
    assert r3_follow(o, rels(o), RULES) is None


def test_r3_follow_silent_too_far():
    o = mk_obj(x=60.0, y=0.0, state="MOVING")
    assert r3_follow(o, rels(o), RULES) is None


def test_r3_follow_silent_adjacent_lane():
    o = mk_obj(x=25.0, y=3.5, state="MOVING")  # adjacent
    assert r3_follow(o, rels(o), RULES) is None


# -------------------- R4 INHIBIT_LANE_CHANGE -------------------------------


def test_r4_inhibit_fires_on_moving_adjacent():
    o = mk_obj(x=5.0, y=-3.5, state="MOVING")  # negative y → right side
    f = r4_inhibit_lane_change(o, rels(o), RULES)
    assert f is not None
    assert f.action == ACTION_INHIBIT_LANE_CHANGE
    assert f.priority == 60
    assert "right" in f.reason


def test_r4_inhibit_left_vs_right():
    left = mk_obj(x=2.0, y=+3.5, state="MOVING")
    right = mk_obj(x=2.0, y=-3.5, state="MOVING")
    assert "left" in r4_inhibit_lane_change(left, rels(left), RULES).reason
    assert "right" in r4_inhibit_lane_change(right, rels(right), RULES).reason


def test_r4_inhibit_silent_on_stopped():
    o = mk_obj(x=2.0, y=3.5, state="STOPPED")
    assert r4_inhibit_lane_change(o, rels(o), RULES) is None


def test_r4_inhibit_silent_offroad():
    o = mk_obj(x=2.0, y=8.0, state="MOVING")
    assert r4_inhibit_lane_change(o, rels(o), RULES) is None


# --------------------------- apply_rules ----------------------------------


def test_apply_rules_always_emits_cruise_fallback():
    findings = apply_rules([], RULES)
    assert any(f.action == ACTION_CRUISE and f.priority == 1 for f in findings)


def test_apply_rules_collects_multiple_findings():
    objs = [
        mk_obj(id_="brake-target", x=6.0, y=0.0, state="MOVING"),  # R1
        mk_obj(id_="ped",          cls="PEDESTRIAN", x=3.0, y=1.0, state="STANDING"),  # R2 STOP
        mk_obj(id_="adj",          x=3.0, y=-3.5, state="MOVING"),  # R4
    ]
    findings = apply_rules(objs, RULES)
    actions = {f.action for f in findings}
    assert ACTION_BRAKE in actions
    assert ACTION_STOP in actions
    assert ACTION_INHIBIT_LANE_CHANGE in actions
    assert ACTION_CRUISE in actions


def test_apply_rules_no_brake_for_parked_in_lane():
    objs = [mk_obj(id_="parked", x=6.0, y=0.0, state="PARKED")]
    findings = apply_rules(objs, RULES)
    actions = [f.action for f in findings]
    assert ACTION_BRAKE not in actions
    assert ACTION_CRUISE in actions
