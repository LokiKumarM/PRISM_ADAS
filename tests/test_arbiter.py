"""Synthetic-input tests for the arbiter."""
from __future__ import annotations

from src.reasoning.arbiter import decide
from src.reasoning.rules import (
    ACTION_BRAKE,
    ACTION_CRUISE,
    ACTION_FOLLOW,
    ACTION_INHIBIT_LANE_CHANGE,
    ACTION_STOP,
    Finding,
    cruise_fallback,
)


def f(action: str, priority: int, rule: str = "R", reason: str = "test") -> Finding:
    return Finding(action=action, priority=priority, reason=reason, object_id="x", rule=rule)


def test_decide_picks_highest_priority():
    findings = [
        f(ACTION_FOLLOW, 50, "R3"),
        f(ACTION_BRAKE, 100, "R1"),
        f(ACTION_INHIBIT_LANE_CHANGE, 60, "R4"),
        cruise_fallback(),
    ]
    d = decide(findings, frame_token="frame-1", num_objects=3)
    assert d.action == ACTION_BRAKE
    assert d.priority == 100
    assert d.frame_token == "frame-1"
    assert d.num_objects == 3


def test_decide_includes_full_trace_sorted_by_priority():
    findings = [
        f(ACTION_FOLLOW, 50, "R3"),
        f(ACTION_BRAKE, 100, "R1"),
        cruise_fallback(),
    ]
    d = decide(findings, "frame", 1)
    priorities = [sf.priority for sf in d.supporting_findings]
    assert priorities == sorted(priorities, reverse=True)
    assert len(d.supporting_findings) == 3


def test_decide_tiebreak_uses_input_order():
    a = f(ACTION_BRAKE, 100, "R1", reason="first")
    b = f(ACTION_STOP, 100, "R2", reason="second")
    d = decide([a, b], "frame", 0)
    assert d.primary_reason == "first"
    assert d.action == ACTION_BRAKE


def test_decide_defaults_to_cruise_when_empty():
    d = decide([], frame_token="frame", num_objects=0)
    assert d.action == ACTION_CRUISE
    assert d.priority == 1
    assert len(d.supporting_findings) == 1


def test_decide_primary_reason_matches_winner():
    findings = [
        f(ACTION_FOLLOW, 50, reason="moving van 30m"),
        f(ACTION_BRAKE, 100, reason="ped 6m closing"),
        cruise_fallback(),
    ]
    d = decide(findings, "frame", 2)
    assert d.primary_reason == "ped 6m closing"
