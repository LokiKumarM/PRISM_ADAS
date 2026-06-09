"""Synthetic-input tests for the narration layer.

No dataset, no nuScenes JSON. Each test builds DetectedObjects, runs the
existing rules + arbiter to get a real Decision, then exercises narrate().
"""
from __future__ import annotations

import math
from typing import Any

from src.object_list import DetectedObject
from src.reasoning.arbiter import decide
from src.reasoning.narrate import (
    BORDERLINE_RATIO,
    SIGNATURE_QUESTIONS,
    CausalStep,
    EngineerRow,
    Narration,
    SignatureOutput,
    _build_confidence,
    format_causal_chain,
    format_driver_hmi,
    format_engineer_trace,
    format_markdown,
    format_signature_output,
    format_text,
    narrate,
)
from src.reasoning.relations import relations_for
from src.reasoning.rules import (
    ACTION_CRUISE,
    ACTION_FOLLOW,
    ACTION_INHIBIT_LANE_CHANGE,
    ACTION_STOP,
    apply_rules,
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
    id_: str,
    cls: str,
    state: str,
    x: float,
    y: float,
    raw_category: str = "vehicle.car",
) -> DetectedObject:
    return DetectedObject(
        id=id_,
        raw_category=raw_category,
        cls=cls,
        state=state,
        x=x,
        y=y,
        distance=math.hypot(x, y),
        yaw=0.0,
        size=(2.0, 4.5, 1.7),
        num_lidar_pts=50,
        visibility=4,
    )


def _run(objects: list[DetectedObject]) -> tuple[Narration, Any]:
    findings = apply_rules(objects, RULES)
    decision = decide(findings, frame_token="frame", num_objects=len(objects))
    facts = {o.id: relations_for(o, RULES) for o in objects}
    return narrate(decision, objects, facts, RULES), decision


# --------------------------------------------------------------------- TESTS


def test_multi_hazard_frame_stop_with_overrides():
    """Two pedestrians in stop zone + a moving van in follow range +
    a moving car in the right adjacent lane.

    Expectations:
      - PERCEPTION lists the peds under "Crossing/near".
      - REASONING is sorted by descending priority (R2_PED_STOP first).
      - WEIGH step names STOP as winner.
      - ALTERNATIVES include FOLLOW (overridden by STOP).
      - CONFIDENCE == "high" because >=2 STOP findings agree.
    """
    # peds placed just outside the ego lane (|y| > 1.75) so R1 BRAKE doesn't
    # fire; we want to exercise the STOP path here.
    objects = [
        mk_obj(id_="ped-1", cls="PEDESTRIAN", state="STANDING",
               x=4.0, y=2.0, raw_category="human.pedestrian.adult"),
        mk_obj(id_="ped-2", cls="PEDESTRIAN", state="STANDING",
               x=4.5, y=-2.0, raw_category="human.pedestrian.adult"),
        mk_obj(id_="van",   cls="VEHICLE",    state="MOVING", x=33.0, y=0.3),
        mk_obj(id_="adj",   cls="VEHICLE",    state="MOVING", x=5.0,  y=-3.5),
    ]
    n, decision = _run(objects)
    assert decision.action == ACTION_STOP

    # PERCEPTION: at least one line about crossing pedestrians.
    crossing = [l for l in n.perception if "Crossing/near" in l]
    assert len(crossing) == 1
    assert "2 pedestrians" in crossing[0]

    # REASONING: priorities monotonically non-increasing in the listed findings.
    # The numbered lines (excluding the final WEIGH step) carry priorities
    # equal to the supporting_findings order.
    findings_steps = n.reasoning[:-1]
    assert len(findings_steps) == len(decision.supporting_findings)
    priorities = [f.priority for f in decision.supporting_findings]
    assert priorities == sorted(priorities, reverse=True)

    # WEIGH step
    weigh = n.reasoning[-1]
    assert "Weighing these" in weigh
    assert "STOP" in weigh
    assert "is selected" in weigh

    # ALTERNATIVES include FOLLOW with override note
    assert any(alt.startswith("FOLLOW") and "overridden" in alt for alt in n.alternatives)
    assert any(alt.startswith("INHIBIT_LANE_CHANGE") for alt in n.alternatives)

    # CONFIDENCE high (2 STOP findings)
    assert n.confidence == "high"
    assert "agree on STOP" in n.confidence_reason


def test_clear_road_cruise():
    """No objects in the frame -> CRUISE, single reasoning step, unambiguous."""
    n, decision = _run([])
    assert decision.action == ACTION_CRUISE
    assert n.perception == ["Road ahead is clear (no relevant objects)."]
    assert len(n.reasoning) == 1
    assert n.reasoning[0].lower().startswith("no hazards")
    assert n.alternatives == ["No competing actions; decision was unambiguous."]
    # Should not crash and confidence is non-empty.
    assert n.confidence in {"low", "medium", "high"}
    assert n.confidence_reason  # non-empty
    # Formatters do not crash.
    assert "PERCEPTION" in format_text(n)
    assert "Perception" in format_markdown(n)


def test_borderline_trigger_low_confidence():
    """Single PEDESTRIAN at distance 7.6 m vs stop radius 8.0 m
    (placed off the ego lane so R1 BRAKE doesn't fire) -> ratio 0.95 ->
    CONFIDENCE == 'low' (within 10% of threshold)."""
    objects = [
        mk_obj(
            id_="ped", cls="PEDESTRIAN", state="STANDING",
            x=0.0, y=7.6, raw_category="human.pedestrian.adult",
        ),
    ]
    n, decision = _run(objects)
    assert decision.action == ACTION_STOP
    assert n.confidence == "low"
    assert "borderline" in n.confidence_reason
    # Should still produce a complete narration block.
    text = format_text(n)
    assert "DECISION:   STOP" in text


def test_wide_margin_high_confidence():
    """Single PEDESTRIAN at distance 2.5 m vs stop radius 8.0 m
    (placed in adjacent lane so R1 BRAKE doesn't fire) -> ratio 0.31 ->
    CONFIDENCE == 'high' (wide margin)."""
    objects = [
        mk_obj(
            id_="ped", cls="PEDESTRIAN", state="STANDING",
            x=0.0, y=2.5, raw_category="human.pedestrian.adult",
        ),
    ]
    n, decision = _run(objects)
    assert decision.action == ACTION_STOP
    assert n.confidence == "high"
    assert "wide margin" in n.confidence_reason


def test_determinism():
    """narrate() is a pure function -> identical strings on repeated calls."""
    objects = [
        mk_obj(id_="ped-1", cls="PEDESTRIAN", state="STANDING",
               x=4.0, y=2.0, raw_category="human.pedestrian.adult"),
        mk_obj(id_="van",   cls="VEHICLE",    state="MOVING", x=33.0, y=0.3),
        mk_obj(id_="adj",   cls="VEHICLE",    state="MOVING", x=5.0,  y=-3.5),
    ]
    n1, _ = _run(objects)
    n2, _ = _run(objects)
    assert n1.to_dict() == n2.to_dict()
    assert format_text(n1) == format_text(n2)
    assert format_markdown(n1) == format_markdown(n2)


def test_perception_buckets_left_right_behind():
    objects = [
        mk_obj(id_="lead", cls="VEHICLE", state="MOVING", x=20.0, y=0.0),
        mk_obj(id_="left", cls="VEHICLE", state="MOVING", x=10.0, y=+3.5),
        mk_obj(id_="right", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5),
        mk_obj(id_="behind", cls="VEHICLE", state="MOVING", x=-15.0, y=0.0),
    ]
    n, _ = _run(objects)
    labels = " ".join(n.perception)
    assert "In my lane" in labels
    assert "Left adjacent" in labels
    assert "Right adjacent" in labels
    assert "Behind" in labels


def test_format_text_layout_has_expected_headers():
    n, _ = _run([])
    text = format_text(n)
    for header in (
        "PERCEPTION (what I observe):",
        "REASONING (step by step):",
        "DECISION:",
        "REASON:",
        "ALTERNATIVES CONSIDERED:",
        "CONFIDENCE:",
    ):
        assert header in text


def test_reasoning_contains_safety_tag_for_stop():
    objects = [
        mk_obj(id_="ped-1", cls="PEDESTRIAN", state="STANDING",
               x=4.0, y=2.0, raw_category="human.pedestrian.adult"),
        mk_obj(id_="van",   cls="VEHICLE",    state="MOVING", x=33.0, y=0.3),
    ]
    n, _ = _run(objects)
    steps_text = "\n".join(n.reasoning)
    assert "[SAFETY]" in steps_text
    assert "[BEHAVIOR]" in steps_text  # FOLLOW
    assert "[FALLBACK]" in steps_text  # CRUISE


def test_alternatives_omit_winning_action():
    # Single pedestrian off-lane -> STOP wins. STOP should not appear in
    # alternatives.
    objects = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=4.0, raw_category="human.pedestrian.adult"),
    ]
    n, _ = _run(objects)
    for alt in n.alternatives:
        assert not alt.startswith("STOP -")


# =============================================================================
# Dual-audience (Driver HMI + Engineer trace) tests
# =============================================================================


def test_driver_hmi_cruise_when_empty():
    n, decision = _run([])
    assert decision.action == "CRUISE"
    assert n.driver_hmi_action == "Cruising"
    assert n.driver_hmi_sentence == "Cruising — the road ahead is clear"
    assert n.driver_hmi_highlight == ""


def test_driver_hmi_brake_template():
    # Car 6 m straight ahead in the ego lane -> R1 BRAKE.
    objs = [mk_obj(id_="car", cls="VEHICLE", state="MOVING", x=6.0, y=0.0)]
    n, decision = _run(objs)
    assert decision.action == "BRAKE"
    assert n.driver_hmi_action == "Braking"
    assert n.driver_hmi_sentence == "Braking — car 6 m ahead in your lane"
    assert n.driver_hmi_highlight == "car"


def test_driver_hmi_stop_template_with_pedestrian():
    # Pedestrian off-lane (|y|>1.75) close to ego -> STOP.
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=4.0, raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "STOP"
    assert n.driver_hmi_action == "Stopping"
    assert n.driver_hmi_sentence == "Stopping — pedestrian crossing 4 m ahead"
    assert n.driver_hmi_highlight == "pedestrian"


def test_driver_hmi_yield_template():
    # Pedestrian inside caution but outside stop radius, off-lane.
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=10.0, raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "YIELD"
    assert n.driver_hmi_action == "Easing off"
    assert "pedestrian" in n.driver_hmi_sentence
    assert n.driver_hmi_highlight == "pedestrian"


def test_driver_hmi_follow_template():
    # Moving van ahead in lane at 25 m -> FOLLOW.
    objs = [
        mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=25.0, y=0.3,
               raw_category="vehicle.car"),
    ]
    n, decision = _run(objs)
    assert decision.action == "FOLLOW"
    assert n.driver_hmi_action == "Following"
    assert n.driver_hmi_sentence == "Following the car 25 m ahead"
    assert n.driver_hmi_highlight == "car"


def test_driver_hmi_inhibit_lane_change_left_vs_right():
    objs_left = [
        mk_obj(id_="adj-l", cls="VEHICLE", state="MOVING", x=10.0, y=+3.5),
    ]
    n_l, decision_l = _run(objs_left)
    assert decision_l.action == "INHIBIT_LANE_CHANGE"
    assert n_l.driver_hmi_action == "Holding lane"
    assert "left adjacent lane" in n_l.driver_hmi_sentence

    objs_right = [
        mk_obj(id_="adj-r", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5),
    ]
    n_r, decision_r = _run(objs_right)
    assert decision_r.action == "INHIBIT_LANE_CHANGE"
    assert "right adjacent lane" in n_r.driver_hmi_sentence


def test_context_density_clear_road():
    n, _ = _run([])
    assert n.context_label == "Clear road"


def test_context_density_light_traffic():
    # 3 vehicles only, no pedestrians anywhere near.
    objs = [
        mk_obj(id_="v1", cls="VEHICLE", state="MOVING", x=30.0, y=0.0),
        mk_obj(id_="v2", cls="VEHICLE", state="MOVING", x=12.0, y=4.0),
        mk_obj(id_="v3", cls="VEHICLE", state="MOVING", x=15.0, y=-3.5),
    ]
    n, _ = _run(objs)
    assert n.context_label == "Light traffic"


def test_context_density_dense_urban_traffic_by_peds():
    # 3+ pedestrians inside caution radius triggers dense even with low count.
    objs = [
        mk_obj(id_="p1", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
               raw_category="human.pedestrian.adult"),
        mk_obj(id_="p2", cls="PEDESTRIAN", state="STANDING", x=0.0, y=6.0,
               raw_category="human.pedestrian.adult"),
        mk_obj(id_="p3", cls="PEDESTRIAN", state="STANDING", x=10.0, y=3.5,
               raw_category="human.pedestrian.adult"),
    ]
    n, _ = _run(objs)
    assert n.context_label == "Dense urban traffic"


def test_risk_label_high_on_brake():
    # In-lane vehicle close + closing fast -> BRAKE priority 100 -> HIGH risk.
    objs = [mk_obj(id_="car", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)]
    n, decision = _run(objs)
    assert decision.action == "BRAKE"
    assert n.risk_label == "HIGH"
    assert n.risk_score >= 0.75


def test_risk_label_low_on_cruise():
    n, _ = _run([])
    # priority 1 -> 0.6 * 0.01 + 0.4 * 0.0 = 0.006 -> rounds to 0.01 -> LOW
    assert n.risk_label == "LOW"
    assert n.risk_score < 0.40


def test_risk_score_formula_stop_borderline():
    # Pedestrian off-lane at distance 7.6 m vs stop radius 8.0 -> STOP wins,
    # barely triggering the rule (ratio 0.95 -> urgency 0.05).
    # priority_norm = 95/100 = 0.95
    # urgency      = 1 - 7.6/8.0 = 0.05
    # score        = 0.6 * 0.95 + 0.4 * 0.05 = 0.59 -> MEDIUM
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=7.6,
               raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "STOP"
    assert n.risk_score == 0.59
    assert n.risk_label == "MEDIUM"


def test_risk_score_formula_stop_deep_in_zone():
    # Pedestrian off-lane at 2.5 m vs stop radius 8.0 -> deep in the zone.
    # urgency = 1 - 2.5/8.0 = 0.6875
    # score   = 0.6 * 0.95 + 0.4 * 0.6875 = 0.845 -> HIGH
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=2.5,
               raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "STOP"
    assert n.risk_label == "HIGH"
    assert n.risk_score >= 0.75


def test_engineer_rows_order_and_labels():
    objs = [mk_obj(id_="car", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)]
    n, _ = _run(objs)
    labels = [r.label for r in n.engineer_rows]
    assert labels == ["CONTEXT", "RISK", "REASON", "ACTION"]
    assert all(isinstance(r, EngineerRow) for r in n.engineer_rows)


def test_engineer_rows_flag_when_confidence_low():
    # Borderline single PEDESTRIAN off-lane -> confidence "low" -> REASON flagged.
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=7.6,
               raw_category="human.pedestrian.adult"),
    ]
    n, _ = _run(objs)
    assert n.confidence == "low"
    reason_row = next(r for r in n.engineer_rows if r.label == "REASON")
    assert reason_row.status == "flag"
    other_rows = [r for r in n.engineer_rows if r.label != "REASON"]
    assert all(r.status == "ok" for r in other_rows)


def test_engineer_rows_no_flag_when_confidence_not_low():
    objs = [mk_obj(id_="car", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)]
    n, _ = _run(objs)
    assert n.confidence != "low"
    for r in n.engineer_rows:
        assert r.status == "ok"


def test_action_qualifier_per_action():
    cases = [
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)], "BRAKE", "Brake firmly"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
                 raw_category="human.pedestrian.adult")], "STOP", "Hold position"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=25.0, y=0.3)], "FOLLOW", "Match leader speed"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5)], "INHIBIT_LANE_CHANGE", "Hold lane"),
        ([], "CRUISE", "Cruise"),
    ]
    for objs, expected_action, expected_qualifier in cases:
        n, decision = _run(objs)
        assert decision.action == expected_action, f"expected {expected_action}, got {decision.action}"
        action_row = next(r for r in n.engineer_rows if r.label == "ACTION")
        assert action_row.value == expected_qualifier


def test_dual_audience_determinism():
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
               raw_category="human.pedestrian.adult"),
        mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=25.0, y=0.3),
    ]
    n1, _ = _run(objs)
    n2, _ = _run(objs)
    assert format_driver_hmi(n1) == format_driver_hmi(n2)
    assert [r.__dict__ for r in format_engineer_trace(n1)] == \
           [r.__dict__ for r in format_engineer_trace(n2)]
    assert n1.risk_score == n2.risk_score
    assert n1.context_label == n2.context_label


# =============================================================================
# Causal chain + driver sub-line tests
# =============================================================================


def test_causal_chain_four_steps_for_stop():
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=4.0, raw_category="human.pedestrian.adult"),
        mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=33.0, y=0.3),
    ]
    n, decision = _run(objs)
    assert decision.action == "STOP"
    chain = n.causal_chain
    assert len(chain) == 4
    assert [s.label for s in chain] == ["OBSERVE", "EVAL", "WEIGH", "DECIDE"]
    assert [s.step_num for s in chain] == [1, 2, 3, 4]
    # OBSERVE mentions the salient pedestrian and a distance.
    assert "pedestrian" in chain[0].text.lower()
    assert "m" in chain[0].text
    # WEIGH names STOP as winner AND at least one competing action.
    assert "STOP" in chain[2].text
    assert ("FOLLOW" in chain[2].text or "CRUISE" in chain[2].text)
    # DECIDE = action -> qualifier
    assert chain[3].text == "STOP → Hold position"


def test_causal_chain_cruise_short_form():
    n, decision = _run([])
    assert decision.action == "CRUISE"
    chain = n.causal_chain
    assert len(chain) == 4
    assert chain[0].label == "OBSERVE"
    assert "no hazards" in chain[0].text.lower()
    assert chain[1].label == "EVAL"
    assert "no applicable" in chain[1].text.lower()
    assert chain[2].label == "WEIGH"
    assert chain[2].text.startswith("Only CRUISE")
    assert chain[3].text == "CRUISE → Cruise"


def test_causal_chain_inhibit_no_numeric_threshold():
    # Single moving vehicle in the right adjacent lane → R4 INHIBIT_LANE_CHANGE.
    objs = [mk_obj(id_="adj", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5)]
    n, decision = _run(objs)
    assert decision.action == "INHIBIT_LANE_CHANGE"
    chain = n.causal_chain
    # OBSERVE mentions the adjacent-lane side.
    assert "right" in chain[0].text.lower() or "adjacent" in chain[0].text.lower()
    # EVAL uses the R4-specific text (no numeric threshold extraction).
    assert "adjacent lane" in chain[1].text.lower()
    assert "blocks" in chain[1].text.lower()
    # DECIDE qualifier
    assert chain[3].text == "INHIBIT_LANE_CHANGE → Hold lane"


def test_causal_chain_for_brake_quotes_gap_or_ttc():
    # In-lane vehicle 6 m ahead → R1 BRAKE via gap.
    objs = [mk_obj(id_="car", cls="VEHICLE", state="MOVING", x=6.0, y=0.0)]
    n, decision = _run(objs)
    assert decision.action == "BRAKE"
    chain = n.causal_chain
    eval_text = chain[1].text.lower()
    assert ("gap" in eval_text or "ttc" in eval_text)
    assert "emergency hazard" in eval_text


def test_causal_chain_for_follow_includes_distance_threshold():
    objs = [mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=25.0, y=0.3)]
    n, decision = _run(objs)
    assert decision.action == "FOLLOW"
    chain = n.causal_chain
    assert "follow distance" in chain[1].text.lower()
    assert "40.0 m" in chain[1].text
    assert chain[3].text == "FOLLOW → Match leader speed"


def test_causal_chain_for_yield_uses_caution_radius():
    # PED off-lane at distance ~10.7 m (inside 15 m caution, outside 8 m stop).
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=10.0,
               raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "YIELD"
    chain = n.causal_chain
    assert "caution radius" in chain[1].text.lower()
    assert chain[3].text == "YIELD → Ease speed"


def test_driver_hmi_subline_per_action():
    expected = {
        "BRAKE":               "Closing fast — pedal pressure increasing.",
        "STOP":                "They're moving into your path; safer to hold.",
        "YIELD":               "Easing off to give them room.",
        "FOLLOW":              "Matching their speed at a safe distance.",
        "INHIBIT_LANE_CHANGE": "Adjacent lane occupied — staying put.",
        "CRUISE":              "Road clear — maintaining cruise.",
    }
    scenarios = [
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)], "BRAKE"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
                 raw_category="human.pedestrian.adult")], "STOP"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=10.0,
                 raw_category="human.pedestrian.adult")], "YIELD"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=25.0, y=0.3)], "FOLLOW"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5)], "INHIBIT_LANE_CHANGE"),
        ([], "CRUISE"),
    ]
    for objs, expected_action in scenarios:
        n, decision = _run(objs)
        assert decision.action == expected_action, \
            f"setup mismatch: expected {expected_action}, got {decision.action}"
        assert n.driver_hmi_subline == expected[expected_action], \
            f"action={expected_action} got subline={n.driver_hmi_subline!r}"


def test_causal_chain_determinism():
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
               raw_category="human.pedestrian.adult"),
        mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=25.0, y=0.3),
        mk_obj(id_="adj", cls="VEHICLE", state="MOVING", x=5.0, y=-3.5),
    ]
    n1, _ = _run(objs)
    n2, _ = _run(objs)
    chain1 = format_causal_chain(n1)
    chain2 = format_causal_chain(n2)
    assert [(s.step_num, s.label, s.text) for s in chain1] == \
           [(s.step_num, s.label, s.text) for s in chain2]
    assert all(isinstance(s, CausalStep) for s in chain1)


def test_format_driver_hmi_returns_subline():
    objs = [mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)]
    n, _ = _run(objs)
    hmi = format_driver_hmi(n)
    assert "subline" in hmi
    assert hmi["subline"] == "Closing fast — pedal pressure increasing."


# =============================================================================
# PRISM Signature Output tests
# =============================================================================


def test_signature_schema_questions_complete():
    """The schema must define exactly these 4 questions, in this order."""
    assert list(SIGNATURE_QUESTIONS.keys()) == [
        "context", "risk", "action", "reason",
    ]
    assert SIGNATURE_QUESTIONS["context"] == "What is happening in Road"
    assert SIGNATURE_QUESTIONS["risk"] == "Level of Risk"
    assert SIGNATURE_QUESTIONS["action"] == "Action taken by ADAS"
    assert SIGNATURE_QUESTIONS["reason"] == "Reasoning behind the Action"


def test_signature_inhibit_lane_change_matches_mockup_shape():
    """Lane-change inhibit scenario. CONTEXT now expresses semantic
    interpretation of the driving environment (intent + conflict) rather
    than naming the detected object."""
    objs = [
        mk_obj(id_="truck", cls="LARGE_VEHICLE", state="MOVING",
               x=10.0, y=+3.5, raw_category="vehicle.truck"),
    ]
    n, decision = _run(objs)
    assert decision.action == "INHIBIT_LANE_CHANGE"
    sig = n.signature
    assert isinstance(sig, SignatureOutput)
    assert sig.context == "Lane-change opportunity blocked by adjacent traffic"
    assert sig.action == "Lane-Change Inhibit active"
    assert sig.risk in {"HIGH", "MEDIUM", "LOW"}
    # REASON keeps the driver-facing object reference + side.
    assert "left blind spot" in sig.reason.lower()
    assert "lane change isn't safe" in sig.reason.lower()


def test_signature_stop_for_pedestrian():
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=4.0, raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "STOP"
    sig = n.signature
    assert sig.context == "Vulnerable road user crossing the protected path"
    assert sig.action == "Auto-Hold engaged"
    # REASON quotes the stop-zone threshold from the winning finding.
    assert "8.0 m stop zone" in sig.reason
    assert "holding position" in sig.reason.lower()


def test_signature_brake_phrasing():
    objs = [mk_obj(id_="car", cls="VEHICLE", state="MOVING", x=6.0, y=0.0)]
    n, decision = _run(objs)
    assert decision.action == "BRAKE"
    sig = n.signature
    assert "Imminent forward collision developing" in sig.context
    assert sig.action == "Autonomous Emergency Braking engaged"
    assert sig.risk in {"HIGH", "MEDIUM"}
    # REASON quotes the brake distance.
    assert "12.0 m brake distance" in sig.reason
    assert "pressing the brakes" in sig.reason.lower()


def test_signature_follow_phrasing():
    objs = [mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=25.0, y=0.3)]
    n, decision = _run(objs)
    assert decision.action == "FOLLOW"
    sig = n.signature
    assert "Sustained car-following" in sig.context
    assert sig.action == "Adaptive Cruise engaged"
    # REASON quotes follow distance and "matching their speed".
    assert "40.0 m follow distance" in sig.reason
    assert "matching their speed" in sig.reason.lower()


def test_signature_yield_phrasing():
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=10.0, raw_category="human.pedestrian.adult"),
    ]
    n, decision = _run(objs)
    assert decision.action == "YIELD"
    sig = n.signature
    assert sig.context == "Pedestrian activity within caution proximity"
    assert sig.action == "Throttle eased · monitoring"
    # REASON quotes the caution radius.
    assert "15.0 m caution radius" in sig.reason
    assert "easing off" in sig.reason.lower()


def test_signature_cruise_phrasing():
    n, decision = _run([])
    assert decision.action == "CRUISE"
    sig = n.signature
    assert sig.context == "Open road operation, no behavioral threats"
    assert sig.action == "Cruise Control active"
    assert sig.risk == "LOW"
    assert "holding cruise speed" in sig.reason.lower()


def test_signature_inhibit_side_lives_in_reason_not_context():
    """CONTEXT is the semantic environment interpretation (same for either
    side); the specific blind-spot side is a driver-facing detail that
    belongs in REASON."""
    left = [mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=10.0, y=+3.5)]
    right = [mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5)]
    n_l, _ = _run(left)
    n_r, _ = _run(right)
    assert n_l.signature.context == n_r.signature.context  # same semantic
    assert n_l.signature.context == "Lane-change opportunity blocked by adjacent traffic"
    assert "left blind spot" in n_l.signature.reason.lower()
    assert "right blind spot" in n_r.signature.reason.lower()


def test_signature_context_is_semantic_environment_interpretation():
    """CONTEXT must convey a semantic interpretation of the driving
    environment — not name the detected object with location/distance.
    Object specifics belong in REASON."""
    import re
    scenarios = [
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)],
         "BRAKE", "Imminent forward collision"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
                 raw_category="human.pedestrian.adult")],
         "STOP", "Vulnerable road user"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=10.0,
                 raw_category="human.pedestrian.adult")],
         "YIELD", "Pedestrian activity"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=25.0, y=0.3)],
         "FOLLOW", "Sustained car-following"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5)],
         "INHIBIT_LANE_CHANGE", "Lane-change opportunity blocked"),
        ([], "CRUISE", "Open road operation"),
    ]
    for objs, expected_action, expected_semantic in scenarios:
        n, decision = _run(objs)
        assert decision.action == expected_action
        ctx = n.signature.context
        # CONTEXT must contain the expected semantic phrase
        assert expected_semantic in ctx, \
            f"CONTEXT for {expected_action}: expected {expected_semantic!r} in {ctx!r}"
        # CONTEXT must not contain a numeric distance — those belong in REASON.
        assert not re.search(r"\d+\.?\d*\s*m\b", ctx), \
            f"CONTEXT for {expected_action} contains a distance: {ctx!r}"
        # CONTEXT must not use object-locative phrasings.
        for forbidden in (" in lane", "ahead in your", "blind spot", " in left",
                          " in right", "crossing ahead", "lead car ahead"):
            assert forbidden not in ctx.lower(), \
                f"CONTEXT for {expected_action} is object-locative ({forbidden!r}): {ctx!r}"


def test_signature_action_is_active_vehicle_function():
    """ACTION must name an *active* vehicle function (system-engaged wording),
    not a recommendation phrased to the driver."""
    scenarios = [
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)],
         "BRAKE",  "Autonomous Emergency Braking engaged"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
                 raw_category="human.pedestrian.adult")],
         "STOP",   "Auto-Hold engaged"),
        ([mk_obj(id_="p", cls="PEDESTRIAN", state="STANDING", x=0.0, y=10.0,
                 raw_category="human.pedestrian.adult")],
         "YIELD",  "Throttle eased · monitoring"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=25.0, y=0.3)],
         "FOLLOW", "Adaptive Cruise engaged"),
        ([mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=10.0, y=-3.5)],
         "INHIBIT_LANE_CHANGE", "Lane-Change Inhibit active"),
        ([], "CRUISE", "Cruise Control active"),
    ]
    for objs, expected_action, expected_value in scenarios:
        n, decision = _run(objs)
        assert decision.action == expected_action
        assert n.signature.action == expected_value
        # Guard against recommendation phrasing creeping back in.
        lower = n.signature.action.lower()
        for forbidden in ("apply ", "hold position", "reduce speed",
                          "match leader", "maintain lane", "maintain cruise"):
            assert forbidden not in lower, \
                f"ACTION for {expected_action} reads as recommendation: {n.signature.action!r}"


def test_signature_reason_is_conversational():
    """REASON should read as a driver-facing sentence (not engineer jargon)."""
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING",
               x=0.0, y=4.0, raw_category="human.pedestrian.adult"),
    ]
    n, _ = _run(objs)
    reason = n.signature.reason
    # Should not contain rule IDs or engineer prefixes
    assert "R2_" not in reason
    assert "STOP:" not in reason
    # Should contain conversational pronouns ("we're", "your", "you")
    assert ("we're" in reason.lower() or "we'll" in reason.lower())
    assert "your" in reason.lower() or "you" in reason.lower()


def test_signature_determinism():
    objs = [
        mk_obj(id_="ped", cls="PEDESTRIAN", state="STANDING", x=0.0, y=4.0,
               raw_category="human.pedestrian.adult"),
        mk_obj(id_="van", cls="VEHICLE", state="MOVING", x=25.0, y=0.3),
    ]
    n1, _ = _run(objs)
    n2, _ = _run(objs)
    s1, s2 = n1.signature, n2.signature
    assert (s1.context, s1.risk, s1.action, s1.reason) == \
           (s2.context, s2.risk, s2.action, s2.reason)


def test_format_signature_output_accessor():
    objs = [mk_obj(id_="v", cls="VEHICLE", state="MOVING", x=4.0, y=0.0)]
    n, _ = _run(objs)
    sig = format_signature_output(n)
    assert isinstance(sig, SignatureOutput)
    assert sig.action == "Autonomous Emergency Braking engaged"
    assert sig.context.startswith("Imminent forward collision developing")


def test_friendly_noun_fallbacks():
    # bus.bendy / emergency.police / unknown / pedestrian.* should all resolve.
    cases = [
        ("vehicle.bus.bendy", "bus"),
        ("vehicle.bus.rigid", "bus"),
        ("vehicle.emergency.police", "emergency vehicle"),
        ("human.pedestrian.child", "pedestrian"),
        ("animal", "obstacle"),
        ("movable_object.barrier", "obstacle"),
    ]
    for raw, expected in cases:
        objs = [mk_obj(id_="o", cls="VEHICLE" if "vehicle" in raw else "PEDESTRIAN" if "pedestrian" in raw else "STATIC",
                       state="MOVING" if "vehicle" in raw else "STANDING",
                       x=4.0, y=0.0, raw_category=raw)]
        n, _ = _run(objs)
        # Driver HMI sentence should contain the friendly noun.
        if n.driver_hmi_highlight:
            assert n.driver_hmi_highlight == expected, f"raw={raw!r} expected {expected!r}, got {n.driver_hmi_highlight!r}"
