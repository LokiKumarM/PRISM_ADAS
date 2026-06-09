"""Narration layer: render a Decision as a human-readable chain of thought.

This module is **pure presentation**. It does not run rules, change
thresholds, or alter the arbiter's chosen action. It consumes:

  * a finished ``Decision`` (with its full ``supporting_findings`` trace),
  * the per-frame ``ObjectList``,
  * a mapping ``object_id -> Relations`` built once by the pipeline,
  * the rules ``cfg`` dict (only used for read-only thresholds).

It produces a ``Narration`` dataclass and two formatters (``format_text`` for
CLI/logs and ``format_markdown`` for the Streamlit panel). Same input ->
identical output, every time. No randomness, no LLM, no network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from src.object_list import DetectedObject
from src.reasoning.arbiter import Decision
from src.reasoning.relations import (
    ADJACENT_LANE,
    EGO_LANE,
    Relations,
)
from src.reasoning.rules import ACTION_CRUISE, Finding

# Tag thresholds (used for narration only — they are NOT new rule thresholds).
SAFETY_PRIORITY_FLOOR = 90
FALLBACK_PRIORITY = 1

# Confidence heuristic ratios (value / threshold).
BORDERLINE_RATIO = 0.90   # ratio >= this -> "low" (within 10% of threshold)
WIDE_MARGIN_RATIO = 0.50  # ratio <= this -> "high" (clears by >= 50%)


@dataclass
class EngineerRow:
    """One row of the Engineer / OEM trace card."""
    label: str   # "CONTEXT" | "RISK" | "REASON" | "ACTION"
    value: str
    status: str  # "ok" | "flag"


@dataclass
class CausalStep:
    """One step in the four-step causal chain shown under the Engineer card."""
    step_num: int   # 1..N
    label: str      # "OBSERVE" | "EVAL" | "WEIGH" | "DECIDE"
    text: str       # one-line clause


@dataclass
class SignatureOutput:
    """Reasoned Alert from the PRISM layer.

    The schema is invariant across actions:
      * CONTEXT / RISK / ACTION are short, specific values describing the
        road scene and the ADAS response.
      * REASON is a plain-language sentence written to the driver that
        explains *why* this action was taken.
    """
    context: str   # "What is happening in Road"    — specific noun phrase
    risk:    str   # "Level of Risk"                — HIGH | MEDIUM | LOW
    action:  str   # "Action taken by ADAS"         — specific imperative
    reason:  str   # "Reasoning behind the Action"  — driver-facing prose


@dataclass
class Narration:
    """Structured narration of a Decision.

    The original CLI-facing fields (perception / reasoning / decision / because
    / alternatives / confidence / confidence_reason) are preserved so the
    ``python -m src.pipeline`` CLI keeps working unchanged.

    The new dual-audience fields are populated alongside them for the
    Streamlit UI (Driver HMI card + Engineer / OEM Trace card).
    """
    perception: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    decision: str = ""
    because: str = ""
    alternatives: list[str] = field(default_factory=list)
    confidence: str = ""
    confidence_reason: str = ""

    # ----- dual-audience fields -----
    driver_hmi_action: str = ""        # e.g. "Stopping"
    driver_hmi_sentence: str = ""      # full plain-language sentence
    driver_hmi_highlight: str = ""     # substring to visually highlight ("" if none)
    driver_hmi_subline: str = ""       # contextual secondary sentence
    context_label: str = ""            # density heuristic
    risk_score: float = 0.0            # 0..1
    risk_label: str = ""               # "HIGH" | "MEDIUM" | "LOW"
    engineer_rows: list[EngineerRow] = field(default_factory=list)
    causal_chain: list[CausalStep] = field(default_factory=list)
    signature: Optional[SignatureOutput] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "perception": list(self.perception),
            "reasoning": list(self.reasoning),
            "decision": self.decision,
            "because": self.because,
            "alternatives": list(self.alternatives),
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "driver_hmi_action": self.driver_hmi_action,
            "driver_hmi_sentence": self.driver_hmi_sentence,
            "driver_hmi_highlight": self.driver_hmi_highlight,
            "driver_hmi_subline": self.driver_hmi_subline,
            "context_label": self.context_label,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "engineer_rows": [
                {"label": r.label, "value": r.value, "status": r.status}
                for r in self.engineer_rows
            ],
            "causal_chain": [
                {"step_num": s.step_num, "label": s.label, "text": s.text}
                for s in self.causal_chain
            ],
            "signature": (
                {
                    "context": self.signature.context,
                    "risk": self.signature.risk,
                    "action": self.signature.action,
                    "reason": self.signature.reason,
                }
                if self.signature is not None
                else None
            ),
        }


# =============================================================================
# Public entry point
# =============================================================================


def narrate(
    decision: Decision,
    objects: list[DetectedObject],
    facts: Mapping[str, Relations],
    cfg: Optional[dict[str, Any]] = None,
) -> Narration:
    """Build a Narration from an already-decided frame.

    Parameters
    ----------
    decision : Decision
        The arbiter's output, including the full sorted findings trace.
    objects : list[DetectedObject]
        The ObjectList that was fed to the rules.
    facts : Mapping[str, Relations]
        ``object_id -> Relations`` (precomputed by the pipeline).
    cfg : dict, optional
        The rules config. Only used for read-only thresholds (e.g. pedestrian
        caution radius for grouping). Defaults to ``{}``.
    """
    cfg = cfg or {}
    n = Narration()
    n.perception = _build_perception(objects, facts, cfg)
    n.reasoning = _build_reasoning(decision)
    n.decision = decision.action
    n.because = decision.primary_reason
    n.alternatives = _build_alternatives(decision)
    n.confidence, n.confidence_reason = _build_confidence(decision)

    # Dual-audience fields (Driver HMI + Engineer / OEM trace).
    obj_by_id = {o.id: o for o in objects}
    salient = obj_by_id.get(_winner_object_id(decision))
    n.driver_hmi_action, n.driver_hmi_sentence, n.driver_hmi_highlight = (
        _build_driver_hmi(decision, salient)
    )
    n.driver_hmi_subline = _build_driver_hmi_subline(decision.action)
    n.context_label = _density_label(objects, facts, cfg)
    n.risk_score, n.risk_label = _build_risk(decision)
    n.engineer_rows = _build_engineer_rows(
        decision=decision,
        context_label=n.context_label,
        risk_label=n.risk_label,
        risk_score=n.risk_score,
        confidence=n.confidence,
    )
    n.causal_chain = _build_causal_chain(decision, salient)
    n.signature = _build_signature_output(
        decision=decision,
        salient=salient,
        risk_label=n.risk_label,
        context_label=n.context_label,
    )
    return n


# =============================================================================
# PERCEPTION
# =============================================================================


_BUCKET_ORDER = (
    "In my lane",
    "Crossing/near",
    "Left adjacent",
    "Right adjacent",
    "Behind",
)


def _bucket_objects(
    objects: list[DetectedObject],
    facts: Mapping[str, Relations],
    cfg: dict[str, Any],
) -> dict[str, list[DetectedObject]]:
    buckets: dict[str, list[DetectedObject]] = {b: [] for b in _BUCKET_ORDER}
    ped_caution = float(cfg.get("ped_caution_radius_m", 15.0))
    for obj in objects:
        rel = facts.get(obj.id)
        # Pedestrians inside the caution radius are surfaced first regardless
        # of lane, because that's the most safety-relevant grouping.
        if obj.cls == "PEDESTRIAN" and obj.distance < ped_caution:
            buckets["Crossing/near"].append(obj)
            continue
        if rel is None:
            continue  # we cannot place it without spatial facts
        if rel.gap < 0:
            buckets["Behind"].append(obj)
            continue
        if rel.zone == EGO_LANE:
            buckets["In my lane"].append(obj)
        elif rel.zone == ADJACENT_LANE:
            if obj.y > 0:
                buckets["Left adjacent"].append(obj)
            else:
                buckets["Right adjacent"].append(obj)
        # OFF_ROAD ahead is omitted -- it doesn't influence reasoning here.
    return buckets


def _summarize_classes(objs: list[DetectedObject]) -> str:
    classes = sorted({o.cls.lower() for o in objs})
    states = sorted({o.state for o in objs})
    if len(classes) == 1:
        word = classes[0] + ("s" if len(objs) > 1 else "")
        if len(states) == 1:
            return f"{word} ({states[0].lower()})"
        return word
    return "/".join(classes)


def _describe_bucket(label: str, objs: list[DetectedObject]) -> str:
    n = len(objs)
    if label == "Crossing/near":
        objs_sorted = sorted(objs, key=lambda o: o.distance)
        dists = ", ".join(f"{o.distance:.1f} m" for o in objs_sorted)
        states = sorted({o.state for o in objs_sorted})
        plural = "pedestrians" if n > 1 else "pedestrian"
        return f"Crossing/near:  {n} {plural} at {dists} ({'/'.join(states)})"
    if label == "In my lane":
        nearest = min(objs, key=lambda o: o.x)
        return f"In my lane:     {n} {_summarize_classes(objs)}, nearest {nearest.x:.1f} m ahead"
    if label == "Left adjacent":
        nearest = min(objs, key=lambda o: o.distance)
        return f"Left adjacent:  {n} {_summarize_classes(objs)}, nearest at x={nearest.x:+.1f} m"
    if label == "Right adjacent":
        nearest = min(objs, key=lambda o: o.distance)
        return f"Right adjacent: {n} {_summarize_classes(objs)}, nearest at x={nearest.x:+.1f} m"
    if label == "Behind":
        nearest = min(objs, key=lambda o: -o.x)
        return f"Behind:         {n} {_summarize_classes(objs)}, nearest {abs(nearest.x):.1f} m behind"
    return f"{label}: {n}"


def _build_perception(
    objects: list[DetectedObject],
    facts: Mapping[str, Relations],
    cfg: dict[str, Any],
) -> list[str]:
    if not objects:
        return ["Road ahead is clear (no relevant objects)."]
    buckets = _bucket_objects(objects, facts, cfg)
    lines = [
        _describe_bucket(label, buckets[label])
        for label in _BUCKET_ORDER
        if buckets[label]
    ]
    if not lines:
        return ["Road ahead is clear (no relevant objects)."]
    return lines


# =============================================================================
# REASONING
# =============================================================================


def _tag(priority: int) -> str:
    if priority >= SAFETY_PRIORITY_FLOOR:
        return "SAFETY"
    if priority == FALLBACK_PRIORITY:
        return "FALLBACK"
    return "BEHAVIOR"


def _inference_for(f: Finding, has_safety: bool) -> str:
    """Deterministic clause for *why* this finding matters."""
    rid = f.rule
    if rid.startswith("R1_BRAKE") or rid == "R2_PED_STOP":
        return "this is a safety-critical hazard and dominates comfort behaviors"
    if rid == "R2_PED_YIELD":
        return "a pedestrian in the caution zone calls for slowing/yielding"
    if rid.startswith("R3"):
        if has_safety:
            return "this is a comfort behavior, overridden by the safety hazard above"
        return "comfort behavior: follow the lead vehicle"
    if rid.startswith("R4"):
        if has_safety:
            return "the adjacent lane is blocked (and a safety hazard already dominates)"
        return "the adjacent lane is blocked, so a lane change is not currently available"
    if rid.startswith("R5"):
        return "no hazards or follow targets in the ego lane"
    return "noted as a relevant factor"


def _action_priority_summary(findings: list[Finding]) -> list[tuple[str, int]]:
    """For each distinct action, the highest priority that proposed it."""
    by_action: dict[str, int] = {}
    for f in findings:
        if f.action not in by_action or f.priority > by_action[f.action]:
            by_action[f.action] = f.priority
    # Stable sort: highest priority first, then alphabetical for ties.
    return sorted(by_action.items(), key=lambda kv: (-kv[1], kv[0]))


def _build_reasoning(decision: Decision) -> list[str]:
    findings = list(decision.supporting_findings)  # already priority-desc

    # Only CRUISE fired -> single short narrative.
    non_cruise = [f for f in findings if not f.rule.startswith("R5")]
    if not non_cruise:
        return ["No hazards or follow targets are present, so I cruise."]

    has_safety = any(f.priority >= SAFETY_PRIORITY_FLOOR for f in findings)
    steps: list[str] = []
    for i, f in enumerate(findings, start=1):
        steps.append(
            f"{i}. [{_tag(f.priority)}] {f.reason} -> {_inference_for(f, has_safety)}"
        )

    # Final WEIGH step.
    action_priorities = _action_priority_summary(findings)
    win_action, win_priority = action_priorities[0]
    if len(action_priorities) == 1:
        weigh = (
            f"Weighing these: only {win_action} (priority {win_priority}) applies, "
            f"so {win_action} is selected."
        )
    else:
        others = ", ".join(f"{a} (priority {p})" for a, p in action_priorities[1:])
        weigh = (
            f"Weighing these: {win_action} (priority {win_priority}) outranks "
            f"{others}, so {win_action} is selected."
        )
    steps.append(f"{len(findings) + 1}. {weigh}")
    return steps


# =============================================================================
# ALTERNATIVES
# =============================================================================


def _trim_action_prefix(action: str, reason: str) -> str:
    prefix = f"{action}: "
    if reason.startswith(prefix):
        return reason[len(prefix):]
    return reason


def _build_alternatives(decision: Decision) -> list[str]:
    winner = decision.action
    # Dedup by action: keep the highest-priority finding for each non-winning action.
    by_action: dict[str, Finding] = {}
    for f in decision.supporting_findings:
        if f.action == winner:
            continue
        cur = by_action.get(f.action)
        if cur is None or f.priority > cur.priority:
            by_action[f.action] = f
    if not by_action:
        return ["No competing actions; decision was unambiguous."]
    ordered = sorted(by_action.values(), key=lambda x: (-x.priority, x.action))
    return [
        f"{f.action} - {_trim_action_prefix(f.action, f.reason)}  "
        f"(overridden: priority {f.priority} < {decision.priority})"
        for f in ordered
    ]


# =============================================================================
# CONFIDENCE
# =============================================================================


# (value m < threshold m) or (value s < threshold s)
_RE_LT = re.compile(r"(\d+(?:\.\d+)?)\s*(m|s)\s*<\s*(\d+(?:\.\d+)?)\s*\2")
# "at X m (< ... Y m)" — used by R2/R3 reasons
_RE_AT_PAREN = re.compile(r"at\s+(\d+(?:\.\d+)?)\s*m\b[^()]*\(<[^)]*?(\d+(?:\.\d+)?)\s*m\)")


def _extract_margin(reason: str) -> Optional[tuple[float, float]]:
    """Return ``(value, threshold)`` for the *tightest* threshold trigger in
    a finding's reason string (largest ``value / threshold`` ratio). Returns
    ``None`` if the reason quotes no numeric threshold (e.g. R4, R5)."""
    pairs: list[tuple[float, float]] = []
    for m in _RE_LT.finditer(reason):
        v, thr = float(m.group(1)), float(m.group(3))
        if thr > 0:
            pairs.append((v, thr))
    for m in _RE_AT_PAREN.finditer(reason):
        v, thr = float(m.group(1)), float(m.group(2))
        if thr > 0:
            pairs.append((v, thr))
    if not pairs:
        return None
    return max(pairs, key=lambda p: p[0] / p[1])


def _build_confidence(decision: Decision) -> tuple[str, str]:
    winner = decision.action
    supporting = [f for f in decision.supporting_findings if f.action == winner]
    n = len(supporting)

    if n >= 2:
        return ("high", f"{n} independent findings agree on {winner}")

    margin = _extract_margin(supporting[0].reason) if supporting else None
    if margin is None:
        return ("medium", "single trigger; no measurement margin to compare")

    v, thr = margin
    ratio = v / thr if thr else 1.0
    if ratio >= BORDERLINE_RATIO:
        pct = (1.0 - ratio) * 100
        return (
            "low",
            f"trigger value {v:.1f} is within {pct:.0f}% of the {thr:.1f} threshold (borderline)",
        )
    if ratio <= WIDE_MARGIN_RATIO:
        return (
            "high",
            f"trigger value {v:.1f} clears the {thr:.1f} threshold by a wide margin",
        )
    return (
        "medium",
        f"trigger value {v:.1f} vs threshold {thr:.1f} (moderate margin)",
    )


# =============================================================================
# Formatters
# =============================================================================


def format_text(n: Narration) -> str:
    """Plain-text block suitable for CLI / log output."""
    out: list[str] = []
    out.append("PERCEPTION (what I observe):")
    for line in n.perception:
        out.append(f"  {line}")
    out.append("")
    out.append("REASONING (step by step):")
    for step in n.reasoning:
        out.append(f"  {step}")
    out.append("")
    out.append(f"DECISION:   {n.decision}")
    out.append(f"REASON:    {n.because}")
    out.append("")
    out.append("ALTERNATIVES CONSIDERED:")
    for alt in n.alternatives:
        out.append(f"  {alt}")
    out.append("")
    out.append(f"CONFIDENCE: {n.confidence} ({n.confidence_reason})")
    return "\n".join(out)


def format_markdown(n: Narration) -> str:
    """Compact markdown for the Streamlit right-hand panel."""
    out: list[str] = []
    out.append("**Perception (what I observe):**")
    for line in n.perception:
        out.append(f"- {line}")
    out.append("")
    out.append("**Reasoning (step by step):**")
    for step in n.reasoning:
        out.append(f"- {step}")
    out.append("")
    out.append(f"**Decision:** `{n.decision}`")
    out.append(f"**Reason:** {n.because}")
    out.append("")
    out.append("**Alternatives considered:**")
    for alt in n.alternatives:
        out.append(f"- {alt}")
    out.append("")
    out.append(f"**Confidence:** _{n.confidence}_ — {n.confidence_reason}")
    return "\n".join(out)


# =============================================================================
# Dual-audience layer — Driver HMI + Engineer / OEM trace
# =============================================================================

# Density heuristic cutoffs (presentation, not rule logic).
_DENSE_OBJ_COUNT = 30
_LIGHT_OBJ_COUNT = 10
_DENSE_PED_NEAR = 3

# Risk score weights & label cuts.
_RISK_WEIGHT_PRIORITY = 0.6
_RISK_WEIGHT_TRIGGER = 0.4
_RISK_HIGH_CUT = 0.75
_RISK_MEDIUM_CUT = 0.40

# Mapping of nuScenes raw category → driver-friendly noun used in the HMI
# sentence (and as the highlight token).
_FRIENDLY_NOUN = {
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicyclist",
    "vehicle.truck": "truck",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.emergency.ambulance": "emergency vehicle",
    "vehicle.emergency.police": "emergency vehicle",
}

# Action -> short imperative qualifier shown on the ACTION row.
_ACTION_QUALIFIER = {
    "BRAKE": "Brake firmly",
    "STOP": "Hold position",
    "YIELD": "Ease speed",
    "FOLLOW": "Match leader speed",
    "INHIBIT_LANE_CHANGE": "Hold lane",
    "CRUISE": "Cruise",
}


def _friendly_noun(raw_category: str) -> str:
    """Map an nuScenes category string to a short driver-facing noun."""
    if raw_category in _FRIENDLY_NOUN:
        return _FRIENDLY_NOUN[raw_category]
    if raw_category.startswith("human.pedestrian"):
        return "pedestrian"
    if raw_category.startswith("vehicle.bus"):
        return "bus"
    if raw_category.startswith("vehicle.emergency"):
        return "emergency vehicle"
    if raw_category.startswith("vehicle."):
        return "vehicle"
    return "obstacle"


def _winner_object_id(decision: Decision) -> str:
    """Return the object_id of the highest-priority finding for the winning
    action; empty string if none (e.g. R5 CRUISE)."""
    for f in decision.supporting_findings:
        if f.action == decision.action and f.object_id:
            return f.object_id
    return ""


def _build_driver_hmi(
    decision: Decision, salient: Optional[DetectedObject]
) -> tuple[str, str, str]:
    """Return (action_word, sentence, highlight)."""
    action = decision.action
    if action == "CRUISE" or salient is None:
        return (
            "Cruising",
            "Cruising — the road ahead is clear",
            "",
        )
    noun = _friendly_noun(salient.raw_category)
    distance = max(0, int(round(salient.distance)))
    if action == "BRAKE":
        return (
            "Braking",
            f"Braking — {noun} {distance} m ahead in your lane",
            noun,
        )
    if action == "STOP":
        return (
            "Stopping",
            f"Stopping — {noun} crossing {distance} m ahead",
            noun,
        )
    if action == "YIELD":
        return (
            "Easing off",
            f"Easing off — {noun} {distance} m near your path",
            noun,
        )
    if action == "FOLLOW":
        return (
            "Following",
            f"Following the {noun} {distance} m ahead",
            noun,
        )
    if action == "INHIBIT_LANE_CHANGE":
        side = "left" if salient.y > 0 else "right"
        return (
            "Holding lane",
            f"Holding lane — {noun} in the {side} adjacent lane",
            noun,
        )
    # Unknown action: fall back to a generic line so the UI never goes blank.
    return (action.title(), f"{action.title()} — {noun} {distance} m away", noun)


def _density_label(
    objects: list[DetectedObject],
    facts: Mapping[str, Relations],
    cfg: dict[str, Any],
) -> str:
    """Bucket the frame into Clear road / Light / Moderate / Dense traffic."""
    if not objects:
        return "Clear road"
    buckets = _bucket_objects(objects, facts, cfg)
    # Relevant = anything we'd surface in PERCEPTION (i.e. not OFF_ROAD ahead).
    relevant = sum(len(v) for v in buckets.values())
    if relevant == 0:
        return "Clear road"
    n_peds_near = len(buckets["Crossing/near"])
    if len(objects) >= _DENSE_OBJ_COUNT or n_peds_near >= _DENSE_PED_NEAR:
        return "Dense urban traffic"
    if relevant <= _LIGHT_OBJ_COUNT and n_peds_near == 0:
        return "Light traffic"
    return "Moderate traffic"


def _build_risk(decision: Decision) -> tuple[float, str]:
    """Compute the deterministic hybrid risk score and its label.

    Reuses ``_extract_margin`` so the numbers come from the same strings
    already quoted in ``Finding.reason`` (nothing is recomputed).

    Intuition: for hazard rules, a *smaller* value/threshold ratio means we
    are deeper into the trigger zone (e.g. a pedestrian at 2 m of an 8 m stop
    radius is more urgent than one at 7.6 m). So the trigger urgency term is
    ``(1 - value/threshold)``. When the winning finding has no measurable
    trigger (R4 INHIBIT_LANE_CHANGE, R5 CRUISE), the urgency contribution
    is treated as 0 (no signal).
    """
    priority_norm = min(1.0, max(0.0, decision.priority / 100.0))
    urgency = 0.0  # 0 if no measurable trigger exists
    for f in decision.supporting_findings:
        if f.action == decision.action:
            margin = _extract_margin(f.reason)
            if margin is not None:
                v, thr = margin
                ratio = min(1.0, max(0.0, v / thr if thr else 0.0))
                urgency = 1.0 - ratio
            break
    score = _RISK_WEIGHT_PRIORITY * priority_norm + _RISK_WEIGHT_TRIGGER * urgency
    score = round(score, 2)
    if score >= _RISK_HIGH_CUT:
        label = "HIGH"
    elif score >= _RISK_MEDIUM_CUT:
        label = "MEDIUM"
    else:
        label = "LOW"
    return score, label


def _build_engineer_rows(
    *,
    decision: Decision,
    context_label: str,
    risk_label: str,
    risk_score: float,
    confidence: str,
) -> list[EngineerRow]:
    """Fixed-order 4-row engineer/OEM trace."""
    flag = "flag" if confidence == "low" else "ok"
    qualifier = _ACTION_QUALIFIER.get(decision.action, decision.action.title())
    return [
        EngineerRow("CONTEXT", context_label, "ok"),
        EngineerRow("RISK", f"{risk_label} · {risk_score:.2f}", "ok"),
        EngineerRow("REASON", decision.primary_reason, flag),
        EngineerRow("ACTION", qualifier, "ok"),
    ]


# ----- public accessors used by the Streamlit app -----


def format_driver_hmi(n: Narration) -> dict[str, str]:
    """Return ``{action, sentence, highlight, subline}`` for the HMI card."""
    return {
        "action": n.driver_hmi_action,
        "sentence": n.driver_hmi_sentence,
        "highlight": n.driver_hmi_highlight,
        "subline": n.driver_hmi_subline,
    }


def format_engineer_trace(n: Narration) -> list[EngineerRow]:
    """Return the engineer/OEM trace rows in display order."""
    return list(n.engineer_rows)


def format_causal_chain(n: Narration) -> list[CausalStep]:
    """Return the four-step causal chain in display order."""
    return list(n.causal_chain)


# =============================================================================
# Causal chain + driver sub-line
# =============================================================================

# Per-action contextual sentence shown under the Driver HMI action verb.
# Deterministic; no per-object templating to keep rhythm consistent.
_DRIVER_SUBLINE = {
    "BRAKE":               "Closing fast — pedal pressure increasing.",
    "STOP":                "They're moving into your path; safer to hold.",
    "YIELD":               "Easing off to give them room.",
    "FOLLOW":              "Matching their speed at a safe distance.",
    "INHIBIT_LANE_CHANGE": "Adjacent lane occupied — staying put.",
    "CRUISE":              "Road clear — maintaining cruise.",
}


def _build_driver_hmi_subline(action: str) -> str:
    return _DRIVER_SUBLINE.get(action, "")


def _winning_finding(decision: Decision) -> Optional[Finding]:
    """Highest-priority finding for the winning action (None if not present)."""
    for f in decision.supporting_findings:
        if f.action == decision.action:
            return f
    return None


def _observe_text(decision: Decision, salient: Optional[DetectedObject]) -> str:
    """One-line description of the salient object's facts."""
    if salient is None or decision.action == "CRUISE":
        return "No hazards or follow targets in the ego lane"
    noun = _friendly_noun(salient.raw_category)
    distance_m = round(salient.distance, 1)
    state = salient.state.lower()
    # Side qualifier helps INHIBIT_LANE_CHANGE which doesn't quote a distance.
    if decision.action == "INHIBIT_LANE_CHANGE":
        side = "left" if salient.y > 0 else "right"
        return f"{noun.capitalize()} {state} in the {side} adjacent lane"
    # Add closing speed only when we have it (skipped on first frame).
    closing_clause = ""
    if salient.vx is not None and -salient.vx > 0.1:
        closing_clause = f" closing at {(-salient.vx):.1f} m/s"
    return f"{noun.capitalize()} {state} at {distance_m} m{closing_clause}"


def _eval_text(decision: Decision) -> str:
    """One-line description of why the winning rule fired."""
    f = _winning_finding(decision)
    if f is None or decision.action == "CRUISE":
        return "No applicable hazard or follow rule fires"
    rid = f.rule
    if rid.startswith("R4"):
        return "Moving object in the adjacent lane blocks a lane change"
    margin = _extract_margin(f.reason)
    if margin is None:
        return "Rule precondition satisfied"
    v, thr = margin
    # Special-case wording per rule family so it reads naturally.
    if rid == "R2_PED_STOP":
        return f"Inside the {thr:.1f} m stop radius — classified as a safety hazard"
    if rid == "R2_PED_YIELD":
        return f"Inside the {thr:.1f} m caution radius — calls for slowing/yielding"
    if rid == "R3_FOLLOW":
        return f"Within the {thr:.1f} m follow distance — track this lead vehicle"
    if rid.startswith("R1"):
        # R1 reason may carry BOTH gap and ttc triggers; surface whichever fired.
        which = "TTC" if "ttc" in f.reason and "<" in f.reason and "s <" in f.reason else "gap"
        if which == "TTC":
            return f"TTC {v:.1f} s under the {thr:.1f} s brake limit — emergency hazard"
        return f"Gap {v:.1f} m under the {thr:.1f} m brake distance — emergency hazard"
    return f"Trigger value {v:.2f} crossed the {thr:.2f} threshold"


def _weigh_text(decision: Decision) -> str:
    """Priority comparison across actions in this frame."""
    summary = _action_priority_summary(decision.supporting_findings)
    if not summary:
        return "No findings to weigh"
    if len(summary) == 1:
        a, p = summary[0]
        return f"Only {a} ({p}) applies"
    win_a, win_p = summary[0]
    others = ", ".join(f"{a} ({p})" for a, p in summary[1:])
    return f"{win_a} ({win_p}) outranks {others}"


def _decide_text(decision: Decision) -> str:
    qualifier = _ACTION_QUALIFIER.get(decision.action, decision.action.title())
    return f"{decision.action} → {qualifier}"


def _build_causal_chain(
    decision: Decision, salient: Optional[DetectedObject]
) -> list[CausalStep]:
    """Synthesize the four-step OBSERVE → EVAL → WEIGH → DECIDE chain.

    Deterministic, derived entirely from existing fields on ``Decision`` and
    the resolved salient ``DetectedObject``.
    """
    return [
        CausalStep(1, "OBSERVE", _observe_text(decision, salient)),
        CausalStep(2, "EVAL",    _eval_text(decision)),
        CausalStep(3, "WEIGH",   _weigh_text(decision)),
        CausalStep(4, "DECIDE",  _decide_text(decision)),
    ]


# =============================================================================
# PRISM Signature Output — Reasoned Alert
# =============================================================================

# The four canonical questions; render order matches the dict iteration order.
SIGNATURE_QUESTIONS = {
    "context": "What is happening in Road",
    "risk":    "Level of Risk",
    "action":  "Action taken by ADAS",
    "reason":  "Reasoning behind the Action",
}

# Per-action templates.
#
# CONTEXT is a *semantic interpretation of the driving environment* — the
# kind of higher-order reading that threshold-based ADAS (AEB, ACC, BSW, LKA)
# do not produce. It frames the SITUATION (VRU safety event, lane-change
# intent conflict, cooperative car-following, behavioral threats) rather
# than naming the detected object. The detected-object detail lives in
# REASON, where it belongs.
#
# ACTION names the *active vehicle function* (what the ADAS is doing now).
_SIG_TEMPLATES: dict[str, dict[str, str]] = {
    "BRAKE": {
        "context": "Imminent forward collision developing — {density_phrase}",
        "action":  "Autonomous Emergency Braking engaged",
    },
    "STOP": {
        "context": "Vulnerable road user crossing the protected path",
        "action":  "Auto-Hold engaged",
    },
    "YIELD": {
        "context": "Pedestrian activity within caution proximity",
        "action":  "Throttle eased · monitoring",
    },
    "FOLLOW": {
        "context": "Sustained car-following in {density_phrase}",
        "action":  "Adaptive Cruise engaged",
    },
    "INHIBIT_LANE_CHANGE": {
        "context": "Lane-change opportunity blocked by adjacent traffic",
        "action":  "Lane-Change Inhibit active",
    },
    "CRUISE": {
        "context": "Open road operation, no behavioral threats",
        "action":  "Cruise Control active",
    },
}


def _signature_reason(
    decision: Decision,
    salient: Optional[DetectedObject],
) -> str:
    """Plain-language driver-facing reasoning for the action taken.

    Reuses the salient ``DetectedObject`` and the winning finding's threshold
    (parsed via the existing ``_extract_margin``) so every number in the
    prose comes from already-computed data — no recomputation.
    """
    if salient is None or decision.action == "CRUISE":
        return (
            "Road ahead is clear and no vehicles are inside follow range, "
            "so we're holding cruise speed."
        )

    noun = _friendly_noun(salient.raw_category)
    distance = round(salient.distance, 1)
    winning = _winning_finding(decision)
    margin = _extract_margin(winning.reason) if winning else None

    if decision.action == "BRAKE":
        if margin is not None:
            v, thr = margin
            # margin gives (value, threshold) — for R1 the trigger can be gap
            # or TTC; prefer 'gap' wording since it's the most familiar.
            return (
                f"The {noun} {distance} m ahead in your lane is already inside "
                f"the {thr:.1f} m brake distance — we're pressing the brakes "
                f"firmly to avoid a collision."
            )
        return (
            f"The {noun} {distance} m ahead in your lane is too close — "
            "we're pressing the brakes firmly to avoid a collision."
        )

    if decision.action == "STOP":
        thr_clause = ""
        if margin is not None:
            _, thr = margin
            thr_clause = f", inside our {thr:.1f} m stop zone"
        return (
            f"A {noun} is {distance} m from you{thr_clause} and entering your "
            "path — we're holding position until they're clear."
        )

    if decision.action == "YIELD":
        thr_clause = ""
        if margin is not None:
            _, thr = margin
            thr_clause = f", inside the {thr:.1f} m caution radius"
        return (
            f"A {noun} is {distance} m near your path{thr_clause} — "
            "we're easing off the speed to give them room."
        )

    if decision.action == "FOLLOW":
        thr_clause = ""
        if margin is not None:
            _, thr = margin
            thr_clause = f" within the {thr:.1f} m follow distance"
        return (
            f"There's a {noun} {distance} m ahead in your lane{thr_clause} — "
            "we're matching their speed to keep a safe gap."
        )

    if decision.action == "INHIBIT_LANE_CHANGE":
        side = "left" if salient.y > 0 else "right"
        return (
            f"A {noun} is moving in your {side} blind spot, so a lane change "
            "isn't safe right now — we're holding the lane and easing speed."
        )

    return (
        f"A {noun} at {distance} m requires action — proceeding with "
        f"{decision.action.lower()}."
    )


def _build_signature_output(
    *,
    decision: Decision,
    salient: Optional[DetectedObject],
    risk_label: str,
    context_label: str,
) -> SignatureOutput:
    """Return the CONTEXT / RISK / ACTION / REASON 4-field Reasoned Alert.

    CONTEXT is a semantic interpretation of the driving environment derived
    from the action class + the density label. Object specifics live in
    REASON.
    """
    template = _SIG_TEMPLATES.get(decision.action, _SIG_TEMPLATES["CRUISE"])

    density_phrase = (context_label or "Clear road").lower()
    context_template = template["context"]
    if "{density_phrase}" in context_template:
        context = context_template.format(density_phrase=density_phrase)
    else:
        context = context_template

    return SignatureOutput(
        context=context,
        risk=risk_label or "LOW",
        action=template["action"],
        reason=_signature_reason(decision, salient),
    )


def format_signature_output(n: Narration) -> Optional[SignatureOutput]:
    """Return the 4-field Reasoned Alert for the app to render."""
    return n.signature
