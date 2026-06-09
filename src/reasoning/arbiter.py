"""Layer C: arbiter.

Picks the highest-priority Finding as the frame's Decision; ties broken by
rule evaluation order (preserved by ``apply_rules``). The full ordered list
of findings is attached as the justification trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.reasoning.rules import Finding, cruise_fallback


@dataclass
class Decision:
    action: str
    priority: int
    primary_reason: str
    supporting_findings: list[Finding] = field(default_factory=list)
    frame_token: str = ""
    num_objects: int = 0


def decide(
    findings: list[Finding],
    frame_token: str,
    num_objects: int,
) -> Decision:
    if not findings:
        findings = [cruise_fallback()]
    # Stable sort preserves rule order for equal priorities.
    ordered = sorted(
        enumerate(findings),
        key=lambda ix_f: (-ix_f[1].priority, ix_f[0]),
    )
    winner = ordered[0][1]
    ordered_findings = [f for _, f in ordered]
    return Decision(
        action=winner.action,
        priority=winner.priority,
        primary_reason=winner.reason,
        supporting_findings=ordered_findings,
        frame_token=frame_token,
        num_objects=num_objects,
    )
