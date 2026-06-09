"""End-to-end per-frame orchestration.

Wires the perception layer, cross-frame velocity, the reasoning layers
(A: relations, B: rules, C: arbiter), and emits a ``Decision`` carrying its
full justification trace.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional

from src.config import Taxonomy, load_rules, load_taxonomy
from src.object_list import DetectedObject, EgoState, build_object_list
from src.reasoning.arbiter import Decision, decide
from src.reasoning.narrate import Narration, format_text, narrate
from src.reasoning.relations import Relations, relations_for
from src.reasoning.rules import apply_rules
from src.store import NuMiniStore
from src.velocity import attach_ego_speed, attach_velocity, dt_seconds


@dataclass
class FrameResult:
    sample_token: str
    ego: EgoState
    objects: list[DetectedObject]
    decision: Decision
    narration: Narration
    facts: dict[str, Relations]


def run_frame(
    store: NuMiniStore,
    sample_token: str,
    rules: dict[str, Any],
    taxonomy: Taxonomy,
    prev: Optional[FrameResult] = None,
) -> FrameResult:
    ego, objects = build_object_list(store, sample_token, rules, taxonomy)
    if prev is not None:
        dt = dt_seconds(prev.ego.timestamp, ego.timestamp)
        attach_velocity(prev.objects, objects, dt)
        attach_ego_speed(prev.ego, ego)
    findings = apply_rules(objects, rules)
    decision = decide(findings, frame_token=sample_token, num_objects=len(objects))
    facts = {o.id: relations_for(o, rules) for o in objects}
    narration = narrate(decision, objects, facts, rules)
    return FrameResult(
        sample_token=sample_token,
        ego=ego,
        objects=objects,
        decision=decision,
        narration=narration,
        facts=facts,
    )


def run_scene(
    store: NuMiniStore,
    scene_token: str,
    rules: dict[str, Any],
    taxonomy: Taxonomy,
) -> list[FrameResult]:
    """Run every keyframe of a scene end-to-end, threading prev for velocity."""
    results: list[FrameResult] = []
    prev: Optional[FrameResult] = None
    for sample in store.samples_in_scene(scene_token):
        r = run_frame(store, sample["token"], rules, taxonomy, prev=prev)
        results.append(r)
        prev = r
    return results


# ----------------------------------------------------------------- demo CLI


def _print_decision(r: FrameResult, frame_idx: int) -> None:
    d = r.decision
    print()
    print("=" * 80)
    print(f"FRAME {frame_idx}  sample={r.sample_token[:8]}..  objects={d.num_objects}")
    print(
        f"EgoState global=({r.ego.x:.1f},{r.ego.y:.1f}) yaw={r.ego.yaw:+.2f} "
        f"speed={r.ego.speed:.2f} m/s"
    )
    print()
    print(format_text(r.narration))


def _main(argv: list[str]) -> int:
    """python -m src.pipeline [json_dir] [scene_idx] [n_frames]"""
    json_dir = argv[1] if len(argv) > 1 else "./nuscenes-mini-JSON"
    scene_idx = int(argv[2]) if len(argv) > 2 else 0
    n_frames = int(argv[3]) if len(argv) > 3 else 5

    store = NuMiniStore(json_dir)
    rules = load_rules()
    taxonomy = load_taxonomy()
    scene = store.lists["scene"][scene_idx]
    print(f"Scene: {scene['name']} - {scene['description']}")
    results = run_scene(store, scene["token"], rules, taxonomy)
    for i, r in enumerate(results[:n_frames]):
        _print_decision(r, i)
    print()
    print(f"Summary: ran {len(results)} frames; "
          f"actions={[r.decision.action for r in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
