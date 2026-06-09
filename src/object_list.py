"""Per-frame perception layer.

Reads nuScenes sample_annotation rows and transforms them into a list of
``DetectedObject`` instances expressed in the ego-vehicle frame
(+x forward, +y left).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from pyquaternion import Quaternion

from src.config import Taxonomy, load_rules, load_taxonomy
from src.store import NuMiniStore


@dataclass
class EgoState:
    """Ego pose at a single keyframe (global frame for x,y,yaw)."""
    x: float
    y: float
    yaw: float
    speed: float
    timestamp: int  # microseconds


@dataclass
class DetectedObject:
    """One annotated object expressed in the ego frame (+x forward, +y left)."""
    id: str                       # instance_token (stable across frames)
    raw_category: str             # untouched nuScenes category string
    cls: str                      # collapsed class (VEHICLE/LARGE_VEHICLE/...)
    state: str                    # collapsed state (MOVING/STOPPED/PARKED/STANDING)
    x: float                      # ego-frame forward
    y: float                      # ego-frame left
    distance: float               # sqrt(x²+y²)
    yaw: float                    # ego-frame heading (radians)
    size: tuple[float, float, float]  # (w, l, h) in metres
    vx: Optional[float] = None    # ego-frame velocity (set by velocity.py)
    vy: Optional[float] = None
    num_lidar_pts: int = 0
    visibility: int = 0           # 1..4 tier
    annotation_token: str = ""    # original sample_annotation token
    raw_attributes: list[str] = field(default_factory=list)


def ego_state_from_pose(pose: dict[str, Any], timestamp: int) -> EgoState:
    q = Quaternion(pose["rotation"])
    yaw = q.yaw_pitch_roll[0]
    tx, ty, _tz = pose["translation"]
    return EgoState(x=float(tx), y=float(ty), yaw=float(yaw), speed=0.0, timestamp=int(timestamp))


def _ego_frame_xy(ann_translation, ego_t: np.ndarray, R_ego_T: np.ndarray) -> tuple[float, float, float]:
    rel = R_ego_T @ (np.asarray(ann_translation, dtype=float) - ego_t)
    return float(rel[0]), float(rel[1]), float(rel[2])


def build_object_list(
    store: NuMiniStore,
    sample_token: str,
    rules: dict[str, Any],
    taxonomy: Taxonomy,
) -> tuple[EgoState, list[DetectedObject]]:
    """Return (ego_state, detected_objects) for one keyframe.

    Objects are filtered by ``rules['min_lidar_pts']`` and ``rules['min_visibility']``.
    """
    sample = store.get("sample", sample_token)
    pose = store.ego_pose_for_sample(sample_token)
    ego = ego_state_from_pose(pose, sample["timestamp"])

    ego_t = np.asarray(pose["translation"], dtype=float)
    R_ego_T = Quaternion(pose["rotation"]).rotation_matrix.T

    min_lidar = int(rules.get("min_lidar_pts", 0))
    min_vis = int(rules.get("min_visibility", 0))

    objects: list[DetectedObject] = []
    for ann in store.annotations_for_sample(sample_token):
        if ann.get("num_lidar_pts", 0) < min_lidar:
            continue
        vis_level = store.visibility_level(ann.get("visibility_token", ""))
        if vis_level < min_vis:
            continue

        x, y, _z = _ego_frame_xy(ann["translation"], ego_t, R_ego_T)
        obj_yaw_global = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
        yaw_ego = _wrap_pi(obj_yaw_global - ego.yaw)

        raw_cat = store.category_name(ann["instance_token"])
        raw_attrs = store.attribute_names(ann.get("attribute_tokens", []) or [])

        size_w, size_l, size_h = ann["size"]
        objects.append(
            DetectedObject(
                id=ann["instance_token"],
                raw_category=raw_cat,
                cls=taxonomy.class_for(raw_cat),
                state=taxonomy.state_for(raw_attrs),
                x=x,
                y=y,
                distance=math.hypot(x, y),
                yaw=yaw_ego,
                size=(float(size_w), float(size_l), float(size_h)),
                vx=None,
                vy=None,
                num_lidar_pts=int(ann.get("num_lidar_pts", 0)),
                visibility=vis_level,
                annotation_token=ann["token"],
                raw_attributes=list(raw_attrs),
            )
        )

    return ego, objects


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------- demo / CLI


def _format_object(o: DetectedObject) -> str:
    return (
        f"  {o.cls:<13s} {o.state:<8s} "
        f"x={o.x:+7.2f}  y={o.y:+7.2f}  d={o.distance:6.2f}  "
        f"yaw={o.yaw:+5.2f}  pts={o.num_lidar_pts:>3d} v={o.visibility} "
        f"raw={o.raw_category}"
    )


def _main(argv: list[str]) -> int:
    """python -m src.object_list [json_dir] [scene_index] [frame_index]"""
    json_dir = argv[1] if len(argv) > 1 else "./nuscenes-mini-JSON"
    scene_idx = int(argv[2]) if len(argv) > 2 else 0
    frame_idx = int(argv[3]) if len(argv) > 3 else 0

    store = NuMiniStore(json_dir)
    rules = load_rules()
    taxonomy = load_taxonomy()

    scene = store.lists["scene"][scene_idx]
    frames = store.samples_in_scene(scene["token"])
    sample = frames[frame_idx]

    ego, objects = build_object_list(store, sample["token"], rules, taxonomy)
    print(f"Scene: {scene['name']} - {scene['description']}")
    print(f"Frame {frame_idx}/{len(frames)-1} sample_token={sample['token']}")
    print(
        f"EgoState global x={ego.x:.2f} y={ego.y:.2f} yaw={ego.yaw:+.3f} "
        f"ts={ego.timestamp}"
    )
    print(f"Objects after filters (min_lidar_pts>={rules['min_lidar_pts']}, "
          f"min_visibility>={rules['min_visibility']}): {len(objects)}")
    for o in objects:
        print(_format_object(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
