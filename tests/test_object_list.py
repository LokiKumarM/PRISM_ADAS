"""Tests for the perception layer."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
from pyquaternion import Quaternion

from src.config import load_rules, load_taxonomy
from src.object_list import build_object_list, ego_state_from_pose
from src.store import NuMiniStore

JSON_DIR = os.environ.get(
    "NUSCENES_JSON_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nuscenes-mini-JSON")),
)


# ----------------------- stub store for geometry test -----------------------

class _StubStore:
    """Minimal store-shaped stub for synthetic geometry checks."""

    def __init__(self, pose, ann, raw_category, raw_attrs):
        self._pose = pose
        self._ann = ann
        self._raw_category = raw_category
        self._raw_attrs = raw_attrs

    def get(self, table, token):
        if table == "sample":
            return {"token": token, "timestamp": 0}
        raise KeyError((table, token))

    def ego_pose_for_sample(self, sample_token):
        return self._pose

    def annotations_for_sample(self, sample_token):
        return [self._ann]

    def category_name(self, instance_token):
        return self._raw_category

    def attribute_names(self, attribute_tokens):
        return list(self._raw_attrs)

    def visibility_level(self, visibility_token):
        return int(visibility_token) if visibility_token else 0


def _quat_yaw(yaw: float) -> list[float]:
    """Quaternion (w,x,y,z) for a pure yaw about +z."""
    return list(Quaternion(axis=[0, 0, 1], angle=yaw).elements)


def test_ego_state_from_pose_yaw_extraction():
    pose = {"translation": [10.0, 20.0, 1.5], "rotation": _quat_yaw(0.5)}
    ego = ego_state_from_pose(pose, timestamp=123)
    assert ego.x == 10.0 and ego.y == 20.0
    assert math.isclose(ego.yaw, 0.5, abs_tol=1e-9)
    assert ego.speed == 0.0
    assert ego.timestamp == 123


@pytest.mark.parametrize("ego_yaw", [0.0, 0.7, -1.2, math.pi / 2, math.pi])
def test_object_10m_ahead_regardless_of_ego_yaw(ego_yaw):
    """An object placed 10m forward of the ego (in global coords aligned with
    ego heading) should land at ego-frame (x≈10, y≈0)."""
    ego_t = np.array([100.0, 50.0, 0.0])
    R_ego = Quaternion(axis=[0, 0, 1], angle=ego_yaw).rotation_matrix
    obj_global = ego_t + R_ego @ np.array([10.0, 0.0, 0.0])

    pose = {"translation": ego_t.tolist(), "rotation": _quat_yaw(ego_yaw)}
    ann = {
        "token": "ann-token",
        "instance_token": "inst-token",
        "visibility_token": "4",
        "attribute_tokens": [],
        "translation": obj_global.tolist(),
        "size": [2.0, 4.5, 1.7],
        "rotation": _quat_yaw(ego_yaw),  # same heading as ego
        "num_lidar_pts": 50,
    }
    store = _StubStore(pose, ann, "vehicle.car", ["vehicle.moving"])
    rules = {"min_lidar_pts": 0, "min_visibility": 0}
    taxonomy = load_taxonomy()

    ego, objs = build_object_list(store, "sample", rules, taxonomy)
    assert len(objs) == 1
    o = objs[0]
    assert math.isclose(o.x, 10.0, abs_tol=1e-6)
    assert math.isclose(o.y, 0.0, abs_tol=1e-6)
    assert math.isclose(o.distance, 10.0, abs_tol=1e-6)
    # Object faces same direction as ego → yaw in ego frame ≈ 0
    assert abs(o.yaw) < 1e-6
    assert o.cls == "VEHICLE"
    assert o.state == "MOVING"


def test_object_to_the_left():
    """Object 5m to the left of the ego (in ego frame) should land at (0, +5)."""
    ego_yaw = 0.0
    ego_t = np.array([0.0, 0.0, 0.0])
    R_ego = Quaternion(axis=[0, 0, 1], angle=ego_yaw).rotation_matrix
    obj_global = ego_t + R_ego @ np.array([0.0, 5.0, 0.0])

    pose = {"translation": ego_t.tolist(), "rotation": _quat_yaw(ego_yaw)}
    ann = {
        "token": "ann-token",
        "instance_token": "inst-token",
        "visibility_token": "4",
        "attribute_tokens": [],
        "translation": obj_global.tolist(),
        "size": [0.6, 0.7, 1.7],
        "rotation": _quat_yaw(0.0),
        "num_lidar_pts": 10,
    }
    store = _StubStore(pose, ann, "human.pedestrian.adult", ["pedestrian.standing"])
    rules = {"min_lidar_pts": 0, "min_visibility": 0}
    ego, objs = build_object_list(store, "sample", rules, load_taxonomy())
    o = objs[0]
    assert math.isclose(o.x, 0.0, abs_tol=1e-6)
    assert math.isclose(o.y, 5.0, abs_tol=1e-6)
    assert o.cls == "PEDESTRIAN"
    assert o.state == "STANDING"


def test_filters_drop_low_lidar_and_low_visibility():
    pose = {"translation": [0.0, 0.0, 0.0], "rotation": _quat_yaw(0.0)}
    base = {
        "token": "ann",
        "instance_token": "inst",
        "visibility_token": "1",
        "attribute_tokens": [],
        "translation": [5.0, 0.0, 0.0],
        "size": [1.0, 2.0, 1.7],
        "rotation": _quat_yaw(0.0),
        "num_lidar_pts": 0,
    }
    store = _StubStore(pose, base, "vehicle.car", [])
    rules = {"min_lidar_pts": 2, "min_visibility": 1}
    _ego, objs = build_object_list(store, "sample", rules, load_taxonomy())
    assert objs == []  # filtered by lidar_pts


# ------------------------ smoke test on real data ---------------------------


@pytest.fixture(scope="module")
def store() -> NuMiniStore:
    if not os.path.isfile(os.path.join(JSON_DIR, "sample_annotation.json")):
        pytest.skip(f"nuScenes JSON folder not found at {JSON_DIR}")
    return NuMiniStore(JSON_DIR)


def test_first_sample_real_data(store):
    rules = load_rules()
    taxonomy = load_taxonomy()
    scene = store.lists["scene"][0]
    sample_token = scene["first_sample_token"]
    ego, objs = build_object_list(store, sample_token, rules, taxonomy)
    assert ego.timestamp > 0
    assert len(objs) > 0
    for o in objs:
        assert math.isfinite(o.x) and math.isfinite(o.y)
        assert o.distance == pytest.approx(math.hypot(o.x, o.y), abs=1e-6)
        assert o.cls in {"VEHICLE", "LARGE_VEHICLE", "PEDESTRIAN", "CYCLIST", "STATIC"}
        assert o.state in {"MOVING", "STOPPED", "PARKED", "STANDING"}
