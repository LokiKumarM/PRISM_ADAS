"""Sanity tests for the custom NuMiniStore JSON loader."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from src.store import TABLES, NuMiniStore

JSON_DIR = os.environ.get(
    "NUSCENES_JSON_DIR",
    os.path.join(os.path.dirname(__file__), "..", "nuscenes-mini-JSON"),
)
JSON_DIR = os.path.abspath(JSON_DIR)


@pytest.fixture(scope="module")
def store() -> NuMiniStore:
    if not os.path.isfile(os.path.join(JSON_DIR, "sample_annotation.json")):
        pytest.skip(f"nuScenes JSON folder not found at {JSON_DIR}")
    return NuMiniStore(JSON_DIR)


def test_all_tables_loaded(store: NuMiniStore) -> None:
    for name in TABLES:
        assert name in store.tables
        assert name in store.lists
        assert len(store.lists[name]) == len(store.tables[name])


def test_expected_row_counts(store: NuMiniStore) -> None:
    assert len(store.lists["scene"]) == 10
    assert len(store.lists["sample"]) == 404
    assert len(store.lists["category"]) == 23
    assert len(store.lists["attribute"]) == 8
    assert len(store.lists["visibility"]) == 4


def test_samples_in_scene_matches_nbr_samples(store: NuMiniStore) -> None:
    for scene in store.lists["scene"]:
        frames = store.samples_in_scene(scene["token"])
        assert len(frames) == scene["nbr_samples"], (
            f"scene {scene['name']}: walked {len(frames)} but "
            f"nbr_samples={scene['nbr_samples']}"
        )


def test_ego_pose_for_first_sample(store: NuMiniStore) -> None:
    scene = store.lists["scene"][0]
    sample_token = scene["first_sample_token"]
    pose = store.ego_pose_for_sample(sample_token)
    assert "translation" in pose and len(pose["translation"]) == 3
    assert "rotation" in pose and len(pose["rotation"]) == 4
    assert all(isinstance(v, (int, float)) for v in pose["translation"])
    assert all(isinstance(v, (int, float)) for v in pose["rotation"])


def test_ego_pose_for_every_sample(store: NuMiniStore) -> None:
    """Every keyframe in mini should have a resolvable ego_pose."""
    for sample in store.lists["sample"]:
        pose = store.ego_pose_for_sample(sample["token"])
        assert "translation" in pose


def test_category_and_attribute_lookup(store: NuMiniStore) -> None:
    ann = store.lists["sample_annotation"][0]
    name = store.category_name(ann["instance_token"])
    assert isinstance(name, str) and name
    attrs = store.attribute_names(ann["attribute_tokens"])
    assert isinstance(attrs, list)
    for a in attrs:
        assert isinstance(a, str) and a


def test_annotations_for_sample(store: NuMiniStore) -> None:
    scene = store.lists["scene"][0]
    anns = store.annotations_for_sample(scene["first_sample_token"])
    assert len(anns) > 0
    for ann in anns:
        assert ann["sample_token"] == scene["first_sample_token"]


def test_visibility_level(store: NuMiniStore) -> None:
    ann = store.lists["sample_annotation"][0]
    level = store.visibility_level(ann["visibility_token"])
    assert 0 <= level <= 4


def test_missing_files_raises(tmp_path) -> None:
    # Empty dir → loader complains about missing sample_annotation.json
    with pytest.raises(FileNotFoundError):
        NuMiniStore(str(tmp_path))


def test_partial_dir_lists_missing(tmp_path) -> None:
    # Put only sample_annotation.json so resolve_root succeeds, then expect a
    # clear missing-files error from _validate_files.
    (tmp_path / "sample_annotation.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        NuMiniStore(str(tmp_path))
    msg = str(excinfo.value)
    assert "scene.json" in msg
    assert "ego_pose.json" in msg


def test_nested_v1_mini_layout_autodetect(tmp_path) -> None:
    """Loader should find tables nested under a v1.0-mini/ subdir."""
    if not os.path.isfile(os.path.join(JSON_DIR, "sample_annotation.json")):
        pytest.skip(f"nuScenes JSON folder not found at {JSON_DIR}")
    nested = tmp_path / "v1.0-mini"
    nested.mkdir()
    for name in TABLES:
        shutil.copyfile(
            os.path.join(JSON_DIR, f"{name}.json"),
            nested / f"{name}.json",
        )
    store = NuMiniStore(str(tmp_path))
    assert store.root == str(nested)
    assert len(store.lists["scene"]) == 10
