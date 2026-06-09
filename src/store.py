"""Custom JSON-only loader for nuScenes Mini.

Reads the 13 JSON tables into token-keyed dicts and provides relational helpers.
Never touches sensor files (sample_data filenames are referenced only for
ego_pose lookup, not opened). Pure stdlib + dict joins.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

TABLES = (
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
)

# Channel preference order when picking the keyframe sample_data row whose
# ego_pose we treat as canonical for a sample.
_EGO_POSE_CHANNEL_PREFERENCE = (
    "LIDAR_TOP",
    "CAM_FRONT",
    "RADAR_FRONT",
)


class NuMiniStore:
    """In-memory relational store over the 13 nuScenes JSON tables.

    Parameters
    ----------
    json_dir : str
        Path to the folder containing the JSONs. The loader auto-detects
        whether the files are at ``json_dir`` directly or nested one level
        deeper under ``v1.0-mini/``.
    """

    def __init__(self, json_dir: str):
        self.root = self._resolve_root(json_dir)
        self._validate_files(self.root)

        # token -> record (one dict per table)
        self.tables: dict[str, dict[str, dict[str, Any]]] = {}
        # original ordered list (needed when iteration order matters)
        self.lists: dict[str, list[dict[str, Any]]] = {}

        for name in TABLES:
            path = os.path.join(self.root, f"{name}.json")
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            self.lists[name] = rows
            self.tables[name] = {row["token"]: row for row in rows}

        self._build_indices()

    # ----- construction helpers -----

    @staticmethod
    def _resolve_root(json_dir: str) -> str:
        """Return the directory that actually contains sample_annotation.json."""
        candidates = [
            json_dir,
            os.path.join(json_dir, "v1.0-mini"),
        ]
        for c in candidates:
            if os.path.isfile(os.path.join(c, "sample_annotation.json")):
                return c
        raise FileNotFoundError(
            f"Could not find sample_annotation.json under {json_dir!r} "
            f"or {json_dir!r}/v1.0-mini/. Point the loader at the folder "
            f"that holds the 13 nuScenes JSON tables."
        )

    @staticmethod
    def _validate_files(root: str) -> None:
        missing = [
            f"{n}.json"
            for n in TABLES
            if not os.path.isfile(os.path.join(root, f"{n}.json"))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing nuScenes JSON tables in {root!r}: {missing}. "
                "All 13 tables are required."
            )

    def _build_indices(self) -> None:
        # annotations grouped by sample_token
        self._anns_by_sample: dict[str, list[dict[str, Any]]] = {}
        for ann in self.lists["sample_annotation"]:
            self._anns_by_sample.setdefault(ann["sample_token"], []).append(ann)

        # (sample_token, channel) -> sample_data record, keyframes only
        self._keyframe_sd_by_sample_channel: dict[tuple[str, str], dict[str, Any]] = {}
        # sample_token -> list of keyframe sample_data records (any channel)
        self._keyframe_sd_by_sample: dict[str, list[dict[str, Any]]] = {}
        for sd in self.lists["sample_data"]:
            if not sd.get("is_key_frame"):
                continue
            cs = self.tables["calibrated_sensor"].get(sd["calibrated_sensor_token"])
            if cs is None:
                continue
            sensor = self.tables["sensor"].get(cs["sensor_token"])
            if sensor is None:
                continue
            channel = sensor["channel"]
            self._keyframe_sd_by_sample_channel[(sd["sample_token"], channel)] = sd
            self._keyframe_sd_by_sample.setdefault(sd["sample_token"], []).append(sd)

    # ----- generic accessors -----

    def get(self, table: str, token: str) -> dict[str, Any]:
        try:
            return self.tables[table][token]
        except KeyError as e:
            raise KeyError(f"No row in table {table!r} with token {token!r}") from e

    # ----- relational helpers -----

    def samples_in_scene(self, scene_token: str) -> list[dict[str, Any]]:
        """Walk the doubly-linked sample list from first to last in scene order."""
        scene = self.get("scene", scene_token)
        out: list[dict[str, Any]] = []
        token = scene["first_sample_token"]
        last = scene["last_sample_token"]
        while token:
            sample = self.get("sample", token)
            out.append(sample)
            if token == last:
                break
            token = sample.get("next") or ""
        return out

    def annotations_for_sample(self, sample_token: str) -> list[dict[str, Any]]:
        return list(self._anns_by_sample.get(sample_token, ()))

    def category_name(self, instance_token: str) -> str:
        inst = self.get("instance", instance_token)
        cat = self.get("category", inst["category_token"])
        return cat["name"]

    def attribute_names(self, attribute_tokens: list[str]) -> list[str]:
        return [self.tables["attribute"][t]["name"] for t in attribute_tokens]

    def visibility_level(self, visibility_token: str) -> int:
        """Return the integer visibility tier (1..4); 0 if token missing.

        nuScenes encodes the tier in the *token* (``"1".."4"``); the ``level``
        field is a human-readable range like ``"v0-40"``.
        """
        if not visibility_token:
            return 0
        try:
            return int(visibility_token)
        except (TypeError, ValueError):
            return 0

    def ego_pose_for_sample(self, sample_token: str) -> dict[str, Any]:
        """Return the ego_pose record for a sample's keyframe.

        Prefers LIDAR_TOP, falls back through a fixed channel preference list,
        and finally to any keyframe sample_data row for the sample.
        """
        for channel in _EGO_POSE_CHANNEL_PREFERENCE:
            sd = self._keyframe_sd_by_sample_channel.get((sample_token, channel))
            if sd is not None:
                return self.get("ego_pose", sd["ego_pose_token"])
        rows = self._keyframe_sd_by_sample.get(sample_token)
        if rows:
            return self.get("ego_pose", rows[0]["ego_pose_token"])
        raise LookupError(
            f"No keyframe sample_data found for sample_token={sample_token!r}; "
            "cannot resolve ego_pose."
        )


def _print_scene_summary(store: NuMiniStore) -> None:
    print(f"Loaded NuMiniStore from {store.root}")
    print(f"Tables: {{name: rows}}")
    for name in TABLES:
        print(f"  {name:<20s} {len(store.lists[name]):>6d}")
    print()
    print(f"{'name':<14s} {'frames':>7s}  description")
    print("-" * 80)
    for scene in store.lists["scene"]:
        frames = store.samples_in_scene(scene["token"])
        print(
            f"{scene['name']:<14s} {len(frames):>7d}  {scene['description']}"
        )


def _main(argv: list[str]) -> int:
    json_dir = argv[1] if len(argv) > 1 else "./nuscenes-mini-JSON"
    store = NuMiniStore(json_dir)
    _print_scene_summary(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
