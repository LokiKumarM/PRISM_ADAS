"""YAML config loaders for rules and taxonomy.

Kept as a separate module so the same loaders are reused by the perception
layer, reasoning core, and Streamlit app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RULES_PATH = os.path.join(REPO_ROOT, "config", "rules.yaml")
DEFAULT_TAXONOMY_PATH = os.path.join(REPO_ROOT, "config", "taxonomy.yaml")


@dataclass(frozen=True)
class Taxonomy:
    categories: dict[str, str]
    attributes: dict[str, str]
    default_category: str
    default_state: str

    def class_for(self, raw_category: str) -> str:
        return self.categories.get(raw_category, self.default_category)

    def state_for(self, raw_attributes: list[str]) -> str:
        """First mapped attribute wins; falls back to default_state."""
        for a in raw_attributes:
            if a in self.attributes:
                return self.attributes[a]
        return self.default_state


def load_rules(path: str = DEFAULT_RULES_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_taxonomy(path: str = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Taxonomy(
        categories=dict(raw.get("categories", {})),
        attributes=dict(raw.get("attributes", {})),
        default_category=str(raw.get("default_category", "STATIC")),
        default_state=str(raw.get("default_state", "STANDING")),
    )
