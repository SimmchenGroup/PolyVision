"""
Helpers for reading configs/config.json.

`load_microplastic_classes` returns the (id, name) class list the GUI and annotation
pipeline use; it falls back to a built-in list (including the "-1 = auto/model" option)
if the config omits an explicit `classes.items` block.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Load and parse a JSON file as a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_microplastic_classes(config: dict) -> list[tuple[int, str]]:
    classes_block = config.get("classes", {})
    items = classes_block.get("items", [])
    if items:
        return [(int(item["id"]), str(item["name"])) for item in items]

    return [
        (-1, "Auto (model)"),
        (0, "Nylon"),
        (1, "PE"),
        (2, "PMMA"),
        (3, "PP"),
        (4, "PS"),
        (5, "PU"),
        (6, "PVC"),
    ]