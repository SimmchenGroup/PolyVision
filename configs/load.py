"""
Helpers for reading the PolyVision config (configs/config.json by default).

Which config file is used
-------------------------
`config_path()` returns `configs/config.json` unless the environment variable
POLYVISION_CONFIG is set. It may be a bare filename resolved inside `configs/`
(e.g. `config.5class.json`), or a relative/absolute path. Every entry point
(GUI, training scripts, dataset builders) goes through `load_config()`, so one
env var switches the whole project between class sets:

    $env:POLYVISION_CONFIG = "config.5class.json"   # PowerShell
    export POLYVISION_CONFIG=config.5class.json     # bash / Colab

Class helpers
-------------
`load_microplastic_classes` returns the (id, name) list the GUI uses (including the
"-1 = auto/model" option). `class_names` returns the model index -> lowercase folder
name list the classifiers/detector are trained with: ids must be contiguous 0..N-1
and, because torchvision ImageFolder assigns indices alphabetically, the names must
be in alphabetical order.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CONFIG_DIR / "config.json"
ENV_VAR = "POLYVISION_CONFIG"


def config_path() -> Path:
    """Resolve the active config file (see module docstring)."""
    override = os.environ.get(ENV_VAR, "").strip()
    if not override:
        return DEFAULT_CONFIG
    p = Path(override)
    if p.is_absolute():
        return p
    in_configs = CONFIG_DIR / p
    if in_configs.exists():
        return in_configs
    return (Path.cwd() / p).resolve()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load and parse a JSON file as a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict[str, Any]:
    """Load the active config file (respects POLYVISION_CONFIG)."""
    path = config_path()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path} ({ENV_VAR}={os.environ.get(ENV_VAR, '')!r})")
    return load_json(path)


def load_microplastic_classes(config: dict) -> list[tuple[int, str]]:
    """Return the [(id, name)] class list from config, or a built-in fallback if none is defined."""
    classes_block = config.get("classes", {})
    items = classes_block.get("items", [])
    if items:
        return [(int(item["id"]), str(item["name"])) for item in items]

    return [
        (-1, "Auto (model)"),
        (0, "Nylon"),
        (1, "PE"),
        (2, "PET"),
        (3, "PLA"),
        (4, "PMMA"),
        (5, "PP"),
        (6, "PS"),
        (7, "PU"),
        (8, "PVC"),
    ]


def class_names(config: dict) -> list[str]:
    """
    Model index -> lowercase class/folder name, from `classes.items` (ids >= 0).

    Validates that ids are contiguous 0..N-1 and alphabetical, which is what the
    ImageFolder loaders and the YOLO data.yaml assume.
    """
    items = sorted((cid, name.lower()) for cid, name in load_microplastic_classes(config) if cid >= 0)
    ids = [cid for cid, _ in items]
    names = [name for _, name in items]
    if ids != list(range(len(ids))):
        raise ValueError(f"class ids must be contiguous 0..N-1, got {ids}")
    if names != sorted(names):
        raise ValueError(
            "class names must be in alphabetical order to match ImageFolder indices: "
            f"{names} -> expected {sorted(names)}"
        )
    return names


def class_to_id(config: dict) -> dict[str, int]:
    """Lowercase class name -> model index."""
    return {name: i for i, name in enumerate(class_names(config))}
