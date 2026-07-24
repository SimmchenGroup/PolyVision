"""
Typed configuration for a classification run — hyperparameters, class list, backbone
choice, and dataset paths — loaded from configs/config.json and passed through the
training pipeline so every run is reproducible from its saved config.
"""
from __future__ import annotations
from typing import List, Dict, Iterable
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ExperimentConfig:
    training_type: str
    learning_rate: float
    base_dirs: Dict[str, str]
    microplastic_classes: List[str]
    whisky_classes: List[str]
    weights_path: str
    save_path: str
    epochs: int = 15
    batch_size: int = 64
    image_size: tuple = (150, 150)
    model_name: str = "inception"
    steps_per_epoch: int | None = None

    train_base_dir_override: str | None = None

@dataclass(frozen=True)
class MicroplasticClass:
  id: int
  name: str

def load_config(path: str | Path) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

def get_microplastic_classes(config: dict) -> list[MicroplasticClass]:
  items = config.get("classes", {}).get("items", [])
  return [MicroplasticClass(int(x["id"]), str(x["name"])) for x in items]