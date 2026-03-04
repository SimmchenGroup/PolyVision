from dataclasses import dataclass
from typing import List, Dict

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