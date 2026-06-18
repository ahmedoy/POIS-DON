from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class TrainedModelArtifact:
    is_trojan: bool
    model_hash: str
    seed: int
    dataset_name: str
    split: str  # train, validation, or test
    architecture_name: str
    attack_name: str
    poison_rate: str
    timestamp: int
    validation_clean_accuracy: float
    validation_attack_success_rate: float
    test_clean_accuracy: float
    test_attack_success_rate: float

    config_dict: dict[str, Any] = field(default_factory=dict)
    metadata_dict: dict[str, Any] = field(default_factory=dict)