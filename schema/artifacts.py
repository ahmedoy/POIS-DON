from dataclasses import dataclass, field
from typing import Any

from schema.attack_experiment import AttackExperiment


@dataclass(frozen=True)
class TrainedModelStats:
    train_clean_accuracy: float
    train_attack_success_rate: float
    validation_clean_accuracy: float
    validation_attack_success_rate: float
    test_clean_accuracy: float
    test_attack_success_rate: float


@dataclass(frozen=True)
class TrainedModelArtifact:
    is_trojan: bool
    model_hash: str
    split: str  # train, validation, or test
    poison_rate: float
    trained_model_stats: TrainedModelStats
    timestamp: str
    repository_clean: bool
    repo_hash: str
    attack_experiment: AttackExperiment
    clean_eval_transform_str: str  # Add str representing eval transform incase pickling fails to properly save it
    metadata_dict: dict[str, Any] = field(default_factory=dict)
