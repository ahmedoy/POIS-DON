from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelCollection:
    image_dataset_name: str
    exp_name: str
    run_id: str
    attack_name: str
    split: str
    model_architecture_name: str


@dataclass(frozen=True)
class DefenseExperiment:
    dataset_name: str
    experiment_id: str
    run_id: str
    defense_name: str
    seed: int
    turn_off_cuda_optimizations: bool

    train_model_collections: list[ModelCollection]
    val_model_collections: list[ModelCollection]
    test_model_collections: list[ModelCollection]

    defense_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DefenseExperiment":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        image_dataset_name = data["dataset_name"]
        experiment_id = data["experiment_id"]

        data["train_model_collections"] = [
            ModelCollection(
                image_dataset_name=image_dataset_name,
                exp_name=experiment_id,
                split="train",
                **x,
            )
            for x in data["train_model_collections"]
        ]

        data["val_model_collections"] = [
            ModelCollection(
                image_dataset_name=image_dataset_name,
                exp_name=experiment_id,
                split="val",
                **x,
            )
            for x in data["val_model_collections"]
        ]

        data["test_model_collections"] = [
            ModelCollection(
                image_dataset_name=image_dataset_name,
                exp_name=experiment_id,
                split="test",
                **x,
            )
            for x in data["test_model_collections"]
        ]

        return cls(**data)


@dataclass
class AnomalyResult:
    model_path: Path
    anomaly_score: float
    is_anomaly: bool


@dataclass
class DefenseExperimentArtifact:
    timestamp: str
    repository_clean: bool
    repo_hash: str
    metadata_dict: dict[str, Any]
    defense_experiment_config: dict[str, Any]
