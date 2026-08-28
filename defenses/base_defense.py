from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from schema.defense_experiment import (
    AnomalyResult,
    DefenseExperiment,
    ModelCollection,
)
from utils.storage_layout import TrainedModelPath


class DefenseBase(ABC):
    """
    Base class for all Trojan defenses.

    The user specifies collections of models rather than filesystem paths.
    This class reconstructs the directories and discovers all contained models.
    """

    def __init__(self) -> None:
        self.experiment: DefenseExperiment | None = None

    def set_config(self, experiment: DefenseExperiment):
        self.experiment = experiment

    @staticmethod
    def _resolve_model_paths(
        collections: Iterable[ModelCollection],
    ) -> list[Path]:
        """
        Converts experiment specifications into individual model directories.
        """

        model_paths: list[Path] = []

        for collection in collections:
            base_dir = TrainedModelPath.get_model_dir(
                image_dataset_name=collection.image_dataset_name,
                exp_name=collection.exp_name,
                run_id=collection.run_id,
                attack_name=collection.attack_name,
                split=collection.split,
                model_architecture_name=collection.model_architecture_name,
                model_hash_id="1",
            ).parent

            if not base_dir.exists():
                raise FileNotFoundError(f"Model directory does not exist: {base_dir}")
            for model_dir in sorted(base_dir.iterdir()):
                if model_dir.is_dir():
                    model_paths.append(model_dir)

        return model_paths

    @abstractmethod
    def train_defense(self):
        pass

    @abstractmethod
    def test_defense(
        self,
    ) -> tuple[
        list[AnomalyResult], dict[str, Any]
    ]:  # returns list of results with optional additional metadata
        pass
