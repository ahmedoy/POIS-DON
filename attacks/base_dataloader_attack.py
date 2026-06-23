from abc import ABC, abstractmethod
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from schema.attack_experiment import AttackExperiment, AttackDataLoaders
from architectures.base_model_architecture import ModelArchitectureBase


class DataloaderBasedAttackBase(ABC):
    @abstractmethod
    def get_data_loaders(
        self,
        experiment_config: AttackExperiment,
        model_architecture: ModelArchitectureBase,
        image_dataset: ImageDatasetConfigBase,
    ) -> (
        AttackDataLoaders
    ):  # Returns train_loader, clean_eval_loader, poison_eval_loader
        pass

    @property
    @abstractmethod
    def is_trojan(self) -> bool:
        # Whether this training method creates a benign or malicious model
        pass


# TODO Later on add in the config or in the attack class itself a binary flag that controls whether the
# standard trainer is used or whether the attack itself should fully implement training.
# This is useful for attacks that modify loss, optimization process itself, etc.
