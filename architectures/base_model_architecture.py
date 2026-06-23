from abc import ABC, abstractmethod
from typing import List, Any, Type, Callable
from torch import nn
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase


class ModelArchitectureBase(ABC):
    """Abstract base class for model architectures."""

    @abstractmethod
    def get_model_factory(
        self,
        image_dataset: ImageDatasetConfigBase,
        pretrained: bool,
    ) -> Callable[[], nn.Module]:  # Returns a module class not an instance. This makes reinitializing with new weights easier.
        pass

    @abstractmethod
    def get_transforms(
        self, image_dataset: ImageDatasetConfigBase, is_train: bool
    ) -> List[Any]:
        """
        Return list of transforms for the model on a particular dataset.
        Should NOT return a composed transform to allow training method to modify them.
        """
        pass
