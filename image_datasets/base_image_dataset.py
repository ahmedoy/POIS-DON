from abc import ABC, abstractmethod
from typing import List, Any


class ImageDatasetConfigBase(ABC):
    """Abstract base class for image datasets."""

    # ---- Required dataset metadata ----

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def img_height(self) -> int:
        pass

    @property
    @abstractmethod
    def img_width(self) -> int:
        pass

    @property
    @abstractmethod
    def num_channels_input(self) -> int:
        pass

    @property
    @abstractmethod
    def img_extension(self) -> str:
        pass

    # ---- Transforms ----

    @abstractmethod
    def get_transforms(self, is_train: bool) -> List[Any]:
        """
        Return the base list of transforms for this dataset.
        Should NOT return a composed transform to allow an architecture to modify them.
        """
        pass
