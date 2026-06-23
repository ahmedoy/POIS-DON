from torchvision.transforms.transforms import Compose
from typing import Any, Callable
import random
from enum import Enum, auto


class TransformMode(Enum):
    DEFAULT = auto()
    CLEAN = auto()
    POISON = auto()


class AttackTransform:
    def __init__(
        self,
        clean_image_transform: Compose,
        poison_image_transform: Compose | None,  # Set to None in case of clean training
        label_transform: Callable[[int], int]
        | None,  # Set to None in case of clean training
        poison_rate: float,
    ):
        self.clean_image_transform = clean_image_transform
        self.poison_image_transform = poison_image_transform
        self.label_transform = label_transform
        self.poison_rate = poison_rate
        self.transform_mode = TransformMode.DEFAULT

    def set_transform_mode(self, transform_mode: TransformMode):
        self.transform_mode = transform_mode

    def __call__(self, image, label: int) -> Any:

        if self.transform_mode is TransformMode.POISON:
            poison = True

        elif self.transform_mode is TransformMode.CLEAN:
            poison = False

        elif self.transform_mode is TransformMode.DEFAULT:
            poison = random.random() < self.poison_rate

        else:
            raise ValueError(f"Unknown transform mode: {self.transform_mode}")

        if poison:
            if (
                self.poison_image_transform is not None
                and self.label_transform is not None
            ):
                image, label = (
                    self.poison_image_transform(image),
                    self.label_transform(label),
                )
            else:
                raise ValueError(
                    "Trying to poison a sample but poison transform or label transform not found."
                )
        else:
            image, label = self.clean_image_transform(image), label

        return image, label
