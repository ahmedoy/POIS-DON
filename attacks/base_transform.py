from torchvision.transforms.transforms import Compose
from typing import Any
import random
from enum import Enum, auto
from attacks.lable_transformer import LableTransformer


class TransformMode(Enum):
    DEFAULT = auto()
    CLEAN = auto()
    POISON = auto()


class AttackTransform:
    def __init__(
        self,
        clean_image_transform: Compose,
        poison_image_transform: Compose | None,  # Set to None in case of clean training
        label_transformer: LableTransformer
        | None,  # Set to None in case of clean training
        poison_rate: float,
    ):
        self.clean_image_transform = clean_image_transform
        self.poison_image_transform = poison_image_transform
        self.label_transformer = label_transformer
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

        # Used in case Default Mode tries to poison a class that's not intended for poisoning.
        if poison:
            if not self.label_transformer:
                raise ValueError("Attempting to poison but no lable transformer found.")

            is_eligible = label in set(self.label_transformer.lable_mapping.keys())

            if not is_eligible:
                if self.transform_mode is TransformMode.POISON:
                    # Caller explicitly requested a fully-poisoned pass, so an
                    # ineligible label reaching here means the caller failed to
                    # filter — this should never silently become "clean".
                    raise ValueError(
                        f"Lable {label} is not a poisonable class, but "
                        "TransformMode.POISON requires every sample to be poisoned."
                    )
                poison = False  # DEFAULT mode: fine to back off per-sample

        if poison:
            if (
                self.poison_image_transform is not None
                and self.label_transformer is not None
            ):
                image, label = (
                    self.poison_image_transform(image),
                    self.label_transformer(label),
                )
            else:
                raise ValueError(
                    "Trying to poison a sample but poison transform or label transform not found."
                )
        else:
            image, label = self.clean_image_transform(image), label

        return image, label
