import warnings
from collections.abc import Callable

import torch
from torch import nn

from architectures.base_model_architecture import ModelArchitectureBase
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase


class SimpleCNN(nn.Module):
    """Runtime-generated CNN class: no-arg constructor creates a fresh model."""

    def __init__(self, H: int, W: int, C: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        H_out = max(1, H // 2)
        W_out = max(1, W // 2)
        flattened = 64 * H_out * W_out

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNNArchitecture(ModelArchitectureBase):
    def get_model_factory(
        self,
        image_dataset: ImageDatasetConfigBase,
        pretrained: bool,
    ) -> Callable[[], nn.Module]:
        if pretrained:
            warnings.warn(
                "SimpleCNN does not support pretrained weights. Models will be randomly initialized."
            )

        n_classes = int(image_dataset.num_classes)
        H = int(image_dataset.img_height)
        W = int(image_dataset.img_width)
        C = int(image_dataset.num_channels_input)

        def model_factory() -> nn.Module:
            return SimpleCNN(H=H, W=W, C=C, n_classes=n_classes)

        return model_factory

    def get_transforms(self, image_dataset: ImageDatasetConfigBase, is_train: bool):
        return image_dataset.get_transforms(is_train)
