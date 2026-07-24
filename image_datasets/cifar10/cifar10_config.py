from torchvision import transforms

from image_datasets.base_image_dataset_config import ImageDatasetConfigBase


class CIFAR10Config(ImageDatasetConfigBase):
    @property
    def dataset_name(self) -> str:
        return "cifar10"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def img_height(self) -> int:
        return 32

    @property
    def img_width(self) -> int:
        return 32

    @property
    def num_channels_input(self) -> int:
        return 3

    @property
    def img_extension(self) -> str:
        return ".png"

    def get_transforms(self, is_train: bool):
        if is_train:
            return [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        else:
            return [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
