from torchvision import transforms
from image_datasets.base_image_dataset import ImageDatasetConfigBase

class MNISTConfig(ImageDatasetConfigBase):

    @property
    def dataset_name(self) -> str:
        return "mnist"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def img_height(self) -> int:
        return 28

    @property
    def img_width(self) -> int:
        return 28

    @property
    def num_channels_input(self) -> int:
        return 1

    @property
    def img_extension(self) -> str:
        return ".png"

    def get_transforms(self, is_train: bool):
        if is_train:
            return [
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        else:
            return [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
