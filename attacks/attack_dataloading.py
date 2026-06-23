from torch.utils.data import Dataset, DataLoader
from utils.storage_layout import ImageDatasetPaths
from PIL import Image
import warnings
from typing import List, Tuple
from attacks.base_transform import AttackTransform, TransformMode


class DefaultImageTorchDataset(Dataset):
    def __init__(
        self,
        input_channels: int,  # Used to either read in greyscale or rgb
        data: List[Tuple[str, int]],
        attack_transform: AttackTransform,
    ):
        """
        Args:
            data: List of (image_path, label) tuples
            transform: torchvision.transforms.Compose or None
        """
        self.data = data

        self.transform = attack_transform

        if input_channels == 1:
            self.channel_mode = "L"
        else:
            self.channel_mode = "RGB"

    def set_transform_mode(self, transform_mode: TransformMode):
        self.transform.set_transform_mode(transform_mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # Load image
        img = Image.open(img_path).convert(self.channel_mode)
        img, label = self.transform(img, label)
        return img, label


def get_dataloader(
    input_channels: int,  # Used to either read in greyscale or rgb
    data: List[Tuple[str, int]],
    attack_transform: AttackTransform,
    batch_size: int,
    shuffle: bool,
):
    image_dataset = DefaultImageTorchDataset(
        input_channels=input_channels, data=data, attack_transform=attack_transform
    )

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def flatten_data_from_dir(
    dataset_name: str, num_classes: int, split: str, img_extension: str
):
    flattened_data = []

    for img_class in range(num_classes):
        img_dir = ImageDatasetPaths.get_images_class_dir(
            dataset_name=dataset_name, split=split, img_class=str(img_class)
        )

        if not img_dir.exists():
            warnings.warn(f"Image Class directory does not exist: {img_dir}")

        # for every file in the img_dir, if its extension ends with img_extension add (filepath, img_class) to flattened data
        for file_path in img_dir.iterdir():
            if file_path.is_file() and file_path.suffix == img_extension:
                flattened_data.append((str(file_path), img_class))

    return flattened_data
