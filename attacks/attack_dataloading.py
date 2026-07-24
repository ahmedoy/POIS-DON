import warnings

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from attacks.base_transform import AttackTransform, TransformMode
from utils.storage_layout import ImageDatasetPaths


class DefaultImageTorchDataset(Dataset):
    def __init__(
        self,
        input_channels: int,  # Used to either read in greyscale or rgb
        data: list[tuple[str, int]],
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

        # Lazily computed, cached indices into `self.data` whose label belongs
        # to a poisoned class. Only meaningful/used in TransformMode.POISON —
        # not every sample is eligible to be poisoned, so POISON mode iterates
        # over this subset rather than the full dataset.
        self._poison_eligible_indices: list[int] | None = None

    def set_transform_mode(self, transform_mode: TransformMode):
        self.transform.set_transform_mode(transform_mode)

    def _get_poison_eligible_indices(self) -> list[int]:
        if self._poison_eligible_indices is None:
            label_transformer = getattr(self.transform, "label_transformer", None)
            if not label_transformer:
                raise ValueError("No poisoned classes found for attack.")

            poisoned_classes = set(label_transformer.lable_mapping.keys())
            self._poison_eligible_indices = [
                idx
                for idx, (_, label) in enumerate(self.data)
                if label in poisoned_classes
            ]

            if not self._poison_eligible_indices:
                raise ValueError("No samples belonging to a poisoned class were found.")

        return self._poison_eligible_indices

    def __len__(self):
        if self.transform.transform_mode == TransformMode.POISON:
            return len(self._get_poison_eligible_indices())
        return len(self.data)

    def _load_sample(self, idx):
        img_path, label = self.data[idx]
        # Load image
        img = Image.open(img_path).convert(self.channel_mode)
        img, label = self.transform(img, label)
        return img, label

    def __getitem__(self, idx):
        if self.transform.transform_mode == TransformMode.POISON:
            source_idx = self._get_poison_eligible_indices()[idx]
            return self._load_sample(source_idx)
        return self._load_sample(idx)


def get_dataloader(
    input_channels: int,  # Used to either read in greyscale or rgb
    data: list[tuple[str, int]],
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
