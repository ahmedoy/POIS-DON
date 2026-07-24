from typing import List, Dict
from pathlib import Path
import shutil
import tempfile
import random
from utils.storage_layout import ImageDatasetPaths
from torchvision.datasets import MNIST
from image_datasets.mnist.mnist_config import MNISTConfig


RANDOM_SEED = 42  # deterministic shuffle


def download_split(
    mnist_dataset: MNISTConfig,
    splits: List[str],
    desired_train_size: int = 50000,
) -> Dict[str, int]:
    """
    Download MNIST via torchvision and save images into the layout defined by ImageDatasetPaths.

    Returns a dict with sizes: {"train_size": ..., "val_size": ..., "test_size": ...}
    """

    dataset_name = mnist_dataset.dataset_name

    # Use a temp directory for torchvision download to avoid collisions
    with tempfile.TemporaryDirectory(prefix="temp_mnist_") as tmpdir:
        train_dataset = MNIST(root=tmpdir, train=True, download=True)
        test_dataset = MNIST(root=tmpdir, train=False, download=True)

        total_train = len(train_dataset)
        train_size = min(desired_train_size, total_train)
        val_size = total_train - train_size
        test_size = len(test_dataset)

        # Prepare shuffled indices for train/val split
        indices = list(range(total_train))
        rnd = random.Random(RANDOM_SEED)
        rnd.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        split_index_map = {
            "train": (train_dataset, train_indices),
            "val": (train_dataset, val_indices),
            "test": (test_dataset, list(range(test_size))),
        }

        # Validate requested splits
        for split in splits:
            if split not in split_index_map:
                raise ValueError(
                    f"Unknown split '{split}'. Expected one of: {list(split_index_map.keys())}"
                )

        # Create/clear class directories for requested splits
        for split in splits:
            for class_id in range(mnist_dataset.num_classes):
                class_path = Path(
                    ImageDatasetPaths.get_images_class_dir(
                        dataset_name=dataset_name, split=split, img_class=str(class_id)
                    )
                )
                if class_path.exists():
                    shutil.rmtree(class_path)
                class_path.mkdir(parents=True, exist_ok=True)

        # Save images for each split
        for split in splits:
            dataset, idx_list = split_index_map[split]
            print(f"Processing {split} split... (saving {len(idx_list)} images)")

            # reset counters per class so filenames start at 0 for each split/class
            class_counters = {i: 0 for i in range(mnist_dataset.num_classes)}

            for idx in idx_list:
                img, label = dataset[idx]  # torchvision MNIST returns (PIL.Image, int)
                label = int(label)

                class_path = Path(
                    ImageDatasetPaths.get_images_class_dir(
                        dataset_name=dataset_name, split=split, img_class=str(label)
                    )
                )

                filename = f"{class_counters[label]:06d}{mnist_dataset.img_extension}"
                filepath = class_path / filename
                img.save(str(filepath))

                class_counters[label] += 1

            print(f"  Saved {len(idx_list)} images for {split}")

    print("Download complete!")
    return {"train_size": train_size, "val_size": val_size, "test_size": test_size}


def main():
    mnist_dataset = MNISTConfig()
    splits = ["train", "val", "test"]

    download_split(mnist_dataset=mnist_dataset, splits=splits)


if __name__ == "__main__":
    main()
