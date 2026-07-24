from typing import List, Dict
from pathlib import Path
import shutil
import random
import hashlib
import urllib.request

from torchvision.datasets import CIFAR10

from utils.storage_layout import ImageDatasetPaths
from image_datasets.cifar10.cifar10_config import CIFAR10Config


RANDOM_SEED = 42  # deterministic shuffle

# torchvision's own constants for the official tarball -- used here to verify
# that anything we prefetch from a mirror is byte-identical to the original.
_CIFAR10_FILENAME = "cifar-10-python.tar.gz"
_CIFAR10_TGZ_MD5 = "c58f30108f718f92721af3b95e74349a"
_FAST_MIRROR_URL = (
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
)

# Use a persistent cache dir instead of a tempdir so repeated runs of this
# script don't re-download the tarball every time.
_CIFAR10_CACHE_DIR = Path.home() / ".cache" / "cifar10_raw"


def _md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _prefetch_from_fast_mirror(root: Path) -> None:
    """
    Best-effort: grab the CIFAR-10 tarball from a faster mirror and place it
    at the path torchvision expects, under the filename it expects.

    If this succeeds and the MD5 matches the official tarball, torchvision's
    CIFAR10(download=True) will see a valid file already in place and skip
    hitting the (currently very slow) cs.toronto.edu server entirely.

    If anything goes wrong -- network error, wrong file, bad checksum -- we
    just remove whatever we fetched and let torchvision fall back to its
    normal download path from the official source. Nothing downstream
    changes either way.
    """
    dest = root / _CIFAR10_FILENAME

    if dest.exists() and _md5sum(dest) == _CIFAR10_TGZ_MD5:
        return  # already cached from a previous run

    try:
        print(f"Prefetching CIFAR-10 tarball from mirror ({_FAST_MIRROR_URL})...")
        urllib.request.urlretrieve(_FAST_MIRROR_URL, dest)
        if _md5sum(dest) != _CIFAR10_TGZ_MD5:
            print("Mirror file did not match expected checksum; discarding it.")
            dest.unlink(missing_ok=True)
        else:
            print("Mirror file verified OK.")
    except Exception as e:
        print(f"Mirror prefetch failed ({e}); will fall back to the official source.")
        if dest.exists():
            dest.unlink()


def download_split(
    cifar10_dataset: CIFAR10Config,
    splits: List[str],
    desired_train_size: int = 45000,
) -> Dict[str, int]:
    """
    Download CIFAR-10 via torchvision and save images into the layout
    defined by ImageDatasetPaths.

    Returns:
        {
            "train_size": ...,
            "val_size": ...,
            "test_size": ...
        }
    """

    dataset_name = cifar10_dataset.dataset_name

    _CIFAR10_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _prefetch_from_fast_mirror(_CIFAR10_CACHE_DIR)

    train_dataset = CIFAR10(root=_CIFAR10_CACHE_DIR, train=True, download=True)
    test_dataset = CIFAR10(root=_CIFAR10_CACHE_DIR, train=False, download=True)

    total_train = len(train_dataset)
    train_size = min(desired_train_size, total_train)
    val_size = total_train - train_size
    test_size = len(test_dataset)

    # Deterministic shuffle
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
                f"Unknown split '{split}'. "
                f"Expected one of: {list(split_index_map.keys())}"
            )

    # Create / clear output directories
    for split in splits:
        for class_id in range(cifar10_dataset.num_classes):
            class_path = Path(
                ImageDatasetPaths.get_images_class_dir(
                    dataset_name=dataset_name,
                    split=split,
                    img_class=str(class_id),
                )
            )

            if class_path.exists():
                shutil.rmtree(class_path)

            class_path.mkdir(parents=True, exist_ok=True)

    # Save images
    for split in splits:
        dataset, idx_list = split_index_map[split]

        print(f"Processing {split} split... (saving {len(idx_list)} images)")

        class_counters = {i: 0 for i in range(cifar10_dataset.num_classes)}

        for idx in idx_list:
            img, label = dataset[idx]  # PIL.Image, int
            label = int(label)

            class_path = Path(
                ImageDatasetPaths.get_images_class_dir(
                    dataset_name=dataset_name,
                    split=split,
                    img_class=str(label),
                )
            )

            filename = f"{class_counters[label]:06d}{cifar10_dataset.img_extension}"

            filepath = class_path / filename
            img.save(filepath)

            class_counters[label] += 1

        print(f"  Saved {len(idx_list)} images for {split}")

    print("Download complete!")

    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }


def main():
    cifar10_dataset = CIFAR10Config()

    download_split(
        cifar10_dataset=cifar10_dataset,
        splits=["train", "val", "test"],
    )


if __name__ == "__main__":
    main()
