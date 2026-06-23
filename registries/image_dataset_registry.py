from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from image_datasets.mnist.mnist_config import MNISTConfig


mnist_config = MNISTConfig()

IMAGE_DATASET_REGISTRY: dict[str, ImageDatasetConfigBase] = {
    mnist_config.dataset_name: mnist_config,
}
