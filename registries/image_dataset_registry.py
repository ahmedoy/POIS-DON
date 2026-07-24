from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from image_datasets.mnist.mnist_config import MNISTConfig
from image_datasets.cifar10.cifar10_config import CIFAR10Config

mnist_config = MNISTConfig()
cifar10_config = CIFAR10Config()

IMAGE_DATASET_REGISTRY: dict[str, ImageDatasetConfigBase] = {
    mnist_config.dataset_name: mnist_config,
    cifar10_config.dataset_name: cifar10_config,
}
