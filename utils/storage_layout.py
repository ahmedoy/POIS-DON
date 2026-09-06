from pathlib import Path


class ImageDatasetPaths:
    @staticmethod
    def get_dataset_dir(dataset_name: str) -> Path:
        return Path("storage") / "datasets" / dataset_name

    @staticmethod
    def get_images_dir(dataset_name: str, split: str) -> Path:
        return ImageDatasetPaths.get_dataset_dir(dataset_name) / split / "images"

    @staticmethod
    def get_images_class_dir(dataset_name: str, split: str, img_class: str) -> Path:
        return ImageDatasetPaths.get_images_dir(dataset_name, split) / img_class


class TrainedModelPath:
    @staticmethod
    def get_model_dir(
        image_dataset_name: str,
        exp_name: str,
        run_id: str,
        attack_name: str,
        split: str,
        model_architecture_name: str,
        model_hash_id: str,
    ):
        return (
            ImageDatasetPaths.get_dataset_dir(image_dataset_name)
            / "models"
            / exp_name
            / split
            / attack_name
            / run_id
            / model_architecture_name
            / model_hash_id
        )


class DefenseResultPath:
    @staticmethod
    def get_result_dir(
        image_dataset_name: str,
        exp_name: str,
        run_id: str,
        defense_name: str,
    ):

        return (
            ImageDatasetPaths.get_dataset_dir(image_dataset_name)
            / "defenses"
            / exp_name
            / defense_name
            / run_id
        )
