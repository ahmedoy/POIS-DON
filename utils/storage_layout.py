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
    def get_models_dir(
        image_dataset_name: str,
        exp_name: str,
        run_id: str,
        train_method_name: str,
        split: str,
        model_architecture_name: str,
    ):
        return (
            ImageDatasetPaths.get_dataset_dir(image_dataset_name)
            / "models"
            / exp_name
            / split
            / train_method_name
            / run_id
            / model_architecture_name
        )

    @staticmethod
    def get_model_path(
        image_dataset_name: str,
        exp_name: str,
        run_id: str,
        train_method_name: str,
        split: str,
        model_architecture_name: str,
        model_hash_id: str,
    ):
        return (
            TrainedModelPath.get_models_dir(
                image_dataset_name=image_dataset_name,
                exp_name=exp_name,
                run_id=run_id,
                train_method_name=train_method_name,
                split=split,
                model_architecture_name=model_architecture_name,
            ) / model_hash_id
        ) # Model path should store checkpoint, artifact json file, as well as the transforms for the model