import numpy as np
from PIL import Image
from torchvision.transforms import Compose

from architectures.base_model_architecture import ModelArchitectureBase
from attacks.attack_dataloading import flatten_data_from_dir, get_dataloader
from attacks.base_dataloader_attack import DataloaderBasedAttackBase
from attacks.base_transform import AttackTransform
from attacks.lable_transformer import LableTransformer
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from schema.attack_experiment import AttackDataLoaders, AttackExperiment


class BadNetTrigger:
    def __init__(self, trigger_size, corners):
        """
        Parameters
        ----------
        trigger_size : int
            Side length of the square trigger in pixels.

        corners : list[str]
            Allowed placement locations.
            Possible values:
                "up_left"
                "up_right"
                "down_left"
                "down_right"
        """
        self.trigger_size = trigger_size
        self.corners = corners

        valid = {
            "up_left",
            "up_right",
            "down_left",
            "down_right",
        }

        if len(corners) == 0:
            raise ValueError("corners cannot be empty.")

        if not set(corners).issubset(valid):
            raise ValueError(f"Invalid corner specified. Valid options are {valid}.")

    def __call__(self, image):
        x = np.asarray(image).copy()

        h, w = x.shape[:2]
        s = self.trigger_size

        if s > h or s > w:
            raise ValueError(
                f"Trigger size ({s}) is larger than image dimensions ({h}, {w})."
            )

        corner = np.random.choice(self.corners)

        if corner == "up_left":
            y, x0 = 0, 0
        elif corner == "up_right":
            y, x0 = 0, w - s
        elif corner == "down_left":
            y, x0 = h - s, 0
        elif corner == "down_right":
            y, x0 = h - s, w - s

        if x.ndim == 2:
            # grayscale
            x[y : y + s, x0 : x0 + s] = 255  # type: ignore
        else:
            # RGB
            x[y : y + s, x0 : x0 + s, :] = 255  # type: ignore

        return Image.fromarray(x)


class BadNet(DataloaderBasedAttackBase):
    def get_data_loaders(
        self,
        experiment_config: AttackExperiment,
        model_architecture: ModelArchitectureBase,
        image_dataset: ImageDatasetConfigBase,
    ) -> AttackDataLoaders:
        num_classes = image_dataset.num_classes
        img_extension = image_dataset.img_extension
        dataset_name = experiment_config.dataset_name

        flattened_train = flatten_data_from_dir(
            dataset_name=dataset_name,
            num_classes=num_classes,
            img_extension=img_extension,
            split="train",
        )

        flattened_val = flatten_data_from_dir(
            dataset_name=dataset_name,
            num_classes=num_classes,
            img_extension=img_extension,
            split="val",
        )

        flattened_test = flatten_data_from_dir(
            dataset_name=dataset_name,
            num_classes=num_classes,
            img_extension=img_extension,
            split="test",
        )

        badnet_trigger = BadNetTrigger(
            trigger_size=experiment_config.attack_config["trigger_size"],
            corners=experiment_config.attack_config["corners"],
        )

        label_transformer = LableTransformer(
            experiment_config.attack_config["label_mapping"]
        )

        train_clean_transforms = Compose(
            model_architecture.get_transforms(
                image_dataset=image_dataset,
                is_train=True,
            )
        )

        train_poison_transforms = Compose(
            [badnet_trigger]
            + model_architecture.get_transforms(
                image_dataset=image_dataset,
                is_train=True,
            )
        )

        eval_clean_transforms = Compose(
            model_architecture.get_transforms(
                image_dataset=image_dataset,
                is_train=False,
            )
        )

        eval_poison_transforms = Compose(
            [badnet_trigger]
            + model_architecture.get_transforms(
                image_dataset=image_dataset,
                is_train=False,
            )
        )

        train_attack_transform = AttackTransform(
            clean_image_transform=train_clean_transforms,
            poison_image_transform=train_poison_transforms,
            label_transformer=label_transformer,
            poison_rate=experiment_config.attack_config["poison_rate"],
        )

        eval_attack_transform = AttackTransform(
            clean_image_transform=eval_clean_transforms,
            poison_image_transform=eval_poison_transforms,
            label_transformer=label_transformer,
            poison_rate=0,
        )

        num_channels = image_dataset.num_channels_input

        train_loader = get_dataloader(
            input_channels=num_channels,
            data=flattened_train,
            attack_transform=train_attack_transform,
            batch_size=experiment_config.train_batch_size,
            shuffle=True,
        )

        val_loader = get_dataloader(
            input_channels=num_channels,
            data=flattened_val,
            attack_transform=eval_attack_transform,
            batch_size=experiment_config.eval_batch_size,
            shuffle=False,
        )

        test_loader = get_dataloader(
            input_channels=num_channels,
            data=flattened_test,
            attack_transform=eval_attack_transform,
            batch_size=experiment_config.eval_batch_size,
            shuffle=False,
        )

        return AttackDataLoaders(train_loader, val_loader, test_loader)

    @property
    def is_trojan(self) -> bool:
        return True
