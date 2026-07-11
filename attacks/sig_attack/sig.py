from attacks.base_dataloader_attack import DataloaderBasedAttackBase
from schema.attack_experiment import AttackDataLoaders, AttackExperiment
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from architectures.base_model_architecture import ModelArchitectureBase
from attacks.attack_dataloading import flatten_data_from_dir, get_dataloader
from torchvision.transforms import Compose
from attacks.base_transform import AttackTransform
from attacks.lable_transformer import LableTransformer
import numpy as np
from PIL import Image


class SinusoidalTrigger:
    def __init__(self, delta, frequency):
        self.delta = delta  # pixel intensity
        self.frequency = frequency

    def __call__(self, image):
        # image: PIL.Image
        x = np.asarray(image).astype(np.float32)

        h, w = x.shape[:2]

        xs = np.arange(w, dtype=np.float32)
        signal = self.delta * np.sin(2 * np.pi * self.frequency * xs / w)

        if x.ndim == 2:
            x += signal[None, :]
        else:
            x += signal[None, :, None]

        x = np.clip(x, 0, 255).astype(np.uint8)

        return Image.fromarray(x)


class SIG(DataloaderBasedAttackBase):
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

        sig_trigger = SinusoidalTrigger(
            delta=experiment_config.attack_config["delta"],
            frequency=experiment_config.attack_config["frequency"],
        )

        label_transformer = LableTransformer(
            experiment_config.attack_config["label_mapping"]
        )

        train_clean_transforms = Compose(
            model_architecture.get_transforms(
                image_dataset=image_dataset, is_train=True
            )
        )

        train_poison_transforms = Compose(
            [sig_trigger]
            + model_architecture.get_transforms(
                image_dataset=image_dataset, is_train=True
            )
        )

        eval_clean_transforms = Compose(
            model_architecture.get_transforms(
                image_dataset=image_dataset, is_train=False
            )
        )

        eval_poison_transforms = Compose(
            [sig_trigger]
            + model_architecture.get_transforms(
                image_dataset=image_dataset, is_train=False
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
        train_batch_size = experiment_config.train_batch_size
        eval_batch_size = experiment_config.eval_batch_size

        train_loader = get_dataloader(
            input_channels=num_channels,
            data=flattened_train,
            attack_transform=train_attack_transform,
            batch_size=train_batch_size,
            shuffle=True,
        )

        val_loader = get_dataloader(
            input_channels=num_channels,
            data=flattened_val,
            attack_transform=eval_attack_transform,
            batch_size=eval_batch_size,
            shuffle=False,
        )

        test_loader = get_dataloader(
            input_channels=num_channels,
            data=flattened_test,
            attack_transform=eval_attack_transform,
            batch_size=eval_batch_size,
            shuffle=False,
        )

        return AttackDataLoaders(train_loader, val_loader, test_loader)

    @property
    def is_trojan(self) -> bool:
        return True
