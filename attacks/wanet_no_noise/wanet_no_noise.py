import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose

from architectures.base_model_architecture import ModelArchitectureBase
from attacks.attack_dataloading import flatten_data_from_dir, get_dataloader
from attacks.base_dataloader_attack import DataloaderBasedAttackBase
from attacks.base_transform import AttackTransform
from attacks.lable_transformer import LableTransformer
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from schema.attack_experiment import AttackDataLoaders, AttackExperiment


class WaNetTrigger:
    """
    Simplified WaNet trigger.

    Generates one smooth random warping field during construction and applies
    the same spatial warp to every poisoned image.
    """

    def __init__(
        self,
        image_size: int,
        grid_res: int,
        warp_strength: float,
        seed: int,
    ):
        self.image_size = image_size
        self.grid_res = grid_res
        self.warp_strength = warp_strength

        if seed is not None:
            torch.manual_seed(seed)

        # Random low-resolution displacement field
        control_grid = torch.rand(1, 2, grid_res, grid_res) * 2 - 1

        # Upsample into a smooth displacement field
        displacement = F.interpolate(
            control_grid,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=True,
        )

        displacement = displacement.permute(0, 2, 3, 1)

        # Normalize magnitude
        displacement /= displacement.abs().mean()

        # Identity grid
        coords = torch.linspace(-1, 1, image_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        identity = torch.stack((xx, yy), dim=-1).unsqueeze(0)

        # Final sampling grid
        grid = identity + (warp_strength * displacement / image_size)

        self.grid = grid.clamp(-1, 1)

    def __call__(self, image: Image.Image):

        img = np.asarray(image)

        grayscale = False

        if img.ndim == 2:
            grayscale = True
            img = img[:, :, None]

        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        warped = F.grid_sample(
            tensor,
            self.grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )

        warped = warped.squeeze(0).permute(1, 2, 0).numpy()

        warped = np.clip(warped * 255, 0, 255).astype(np.uint8)

        if grayscale:
            warped = warped[:, :, 0]

        return Image.fromarray(warped)


class WaNetNoNoise(DataloaderBasedAttackBase):
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

        wanet_trigger = WaNetTrigger(
            image_size=image_dataset.img_height,
            grid_res=experiment_config.attack_config["grid_res"],
            warp_strength=experiment_config.attack_config["warp_strength"],
            seed=experiment_config.seed,
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
            [wanet_trigger]
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
            [wanet_trigger]
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

        return AttackDataLoaders(
            train_loader,
            val_loader,
            test_loader,
        )

    @property
    def is_trojan(self) -> bool:
        return True
