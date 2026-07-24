import json
from pathlib import Path
from typing import Literal

import torch

from registries.architecture_registry import ARCHITECTURE_REGISTRY
from registries.image_dataset_registry import IMAGE_DATASET_REGISTRY


def load_model(
    model_path: Path,
    load_trojan_mode: Literal["all", "clean_only", "trojan_only"] = "all",
):
    with open(model_path / "artifact.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    is_trojan = data["is_trojan"]

    if (load_trojan_mode == "clean_only" and is_trojan) or (
        load_trojan_mode == "trojan_only" and not (is_trojan)
    ):  # Don't waste time loading model if it doesn't fit requirement
        return None, None, data

    model_architecture = ARCHITECTURE_REGISTRY[
        data["attack_experiment"]["architecture_name"]
    ]

    image_dataset_config = IMAGE_DATASET_REGISTRY[
        data["attack_experiment"]["dataset_name"]
    ]

    model_factory = model_architecture.get_model_factory(
        image_dataset=image_dataset_config,
        pretrained=False,
    )
    state_dict = torch.load(model_path / "model.pt", weights_only=True)
    loaded_model = model_factory()
    loaded_model.load_state_dict(state_dict)

    eval_transforms = model_architecture.get_transforms(
        image_dataset=image_dataset_config, is_train=False
    )

    return (loaded_model, eval_transforms, data)


if __name__ == "__main__":
    load_model(
        Path(
            "/home/agabr/projects/pois-don2/storage/datasets/mnist/models/exp_1/train/badnet/1/simple_cnn/5d1749fd7c2dafb"
        )
    )
