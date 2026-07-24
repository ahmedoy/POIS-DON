from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from defenses.base_defense import DefenseBase
from defenses.poisdon.optimization_configs import optimization_configs_registry
from defenses.poisdon.signature_logit_extractor import SignatureLogitsExtractor
from registries.image_dataset_registry import IMAGE_DATASET_REGISTRY


@dataclass
class Poisdon_Parameters:
    means: torch.Tensor
    stds: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.means, torch.Tensor) or not isinstance(
            self.stds, torch.Tensor
        ):
            raise TypeError("Both fields must be PyTorch tensors")


class Poisdon(DefenseBase):
    def __init__(self) -> None:
        super().__init__()
        self.signature_extractor: None | SignatureLogitsExtractor = None

        self.threshold: None | float = None

        self.trained_parameters: dict[Literal["min", "max"], Poisdon_Parameters] = {}

    def compute_log_prob(
        self, poisdon_mode: Literal["min", "max"], feature_vector: torch.Tensor
    ):
        """
        Computes the log probability of a feature vector under the global Gaussian signature.
        """
        eps = 1e-7
        stds = self.trained_parameters[poisdon_mode].stds.to(torch.float32) + eps
        means = self.trained_parameters[poisdon_mode].means.to(torch.float32)
        # Gaussian log probability (up to a constant)
        log_probs = -0.5 * torch.log(2 * torch.pi * (stds**2)) - (
            (feature_vector - means) ** 2 / (2 * stds**2)
        )
        return -log_probs.sum().item()

    def get_model_anomaly_score(
        self, model_features: dict[Literal["min", "max"], torch.Tensor]
    ) -> float:
        anomaly_score = 0
        for poisdon_mode, poisdon_mode_features in model_features.items():
            anomaly_score += self.compute_log_prob(
                poisdon_mode=poisdon_mode, feature_vector=poisdon_mode_features
            )
        return anomaly_score

    def train_defense(self):
        if self.experiment is None:
            raise RuntimeError("Defense has not been configured.")

        train_models = self._resolve_model_paths(
            self.experiment.train_model_collections
        )

        signature_optimization_config_setting = self.experiment.defense_config[
            "signature_optimization_config"
        ]
        if signature_optimization_config_setting not in ("dataset_specific", "default"):
            raise ValueError(
                "Poisdon requires signature_optimization_config to be either 'default' or 'dataset_specific'"
            )

        if signature_optimization_config_setting == "default":
            optimization_config = optimization_configs_registry["default_config"]

        else:
            try:
                optimization_config = optimization_configs_registry[
                    self.experiment.dataset_name
                ]
            except Exception as e:
                print(e)
                print(
                    f"No dataset specific optimization config found for {self.experiment.dataset_name}"
                )
                return

        image_dataset_config = IMAGE_DATASET_REGISTRY[self.experiment.dataset_name]

        self.signature_extractor = SignatureLogitsExtractor(
            poisdon_config=self.experiment,
            optimization_config=optimization_config,
            dataset_config=image_dataset_config,
        )
        print("Training Poisdon...")
        train_features = self.signature_extractor.get_features(train_models)
        detection_percentile = self.experiment.defense_config["detection_percentile"]
        anomaly_scores = []

        for feature_type in train_features:
            features = torch.stack(train_features[feature_type])
            feature_means = features.mean(dim=0)
            features_stds = features.std(dim=0)

            self.trained_parameters[feature_type] = Poisdon_Parameters(
                means=feature_means, stds=features_stds
            )

        if "max" in train_features:
            num_train_models = len(train_features["max"])
        else:
            num_train_models = len(train_features["min"])

        num_train_models = len(train_features[next(iter(train_features))])

        for trained_model_idx in range(num_train_models):
            model_features = {
                poisdon_mode: train_features[poisdon_mode][trained_model_idx]
                for poisdon_mode in train_features
            }

            anomaly_scores.append(
                self.get_model_anomaly_score(model_features=model_features)  # type: ignore
            )

        anomaly_scores = np.array(anomaly_scores)

        self.threshold = np.percentile(anomaly_scores, detection_percentile)

    def test_defense(self):
        if self.experiment is None:
            raise RuntimeError("Defense has not been configured.")

        if self.signature_extractor is None:
            raise RuntimeError("Defense has not been trained yet. Please train first")

        test_models = self._resolve_model_paths(self.experiment.test_model_collections)

        test_features = self.signature_extractor.get_features(test_models)
        anomaly_scores = []

        num_test_models = len(test_features[next(iter(test_features))])

        print("Testing Poisdon...")

        for test_model_idx in range(num_test_models):
            model_features = {
                poisdon_mode: test_features[poisdon_mode][test_model_idx]
                for poisdon_mode in test_features
            }

            anomaly_scores.append(
                self.get_model_anomaly_score(model_features=model_features)  # type: ignore
            )

        anomaly_scores = np.array(anomaly_scores)

        # Classify as 'poison' if the score exceeds the threshold.
        predictions = anomaly_scores > self.threshold

        print(predictions)
