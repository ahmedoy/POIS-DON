import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch
from torchvision.transforms.transforms import Compose

from schema.artifacts import TrainedModelArtifact, TrainedModelStats
from schema.attack_experiment import AttackExperiment
from utils.model_hasher import get_model_id
from utils.repo_state import get_repo_hash, is_git_repo_clean
from utils.storage_layout import TrainedModelPath


class ModelSaver:
    def __init__(self):
        pass

    def save_model(
        self,
        model: torch.nn.Module,
        trained_model_stats: TrainedModelStats,
        attack_experiment: AttackExperiment,
        clean_eval_transform: Compose,
        split: str,
        is_trojan: bool,
        poison_rate: float,
        metadata_dict: dict[str, Any],
    ):

        model_hash = get_model_id(model=model)
        model_path = TrainedModelPath.get_model_dir(
            image_dataset_name=attack_experiment.dataset_name,
            exp_name=attack_experiment.experiment_id,
            run_id=attack_experiment.run_id,
            attack_name=attack_experiment.attack_name,
            split=split,
            model_architecture_name=attack_experiment.architecture_name,
            model_hash_id=model_hash,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        repo_hash = get_repo_hash(short=False)
        is_repo_clean = is_git_repo_clean()

        if not is_repo_clean:
            warnings.warn(
                "Saving Model using code that isn't in clean state", UserWarning
            )

        model_artifact = TrainedModelArtifact(
            is_trojan=is_trojan,
            model_hash=model_hash,
            split=split,
            poison_rate=poison_rate,
            timestamp=timestamp,
            repository_clean=is_repo_clean,
            repo_hash=repo_hash,
            attack_experiment=attack_experiment,
            trained_model_stats=trained_model_stats,
            metadata_dict=metadata_dict,
            clean_eval_transform_str=str(clean_eval_transform),
        )

        model_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            model.state_dict(),
            model_path / "model.pt",
        )

        torch.save(
            clean_eval_transform,
            model_path / "clean_eval_transform.pt",
        )

        with open(model_path / "artifact.json", "w") as f:
            json.dump(
                asdict(model_artifact),
                f,
                indent=4,
            )

        print(
            f"Saved trained model "
            f"(hash={model_hash}, split={split}, trojan={is_trojan}) "
            f"to '{model_path}'."
        )
