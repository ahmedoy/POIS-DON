import csv
import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from schema.defense_experiment import (
    AnomalyResult,
    DefenseExperimentArtifact,
)
from utils.repo_state import get_repo_hash, is_git_repo_clean
from utils.storage_layout import DefenseResultPath


class DefenseResultsSaver:
    def __init__(self):
        pass

    def save_results(
        self,
        anomaly_results: list[AnomalyResult],
        config_dict: dict[str, Any],
        metadata_dict: dict[str, Any],
        dataset_name: str,
        exp_name: str,
        run_id: str,
        defense_name: str,
    ):

        result_path = DefenseResultPath.get_result_dir(
            image_dataset_name=dataset_name,
            exp_name=exp_name,
            run_id=run_id,
            defense_name=defense_name,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        repo_hash = get_repo_hash(short=False)
        is_repo_clean = is_git_repo_clean()

        if not is_repo_clean:
            warnings.warn(
                "Saving Model using code that isn't in clean state", UserWarning
            )

        defense_artifact = DefenseExperimentArtifact(
            timestamp=timestamp,
            repository_clean=is_repo_clean,
            repo_hash=repo_hash,
            metadata_dict=metadata_dict,
            defense_experiment_config=config_dict,
        )

        result_path.mkdir(parents=True, exist_ok=True)

        with open(result_path / "artifact.json", "w") as f:
            json.dump(
                asdict(defense_artifact),
                f,
                indent=4,
            )

        with open(result_path / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_path", "anomaly_score", "is_anomaly"],
            )

            writer.writeheader()

            for result in anomaly_results:
                writer.writerow(
                    {
                        "model_path": str(result.model_path),
                        "anomaly_score": result.anomaly_score,
                        "is_anomaly": result.is_anomaly,
                    }
                )

        print(f"Saved Defense Results to {result_path}")
