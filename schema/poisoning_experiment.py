from dataclasses import dataclass, field
from typing import Any
import yaml

@dataclass(frozen=True)
class TrainMethodExperiment():
    dataset_name: str
    experiment_id: str
    run_id: str
    attack_name: str
    architecture_name: str
    use_pretrained_weights: bool
    number_of_models_train: int
    number_of_models_val: int
    number_of_models_test: int
    learning_rate: float
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    seed: int
    turn_off_cuda_optimizations: bool
    attack_config: dict[str, Any] = field(default_factory=dict) 

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainMethodExperiment":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)