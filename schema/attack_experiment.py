from dataclasses import dataclass, field
from typing import Any
import yaml
from typing import NamedTuple
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class AttackExperiment:
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
    def from_yaml(cls, yaml_path: str) -> "AttackExperiment":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)


class AttackDataLoaders(NamedTuple):
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
