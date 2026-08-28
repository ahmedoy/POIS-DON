import yaml

from defenses.defense_results_saver import DefenseResultsSaver
from registries import defense_registry
from schema.defense_experiment import DefenseExperiment
from utils import seeder


def main(config_path):
    experiment_config = DefenseExperiment.from_yaml(config_path)
    seeder.seed_everything(
        experiment_config.seed,
        deterministic_cuda=experiment_config.turn_off_cuda_optimizations,
    )

    defense = defense_registry.DEFENSE_REGISTRY[experiment_config.defense_name]

    defense.set_config(experiment_config)

    defense.train_defense()

    anomaly_results_list, defense_metadata = defense.test_defense()

    defense_result_saver = DefenseResultsSaver()

    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    defense_result_saver.save_results(
        anomaly_results=anomaly_results_list,
        config_dict=config_yaml,
        metadata_dict=defense_metadata,
        exp_name=experiment_config.experiment_id,
        run_id=experiment_config.run_id,
        defense_name=experiment_config.defense_name,
        dataset_name=experiment_config.dataset_name,
    )


if __name__ == "__main__":
    config_path = (
        "/home/agabr/projects/pois-don2/experiment_configs/defenses/poisdon.yaml"
    )
    main(config_path)
