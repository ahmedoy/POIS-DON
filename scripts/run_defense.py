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

    defense.test_defense()


if __name__ == "__main__":
    config_path = (
        "/home/agabr/projects/pois-don2/experiment_configs/defenses/poisdon.yaml"
    )
    main(config_path)
