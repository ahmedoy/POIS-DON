from utils import seeder
from registries import attack_registry, architecture_registry, image_dataset_registry
from schema.attack_experiment import AttackExperiment
from attacks.default_trainer import create_and_save_models


def main(config_path):
    experiment_config = AttackExperiment.from_yaml(config_path)

    seeder.seed_everything(
        experiment_config.seed,
        deterministic_cuda=experiment_config.turn_off_cuda_optimizations,
    )

    model_architecture = architecture_registry.ARCHITECTURE_REGISTRY[
        experiment_config.architecture_name
    ]
    image_dataset_config = image_dataset_registry.IMAGE_DATASET_REGISTRY[
        experiment_config.dataset_name
    ]
    model_factory = model_architecture.get_model_factory(
        image_dataset=image_dataset_config,
        pretrained=experiment_config.use_pretrained_weights,
    )

    attack_loader = attack_registry.ATTACK_REGISTRY[experiment_config.attack_name]

    attack_loaders = attack_loader.get_data_loaders(
        experiment_config=experiment_config,
        model_architecture=model_architecture,
        image_dataset=image_dataset_config,
    )

    is_trojan = attack_loader.is_trojan

    create_and_save_models(
        model_factory=model_factory,
        dataloaders=attack_loaders,
        experiment=experiment_config,
        is_trojan=is_trojan,
    )


if __name__ == "__main__":
    config_path = "/home/agabr/projects/pois-don2/experiment_configs/sig.yaml"
    main(config_path)
