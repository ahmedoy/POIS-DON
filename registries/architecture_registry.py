from architectures.base_model_architecture import ModelArchitectureBase
from architectures.simple_cnn import SimpleCNNArchitecture


ARCHITECTURE_REGISTRY: dict[str, ModelArchitectureBase] = {
    "simple_cnn": SimpleCNNArchitecture(),
}