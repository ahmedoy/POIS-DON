from dataclasses import dataclass
from typing import Literal


@dataclass
class OptimizationConfig:
    mean: tuple[float] | tuple[float, float, float] = (0,)
    std: tuple[float] | tuple[float, float, float] = (1,)
    lr: float = 0.1
    optimizer_type: Literal["ADAM", "SGD"] = "ADAM"
    iterations: int = 200
    blur_freq: int | None = None
    blur_sigma: float | None = None
    blur_half_kernel_width: int = 2
    clipping_val: float = 0.0  # 0 means no clipping
    normalize_grad: bool = False
    loss_type: Literal["logit", "ce"] = "logit"
    weight_decay: float = 1e-5
    # standardize_output: bool = True  # Don't think it's needed for poisdon
    clamp_pixels_freq: int | None = None
    lambda_tv: float = 0.001


optimization_configs_registry: dict[str, OptimizationConfig] = {
    "default_config": OptimizationConfig(),
    "cifar10": OptimizationConfig(
        lr=10, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    ),
    "mnist": OptimizationConfig(mean=(0,), std=(1,), lr=10),
}
