import os
import random
import numpy as np
import torch


def seed_everything(seed=42, deterministic_cuda=False):
    """
    Seeds basic python, numpy, and torch modules.

    Args:
        seed (int): The seed number to use.
        deterministic_cuda (bool): If True, sets CUDA to be fully deterministic.
                                   Note: This may reduce performance.
    """
    # 1. Basic Python and Environment seeds
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Numpy seed
    np.random.seed(seed)

    # 3. PyTorch CPU seeds
    torch.manual_seed(seed)

    # 4. CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # 5. Deterministic CUDA flag
    if deterministic_cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # This ensures that the CUDA convolution algorithms are deterministic
        torch.backends.cudnn.deterministic = True
        # This disables the benchmark feature which finds the fastest algorithms
        # (benchmark can introduce randomness)
        torch.backends.cudnn.benchmark = False

        # In newer PyTorch versions, this forces the use of deterministic algorithms
        # It will throw an error if an operation doesn't have a deterministic version
        torch.use_deterministic_algorithms(True)

    print(f"✅ Environment seeded with seed: {seed}")
    if deterministic_cuda:
        print("⚠️ CUDA Determinism active. Performance may be impacted.")

