import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across multiple libraries."""
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)

    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure PyTorch operations are deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)
