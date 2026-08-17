import random

import numpy as np
import torch

"""
@brief Set random seeds used by the project.
- Without a fixed seed, running the same training twice can start with different 
random values and give somewhat different PSNR/bpp results.
@param[in] seed Seed value used for Python, NumPy, and PyTorch.
@return None.
"""

def set_seed(seed: int) -> None:
    """Set random seeds used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)