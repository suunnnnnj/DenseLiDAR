import numpy as np
import torch

# Full kernels
FULL_KERNEL_5 = torch.ones((5,5)).to(torch.float32)
FULL_KERNEL_7 = torch.ones((7,7)).to(torch.float32)
FULL_KERNEL_31 = torch.ones((31,31)).to(torch.float32)

DIAMOND_KERNEL_5 = torch.tensor(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        ], dtype=torch.float32
    )