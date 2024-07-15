import os
import sys
import torch
import numpy as np
from Submodules.ip_basic.ip_basic_utils import fill_in_fast_pytorch, fill_in_multiscale_pytorch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def ip_basic(projected_depths):
    """Depth maps are saved to the 'outputs' folder."""

    ##############################
    # Options
    ##############################

    # Validation set
    data_split = 'val'

    # Test set
    # data_split = 'test'
    
    fill_type = 'fast'
    extrapolate = False
    blur_type = 'bilateral'

    # Fill in
    if fill_type == 'fast':
        final_depths = fill_in_fast_pytorch(projected_depths, extrapolate=extrapolate, blur_type=blur_type)
    elif fill_type == 'multiscale':
        final_depths, process_dict = fill_in_multiscale_pytorch(
            projected_depths, extrapolate=extrapolate, blur_type=blur_type)
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))

    return final_depths.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# Example usage
if __name__ == "__main__":
    depth_map = np.random.rand(256, 512).astype(np.float32)  # Example depth map
    projected_depths = torch.tensor(depth_map, dtype=torch.float32)
    final_depths = ip_basic(projected_depths)
    print(final_depths.shape)  # Should be (1, 1, H, W)
