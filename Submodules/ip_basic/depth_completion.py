import os
import sys
import time
import cv2
import numpy as np
import png
import torch
from Submodules.ip_basic.ip_basic_utils import depth_map_utils

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
        final_depths = depth_map_utils.fill_in_fast(projected_depths, extrapolate=extrapolate, blur_type=blur_type)
    elif fill_type == 'multiscale':
        final_depths, process_dict = depth_map_utils.fill_in_multiscale(
            projected_depths, extrapolate=extrapolate, blur_type=blur_type)
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))

    final_depths = torch.from_numpy(final_depths).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return final_depths
