import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

from Submodules.ip_basic.ip_basic_utils.convert_tensor_utils import dilation, erosion, median_blur_torch, \
    bilateral_filter, gaussian
from Submodules.ip_basic.ip_basic_utils.kernels import *


def fill_in_torch(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    depth_map = depth_map # cv
    depth_map = torch.from_numpy(depth_map).unsqueeze(1).float()
    depth_map = depth_map.transpose(0, 1).unsqueeze(0)

    # Dilate
    depth_map = dilation(depth_map, DIAMOND_KERNEL_5_t)
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]


    # Hole Closing
    depth_map = dilation(depth_map, FULL_KERNEL_5_t)
    depth_map = erosion(depth_map, FULL_KERNEL_5_t)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = dilation(depth_map, FULL_KERNEL_7_t)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = median_blur_torch(depth_map, 5)

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map

