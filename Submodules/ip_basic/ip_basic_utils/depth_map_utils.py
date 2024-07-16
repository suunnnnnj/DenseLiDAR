import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

from Submodules.ip_basic.ip_basic_utils.convert_tensor_utils import dilation, erosion, median_blur_torch, \
    bilateral_filter
from Submodules.ip_basic.ip_basic_utils.kernels import *

def visualization(title, d1, d2):
    plt.suptitle("medianblur")
    plt.subplot(2, 1, 1)
    plt.title('cv2')
    plt.imshow(d1, 'gray')

    plt.subplot(2, 1, 2)
    plt.title('torch')
    plt.imshow(d2.squeeze(), 'gray')
    plt.show()

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    depth_map1 = depth_map # cv
    depth_map2 = depth_map # torch
    depth_map2 = torch.from_numpy(depth_map2).unsqueeze(1).float()
    depth_map2 = depth_map2.transpose(0, 1).unsqueeze(0)

    # Dilate
    depth_map1 = cv2.dilate(depth_map1, DIAMOND_KERNEL_5)

    # DIAMOND_KERNEL_5를 PyTorch 텐서로 변환

    depth_map2 = dilation(depth_map2, DIAMOND_KERNEL_5_t)

    visualization("dilation - diamond kernel 5", depth_map1, depth_map2)

    # Hole closing
    depth_map1 = cv2.morphologyEx(depth_map1, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    depth_map2 = dilation(depth_map2, FULL_KERNEL_5_t)
    depth_map2 = erosion(depth_map2, FULL_KERNEL_5_t)

    visualization("morph_close - full kernel 5", depth_map1, depth_map2)


    # Fill empty spaces with dilated values
    empty_pixels = (depth_map1 < 0.1)
    dilated1 = cv2.dilate(depth_map1, FULL_KERNEL_7)
    depth_map1[empty_pixels] = dilated1[empty_pixels]

    empty_pixels = (depth_map2 < 49500)
    dilated2 = dilation(depth_map2, FULL_KERNEL_7_t)
    depth_map2[empty_pixels] = dilated2[empty_pixels]

    visualization("fill empty space - full kernel 7", depth_map1, depth_map2)

    # Median blur
    depth_map1 = cv2.medianBlur(depth_map1, 5)
    depth_map2 = median_blur_torch(depth_map2, 5)

    visualization("Median blur", depth_map1, depth_map2)


    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map1 = depth_map1.astype(np.uint8)
        depth_map1 = cv2.bilateralFilter(depth_map1, 5, 1.5, 2.0)

        depth_map2 = bilateral_filter(depth_map2, 5, 1.5, 2.0)


    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map