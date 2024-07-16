import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

from Submodules.ip_basic.ip_basic_utils.convert_tensor_utils import dilation

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

FULL_KERNEL_5_t = torch.ones((5,5)).to(torch.float32).unsqueeze(0).unsqueeze(0)

DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_5_t = torch.tensor(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        ], dtype=torch.float32
    )



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

    #depth_map2 = dilate(depth_map2, DIAMOND_KERNEL_5_t)

    plt.subplot(2, 1, 1)
    plt.title('cv2')
    plt.imshow(depth_map1, 'gray')

    plt.subplot(2, 1, 2)
    plt.title('torch')
    plt.imshow(depth_map2.squeeze(), 'gray')
    plt.show()


    # Hole closing
    #depth_map1 = cv2.morphologyEx(depth_map1, cv2.MORPH_CLOSE, FULL_KERNEL_5)
    depth_map1 = cv2.dilate(depth_map1, FULL_KERNEL_5)
    depth_map1 = cv2.erode(depth_map1, FULL_KERNEL_5)


    #depth_map2 = dilate(depth_map2, FULL_KERNEL_5_t)
    #depth_map2 = erode(depth_map2, FULL_KERNEL_5_t)
    #depth_map2 = cv2.erode(depth_map2.numpy().squeeze(), FULL_KERNEL_5)





    plt.subplot(2, 1, 1)
    plt.title('cv2')
    plt.imshow(depth_map1, 'gray')

    plt.subplot(2, 1, 2)
    plt.title('torch')
    plt.imshow(depth_map2.squeeze(), 'gray')
    plt.show()








    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map