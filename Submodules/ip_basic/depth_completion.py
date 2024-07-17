import os
import sys
import time
import cv2
import numpy as np
import png
import torch
from Submodules.ip_basic.ip_basic_utils import depth_map_utils
import matplotlib.pyplot as plt


def ip_basic(projected_depths):

    # 검증 세트
    data_split = 'val'

    # 테스트 세트
    # data_split = 'test'

    extrapolate = False
    blur_type = 'bilateral'
    projected_depths = np.float32(projected_depths / 256.0)

    # Fill in
    final_depths_torch = depth_map_utils.fill_in_torch(projected_depths, extrapolate=extrapolate, blur_type=blur_type)

    return final_depths_torch

img = cv2.imread('test/0000000005.png', cv2.COLOR_BGR2GRAY)

res_torch = ip_basic(img)
plt.title('fin')
plt.imshow(res_torch.squeeze(), 'gray')
plt.show()