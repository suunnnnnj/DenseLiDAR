import os
import sys
import time
import cv2
import numpy as np
import png
import torch
from Submodules.ip_basic.ip_basic_utils import depth_map_utils

def ip_basic(projected_depths):
    """깊이 맵이 'outputs' 폴더에 저장됩니다."""

    # 디버깅 문구
    print("fill_in_fast 호출 전:")
    print(f"projected_depths의 타입: {type(projected_depths)}")
    print(f"projected_depths의 형상: {projected_depths.shape if isinstance(projected_depths, np.ndarray) else 'NumPy 배열이 아님'}")

    # projected_depths가 NumPy 배열인지 확인
    if not isinstance(projected_depths, np.ndarray):
        projected_depths = np.array(projected_depths).squeeze()

    print("필요한 경우 NumPy 배열로 변환 후:")
    print(f"projected_depths의 타입: {type(projected_depths)}")
    print(f"projected_depths의 형상: {projected_depths.shape}\n\n")

    ##############################
    # 옵션
    ##############################

    # 검증 세트
    data_split = 'val'

    # 테스트 세트
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
        raise ValueError('유효하지 않은 fill_type: {}'.format(fill_type))

    final_depths = torch.from_numpy(final_depths).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return final_depths
