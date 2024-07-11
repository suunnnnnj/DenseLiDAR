"""
command : python3 Data_Rectification.py
input : Raw LiDAR data, Pseudo depth map, output path
"""

import torch
import numpy as np
import cv2

# 기존 논문에 구현된 morphological processing method

# def morphological_processing(sparse_depth):
    
#     kernel = np.ones((5, 5), np.uint8)
#     dilated = cv2.dilate(sparse_depth, kernel, iterations=1)
#     return dilated


def rectify_depth(sparse_depth, pseudo_depth, threshold=1.0):
    
    difference = torch.abs(sparse_depth - pseudo_depth)
    rectified_depth = torch.where(difference > threshold, torch.tensor(0.0, device=sparse_depth.device), sparse_depth)
    return rectified_depth

# image path
sparse_depth_path = '/home/mobiltech/Desktop/Test/lidar.png' #raw lidar data dir
pseudo_depth_path = '/home/mobiltech/Desktop/Test/1.png'    # ip basic result dir
output_path = '/home/mobiltech/Desktop/Test/rectified_depth3.png'
concatenated_output_path = '/home/mobiltech/Desktop/Test/concatenated_depth.png'

# Transform Tensor
sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
pseudo_depth_np = cv2.imread(pseudo_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

sparse_depth = torch.tensor(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
pseudo_depth = torch.tensor(pseudo_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# optional
# processed_sparse_depth = morphological_processing(sparse_depth)

rectified_depth = rectify_depth(sparse_depth, pseudo_depth, threshold=1)

# save result
rectified_depth_np = rectified_depth.squeeze().numpy()

cv2.imwrite(output_path, rectified_depth_np)

print(f"Rectified depth map saved to {output_path}\n")
