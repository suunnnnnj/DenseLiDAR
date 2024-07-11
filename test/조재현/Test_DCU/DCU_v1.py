"""

파일 경로에 알맞는 데이터 경로 지정
command : python3 DCU_v1.py
input : RGB image, LiDAR Raw data, Pseudo Depth map
"""
import torch
import numpy as np
import cv2
import os
from submodels.depthCompletionNew_blockN import depthCompletionNew_blockN


def rectify_depth(sparse_depth, pseudo_depth, threshold=1.0):
    difference = torch.abs(sparse_depth - pseudo_depth)
    rectified_depth = torch.where(difference > threshold, torch.tensor(0.0, device=sparse_depth.device), sparse_depth)
    return rectified_depth

# path
sparse_depth_path = '/home/mobiltech/Desktop/Test/lidar.png'  # Raw lidar data
pseudo_depth_path = '/home/mobiltech/Desktop/Test/1.png'  # Pseudo depth
left_image_path = '/home/mobiltech/Desktop/Test/image.png'  # RGB image
output_path = '/home/mobiltech/Desktop/Test/multiplied_output.png'  # Output image

# Transform tensor
sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
pseudo_depth_np = cv2.imread(pseudo_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
left_image_np = cv2.imread(left_image_path).astype(np.float32)

sparse_depth = torch.from_numpy(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
pseudo_depth = torch.from_numpy(pseudo_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
left_image = torch.from_numpy(left_image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

# rectified_depth
rectified_depth = rectify_depth(sparse_depth, pseudo_depth, threshold=1)

# DCU
model = depthCompletionNew_blockN(bs=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
sparse_depth = sparse_depth.to(device)
pseudo_depth = pseudo_depth.to(device)
left_image = left_image.to(device)

sparse2 = sparse_depth 
mask = pseudo_depth  

# forward pass
with torch.no_grad():
    output_normal2, output_concat2 = model(left_image, sparse2, mask)

print(f"output_normal2 shape: {output_normal2.shape}")
print(f"output_concat2 shape: {output_concat2.shape}")
print(f"output_normal2 min: {output_normal2.min().item()}, max: {output_normal2.max().item()}")
print(f"output_concat2 min: {output_concat2.min().item()}, max: {output_concat2.max().item()}")

# normal2 * concat2
output_concat2_first_channel = output_concat2[:, 0, :, :].unsqueeze(1)
multiplied_output = output_normal2 * output_concat2_first_channel
multiplied_output_np = multiplied_output.squeeze().detach().cpu().numpy()

cv2.imwrite(output_path, multiplied_output_np)

print(f"Multiplied normal2 and concat2 depth map saved to {output_path}")
