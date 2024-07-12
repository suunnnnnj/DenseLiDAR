# denselidar.py
"""
maskFT(Attention map) 적용 version
사용법
파일 경로에 알맞는 데이터 경로 지정
command : python3 test.py
input : RGB image, LiDAR Raw data, Pseudo Depth map
result : Dense Depth * maskFT(Attention map)
"""

import torch
import numpy as np
import cv2
import os
from sample_dataloader.dataLoader import dataloader  # Import the dataloader function

from Submodules.DCU.submodels.depthCompletionNew_blockN import depthCompletionNew_blockN, maskFt
from Submodules.data_rectification import rectify_depth
from Submodules.ip_basic.depth_completion import ip_basic

def tensor_transform(sparse_depth_path, pseudo_depth_map, left_image_path):
    current_path = os.path.dirname(os.path.abspath(__file__))
    sparse_depth_np = cv2.imread(os.path.join(current_path, sparse_depth_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    left_image_np = cv2.imread(os.path.join(current_path, left_image_path)).astype(np.float32)

    sparse_depth = torch.from_numpy(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pseudo_depth = torch.from_numpy(pseudo_depth_map).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    left_image = torch.from_numpy(left_image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return sparse_depth, pseudo_depth, left_image

if __name__ == '__main__':
    # Use the dataloader to load the data
    datapath = './'  # Specify your data path here
    images, lidars, depths = dataloader(datapath)

    # Assuming we are using the first set of files for this example
    left_image_path = images[0]
    sparse_depth_path = lidars[0]
    output_path = 'result/sample1.png'  # Output image
    pseudo_depth_path = depths[0]

    # making pseudo depth map
    pseudo_depth_map = ip_basic(sparse_depth_path)

    # Transform tensor
    sparse_depth, pseudo_depth, left_image = tensor_transform(sparse_depth_path, pseudo_depth_map.astype(np.float32), left_image_path)

    # Rectified depth
    rectified_depth = rectify_depth(sparse_depth, pseudo_depth, threshold=1)

    # Depth Completion Model
    model = depthCompletionNew_blockN(bs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    rectified_depth = rectified_depth.to(device)
    pseudo_depth = pseudo_depth.to(device)
    left_image = left_image.to(device)

    sparse2 = rectified_depth
    mask = pseudo_depth

    # Forward pass
    with torch.no_grad():
        output_normal2, output_concat2 = model(left_image, sparse2, mask)

    print(f"output_normal2 shape: {output_normal2.shape}")
    print(f"output_concat2 shape: {output_concat2.shape}")
    print(f"output_normal2 min: {output_normal2.min().item()}, max: {output_normal2.max().item()}")
    print(f"output_concat2 min: {output_concat2.min().item()}, max: {output_concat2.max().item()}")

    # Initialize maskFt model and use it to process output_concat2
    mask_model = maskFt()
    mask_model.to(device)
    output_concat2_processed = mask_model(output_concat2)

    # Use the processed output for multiplication
    multiplied_output = output_normal2 * output_concat2_processed
    multiplied_output_np = multiplied_output.squeeze().detach().cpu().numpy()

    # Normalize the output for saving
    multiplied_output_np = cv2.normalize(multiplied_output_np, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(output_path, multiplied_output_np.astype(np.uint8))

    print(f"Multiplied normal2 and concat2 depth map saved to {output_path}")
