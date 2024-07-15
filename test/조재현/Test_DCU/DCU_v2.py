"""
command : python3 DCU_v2.py 
input : Raw LiDAR data, Pseudo depth map, RGB image, output path
Result : Dense depth map
"""
import torch
import numpy as np
import cv2
import os
from sample_dataloader.dataLoader import dataloader
from submodels.depthCompletionNew_blockN import depthCompletionNew_blockN


def rectify_depth(sparse_depth, pseudo_depth, threshold=1.0):
    difference = torch.abs(sparse_depth - pseudo_depth)
    rectified_depth = torch.where(difference > threshold, torch.tensor(0.0, device=sparse_depth.device), sparse_depth)
    return rectified_depth

if __name__ == '__main__':
    datapath = '.' # Insert your data path

    # Load data using the dataloader
    images, lidars, depths = dataloader(datapath)

    # Assuming single set of data for this example
    left_image_path = images[0]
    sparse_depth_path = lidars[0]
    pseudo_depth_path = depths[0]

    # Paths for saving results
    residual_output_path = '/home/mobiltech/Desktop/Test/residual_output.png'
    final_output_path = '/home/mobiltech/Desktop/Test/final_output.png'

    # Transform tensor
    sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    pseudo_depth_np = cv2.imread(pseudo_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    left_image_np = cv2.imread(left_image_path).astype(np.float32)

    sparse_depth = torch.from_numpy(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pseudo_depth = torch.from_numpy(pseudo_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    left_image = torch.from_numpy(left_image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Move tensors to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sparse_depth = sparse_depth.to(device)
    pseudo_depth = pseudo_depth.to(device)
    left_image = left_image.to(device)

    # Rectified depth
    rectified_depth = rectify_depth(sparse_depth, pseudo_depth, threshold=1)

    # DCU
    model = depthCompletionNew_blockN(bs=1)
    model.to(device)

    sparse2 = rectified_depth 
    mask = pseudo_depth  

    # Forward pass
    with torch.no_grad():
        output_normal2, output_concat2 = model(left_image, sparse2, mask)

    print(f"output_normal2 shape: {output_normal2.shape}")
    print(f"output_concat2 shape: {output_concat2.shape}")
    print(f"output_normal2 min: {output_normal2.min().item()}, max: {output_normal2.max().item()}")
    print(f"output_concat2 min: {output_concat2.min().item()}, max: {output_concat2.max().item()}")

    # normal2 * concat2
    output_concat2_first_channel = output_concat2[:, 0, :, :].unsqueeze(1)
    multiplied_output = output_normal2 * output_concat2_first_channel

    # Compute the residual depth map
    residual_depth = multiplied_output - rectified_depth

    # Convert residual depth to numpy and save
    residual_depth_np = residual_depth.squeeze().detach().cpu().numpy()
    cv2.imwrite(residual_output_path, residual_depth_np)

    print(f"Residual depth map saved to {residual_output_path}")

    # Add the residual depth map to the pseudo depth map
    final_depth = residual_depth + pseudo_depth

    # Convert final depth to numpy and save
    final_depth_np = final_depth.squeeze().detach().cpu().numpy()
    cv2.imwrite(final_output_path, final_depth_np)

    print(f"Final depth map saved to {final_output_path}")
