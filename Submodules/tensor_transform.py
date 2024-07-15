import cv2
import torch
import numpy as np

def tensor_transform(sparse_depth_path, left_image_path):
    sparse_depth_np = cv2.imread(sparse_depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    sparse_depth = torch.from_numpy(sparse_depth_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    left_image_np = cv2.imread(left_image_path).astype(np.float32)
    left_image = torch.from_numpy(left_image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return sparse_depth, left_image

def multiply_output(output_normal, output_concat2_processed):
    multiplied_output = output_normal * output_concat2_processed
    multiplied_output_np = multiplied_output.squeeze().detach().cpu().numpy()
    output = cv2.normalize(multiplied_output_np, None, 0, 255, cv2.NORM_MINMAX)
    return output