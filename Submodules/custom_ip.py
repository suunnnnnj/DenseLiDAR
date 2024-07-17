import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import griddata
import torch.nn.functional as F
from scipy.spatial import cKDTree

def interpolate_depth_map(sparse_depth):
    batch_size, channel, height, width = sparse_depth.shape
    device = sparse_depth.device
    max_distance=5

    dense_depth = torch.zeros_like(sparse_depth, device=device)
    
    for b in range(batch_size):
        sparse_depth_np = sparse_depth[b, 0].cpu().numpy()
        
        # Extract valid depth values
        y, x = np.where(sparse_depth_np > 0)
        z = sparse_depth_np[y, x]
        
        if len(z) > 0:
            # Create a KDTree for fast distance computation
            tree = cKDTree(np.vstack((x, y)).T)
            
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            
            # Find the nearest distance to a valid point for each grid point
            distances, _ = tree.query(grid_points)
            distances = distances.reshape((height, width))
            
            # Perform interpolation only for points within the max_distance
            dense_depth_np = griddata((x, y), z, (grid_x, grid_y), method='linear')
            dense_depth_np[np.isnan(dense_depth_np)] = 0
            
            # Only update the valid regions in dense_depth
            valid_mask = (distances <= max_distance)
            dense_depth[b, 0][valid_mask] = torch.tensor(dense_depth_np[valid_mask], device=device, dtype=dense_depth.dtype)
    
    return dense_depth

def load_images(image_paths):
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]
    images = [cv2.resize(image, (images[0].shape[1], images[0].shape[0])) for image in images]
    images = [image.astype(np.float32) for image in images]
    images_tensor = torch.stack([torch.tensor(image) for image in images])
    return images_tensor

def visualize_depth_maps(sparse_depth, dense_depth, batch_index=0):
    sparse_depth_np = sparse_depth[batch_index].cpu().numpy()
    dense_depth_np = dense_depth[batch_index].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Sparse Depth Map')
    plt.imshow(sparse_depth_np, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title('Dense Depth Map')
    plt.imshow(dense_depth_np, cmap='gray')
    
    plt.show()

def save_depth_map(dense_depth, file_path):
    dense_depth_np = dense_depth.cpu().numpy()
    normalized_dense_depth = cv2.normalize(dense_depth_np, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(file_path, normalized_dense_depth.astype(np.uint8))


def main(image_paths):
    sparse_depth = load_images(image_paths)
    dense_depth = interpolate_depth_map(sparse_depth)
    visualize_depth_maps(sparse_depth, dense_depth, batch_index=0)
    
    for i in range(len(image_paths)):
        save_depth_map(dense_depth[i], f'dense_depth_map_batch123_{i}.png')

if __name__ == "__main__":
    image_folder = 'sample/'  # 이미지 파일이 있는 폴더 경로로 변경하세요
    image_paths = glob(f'{image_folder}/*.png')
    main(image_paths)
