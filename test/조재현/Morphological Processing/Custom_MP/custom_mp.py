"""
command : python3 custom_mp.py 
input : Raw LiDAR data
Result : Pseudo depth map
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def anisotropic_diffusion(img, num_iter, kappa, gamma):
    # Apply anisotropic diffusion filter
    img = img.astype(np.float32)
    for i in range(num_iter):
        # Calculate gradients in each direction
        delta_n = img[:-2, 1:-1] - img[1:-1, 1:-1]
        delta_s = img[2:, 1:-1] - img[1:-1, 1:-1]
        delta_e = img[1:-1, 2:] - img[1:-1, 1:-1]
        delta_w = img[1:-1, :-2] - img[1:-1, 1:-1]
        
        # Calculate diffusion coefficients
        c_n = np.exp(-(delta_n/kappa)**2)
        c_s = np.exp(-(delta_s/kappa)**2)
        c_e = np.exp(-(delta_e/kappa)**2)
        c_w = np.exp(-(delta_w/kappa)**2)
        
        # Update image
        img[1:-1, 1:-1] += gamma * (
            c_n * delta_n + c_s * delta_s +
            c_e * delta_e + c_w * delta_w
        )
    return img

def sobel_edge_detection(image, ksize=3):
    # Convert image to 8-bit if not already
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))
    
    # Apply Sobel edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    
    # Convert edge values to 8-bit
    sobel_edges = cv2.convertScaleAbs(sobel_edges)
    
    return sobel_edges

def inpaint_with_edges(sparse_depth, edges, inpaintRadius=3, method='telea'):
    # Create a mask to differentiate between points with and without depth values
    mask = (sparse_depth == 0).astype(np.uint8)
    
    # Combine edges and mask
    combined_mask = cv2.bitwise_or(mask, edges)
    
    # Select inpainting method
    if method == 'telea':
        inpaint_method = cv2.INPAINT_TELEA
    else:
        inpaint_method = cv2.INPAINT_NS
    
    # Perform inpainting
    dense_depth = cv2.inpaint(sparse_depth, combined_mask, inpaintRadius=inpaintRadius, flags=inpaint_method)
    
    return dense_depth

def smooth_depth_map(depth_map):
    # Convert depth map to 32-bit floating point format
    depth_map_32f = depth_map.astype(np.float32)
    
    # Apply anisotropic diffusion filter
    depth_map_ad = anisotropic_diffusion(depth_map_32f, num_iter=15, kappa=30, gamma=0.2)
    
    # Apply bilateral filter
    smoothed_depth = cv2.bilateralFilter(depth_map_ad, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Enhance edges using Sobel filter
    sobel_edges = sobel_edge_detection(smoothed_depth, ksize=3)
    smoothed_depth += sobel_edges.astype(np.float32) * 0.5
    
    return smoothed_depth

def normalize_and_convert_to_uint8(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_image.astype(np.uint8)

if __name__ == "__main__":
    # Load LiDAR data image
    lidar_path = "lidar.png"
    sparse_depth = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)
    if sparse_depth is None:
        raise FileNotFoundError(f"LiDAR file {lidar_path} not found.")
    
    # Apply Sobel edge detection
    edges = sobel_edge_detection(sparse_depth, ksize=5)
    
    # Inpaint depth map using edges
    dense_depth = inpaint_with_edges(sparse_depth, edges, inpaintRadius=5, method='telea')
    
    # Smooth the depth map
    smoothed_depth = smooth_depth_map(dense_depth)
    
    # Normalize and convert image to uint8 before saving
    smoothed_depth_uint8 = normalize_and_convert_to_uint8(smoothed_depth)
    
    # Visualize results
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1)
    plt.title("Sparse Depth")
    plt.imshow(sparse_depth, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Dense Depth with Edges")
    plt.imshow(dense_depth, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Smoothed Dense Depth")
    plt.imshow(smoothed_depth, cmap='gray')
    plt.axis('off')
    
    plt.show()

    # Save the result
    output_path = "smoothed_dense_depth.png"
    cv2.imwrite(output_path, smoothed_depth_uint8)
    print(f"Smoothed dense depth map saved to {output_path}.")

