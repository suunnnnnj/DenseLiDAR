from Submodules.utils.utils_morphology import dilation, erosion, median_blur
from Submodules.utils.kernels import *

def morphology_torch(depth_map, device):
    max_depth = 1
    
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = dilation(depth_map, DIAMOND_KERNEL_5, device)
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Hole Closing
    depth_map = dilation(depth_map, FULL_KERNEL_5, device)
    depth_map = erosion(depth_map, FULL_KERNEL_5, device)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = dilation(depth_map, FULL_KERNEL_7, device)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = median_blur(depth_map, 5)

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map
