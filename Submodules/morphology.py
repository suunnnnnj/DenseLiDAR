from Submodules.utils.utils_morphology import dilation, erosion, median_blur
from Submodules.utils.kernels import *

def morphology_torch(projected_depths):
    depth_map = np.float32(projected_depths / 256.0)
    max_depth = 100
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    depth_map = torch.from_numpy(depth_map).unsqueeze(1).float()
    depth_map = depth_map.transpose(0, 1).unsqueeze(0)

    # Dilate
    depth_map = dilation(depth_map, DIAMOND_KERNEL_5)
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Hole Closing
    depth_map = dilation(depth_map, FULL_KERNEL_5)
    depth_map = erosion(depth_map, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = dilation(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = median_blur(depth_map, 5)

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map