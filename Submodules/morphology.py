import cv2
from torchvision.transforms import InterpolationMode, transforms

from Submodules.utils.utils_morphology import dilation, erosion, median_blur
from Submodules.utils.kernels import *
from Submodules.utils.visualization import visualize_1


def morphology_torch(projected_depths, device):
    np.set_printoptions(threshold=np.inf)
    torch.set_printoptions(profile="full")

    depth_map = projected_depths / 256.0
    max_depth = 100
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

    BICUBIC = InterpolationMode.BICUBIC
    resize_transform = transforms.Resize((256, 512), antialias=True, interpolation=BICUBIC)
    depth_map = resize_transform(depth_map)

    return depth_map
