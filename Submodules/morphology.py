import torchvision.transforms
from PIL import Image

from Submodules.utils.utils_morphology import dilation, erosion, median_blur
from Submodules.utils.kernels import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

def morphology_torch(depth_map, device):
    depth_map = depth_map.unsqueeze(0)

    depth_map = depth_map / 256.0
    max_depth = 100

    depth_map.to(device)
    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = dilation(depth_map, DIAMOND_KERNEL_5.to(device), device)
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


"""image_path = '/home/mobiltech/Desktop/SSDC/sample/0000000005.png'

# 이미지 읽기
image = Image.open(image_path)

# 이미지 변환 (PIL 이미지 -> 텐서)
transform = transforms.ToTensor()
image = transform(image).to(torch.float32)
morphology_torch(image, torch.device("cuda"))"""