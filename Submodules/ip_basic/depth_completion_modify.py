import os
import sys
import time
import cv2
import numpy as np
import png
import torch
from PIL import Image
import torchvision.transforms as transforms
from ip_basic_utils import depth_map_utils

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def ip_basic_m(sparse_depth):
    """Depth maps are saved to the 'outputs' folder."""

    ##############################
    # Options
    ##############################
    # Validation set
    data_split = 'val'

    # Test set
    # data_split = 'test'
    
    fill_type = 'fast'
    extrapolate = False
    blur_type = 'bilateral'

    # Save output to disk
    save_output = True

    ##############################
    # Processing
    ##############################
    if save_output:
        save_depth_maps = True
    else:
        save_depth_maps = False

    # Create output folder
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = os.path.join(this_file_path, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    output_folder_prefix = 'depth_' + data_split
    output_list = sorted(os.listdir(outputs_dir))
    if len(output_list) > 0:
        split_folders = [folder for folder in output_list if folder.startswith(output_folder_prefix)]
        if len(split_folders) > 0:
            last_output_folder = split_folders[-1]
            last_output_index = int(last_output_folder.split('_')[-1])
        else:
            last_output_index = -1
    else:
        last_output_index = -1
    output_depth_dir = os.path.join(outputs_dir, '{}_{:03d}'.format(output_folder_prefix, last_output_index + 1))

    if save_output:
        if not os.path.exists(output_depth_dir):
            os.makedirs(output_depth_dir)
        else:
            raise FileExistsError('Already exists!')
        print('Output dir:', output_depth_dir)

    # Convert sparse depth tensor to numpy array
    sparse_depth = sparse_depth.squeeze().cpu().numpy()
    projected_depths = np.float32(sparse_depth / 256.0)

    # Fill in
    if fill_type == 'fast':
        final_depths = depth_map_utils.fill_in_fast(projected_depths, extrapolate=extrapolate, blur_type=blur_type)
    elif fill_type == 'multiscale':
        final_depths, process_dict = depth_map_utils.fill_in_multiscale(
            projected_depths, extrapolate=extrapolate, blur_type=blur_type)
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))

    # Save depth images to disk
    if save_depth_maps:
        depth_image_file_name = 'pseudo_map.png'

        # Save depth map to a uint16 png (same format as disparity maps)
        file_path = os.path.join(output_depth_dir, depth_image_file_name)
        with open(file_path, 'wb') as f:
            depth_image = (final_depths * 256).astype(np.uint16)

            # pypng is used because cv2 cannot save uint16 format images
            writer = png.Writer(width=depth_image.shape[1],
                                height=depth_image.shape[0],
                                bitdepth=16,
                                greyscale=True)
            writer.write(f, depth_image)

    final_depths = torch.from_numpy(final_depths).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return final_depths
    
def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('L')

    transform = transforms.ToTensor()
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
    
# Example
image_path = '/home/mobiltech/Desktop/Test/SSDC/Submodules/ip_basic/0000000005.png'

image_tensor = load_image_as_tensor(image_path)

final_depths = ip_basic(image_tensor)

print(final_depths)