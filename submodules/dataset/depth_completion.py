import os
import sys
import time

import cv2
import numpy as np
import png
from tqdm import tqdm

from utils.depth_map_utils import fill_in_fast


def ip_basic(path, outputs_dir):
    data_split = 'val'

    # Test set
    input_depth_root_dir = os.path.expanduser(path)
    # data_split = 'test'

    # Fast fill with Gaussian blur @90Hz (paper result)
    fill_type = 'fast'
    extrapolate = False
    blur_type = 'gaussian'
    # Save output to disk or show process
    save_depth_maps = True

    os.makedirs(outputs_dir, exist_ok=True)

    output_depth_root_dir = os.path.join(outputs_dir)

    # Get images in sorted order
    images_to_use = []
    for root, _, files in os.walk(input_depth_root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                images_to_use.append(os.path.join(root, file))

    images_to_use = sorted(images_to_use)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 10
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)

    num_images = len(images_to_use)
    for i in tqdm(range(num_images)):
        depth_image_path = images_to_use[i]

        # Start timing
        start_total_time = time.time()

        # Load depth projections from uint16 image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            print(f"\nWarning: Could not load image at {depth_image_path}")
            continue

        projected_depths = np.float32(depth_image / 256.0)

        # Fill in
        start_fill_time = time.time()
        if fill_type == 'fast':
            final_depths = fill_in_fast(
                projected_depths, extrapolate=extrapolate, blur_type=blur_type)

        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        end_fill_time = time.time()

        # Save depth images to disk
        if save_depth_maps:
            depth_image_file_name = os.path.relpath(depth_image_path, input_depth_root_dir)
            depth_image_output_path = os.path.join(output_depth_root_dir, depth_image_file_name)
            depth_image_output_dir = os.path.dirname(depth_image_output_path)

            if not os.path.exists(depth_image_output_dir):
                os.makedirs(depth_image_output_dir)

            # Save depth map to a uint16 png (same format as disparity maps)
            with open(depth_image_output_path, 'wb') as f:
                depth_image = (final_depths * 256).astype(np.uint16)
                # pypng is used because cv2 cannot save uint16 format images
                writer = png.Writer(width=depth_image.shape[1],
                                    height=depth_image.shape[0],
                                    bitdepth=16,
                                    greyscale=True)
                writer.write(f, depth_image)

        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time