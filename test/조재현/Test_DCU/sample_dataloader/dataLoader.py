import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.dirname(ROOT_DIR)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    images = []
    lidars = []
    depths = []

    image_path = os.path.join(filepath, 'sample_data/image')
    lidar_path = os.path.join(filepath, 'sample_data/lidar')
    depth_path = os.path.join(filepath, 'sample_data/groundtruth')

    # Load image files
    for img_file in os.listdir(image_path):
        if is_image_file(img_file):
            images.append(os.path.join(image_path, img_file))

    # Load lidar files
    for lidar_file in os.listdir(lidar_path):
        lidars.append(os.path.join(lidar_path, lidar_file))

    # Load depth files
    for depth_file in os.listdir(depth_path):
        depths.append(os.path.join(depth_path, depth_file))

    # Sort the files to ensure correct ordering
    images.sort()
    lidars.sort()
    depths.sort()

    return images, lidars, depths

if __name__ == '__main__':
    datapath = '.' #insert your data path
    left_train, lidar2_train, depth_train = dataloader(datapath)
