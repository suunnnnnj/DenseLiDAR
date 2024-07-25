import os
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(ROOT_DIR)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_fold_path(filepath, seq, fold):
    dir = os.path.join(filepath, seq, fold)
    path = [os.path.join(dir, img) for img in os.listdir(dir) if is_image_file(img)]
    path.sort()
    return path

def get_file_path(filepath, seq):
    left_fold = 'proj_depth/image_02'
    right_fold = 'proj_depth/image_03'

    left = get_fold_path(filepath, seq, left_fold)
    right = get_fold_path(filepath, seq, right_fold)

    paths = np.append(left, right)
    return paths

def dataloader(filepath, mode='train'):
    if mode not in ['train', 'val']:
        raise ValueError("mode should be either 'train' or 'val'")

    temp = filepath
    filepath_ri = os.path.join(temp, f'kitti_raw/{mode}')
    filepath_gtl = os.path.join(temp, f'data_depth_annotated/{mode}')
    filepath_rl = os.path.join(temp, f'data_depth_velodyne/{mode}')
    filepath_pd = os.path.join(temp, f'pseudo_depth_map/{mode}')
    filepath_pg = os.path.join(temp, f'pseudo_gt_map/{mode}')

    seqs = [seq for seq in os.listdir(filepath_ri) if seq.find('sync') > -1]
    
    for seq in seqs:
        images = get_file_path(filepath_ri, seq)    # raw image
        lidars = get_file_path(filepath_rl, seq)    # raw lidar
        gt = get_file_path(filepath_gtl, seq)       # ground truth lidar
        pseudo = get_file_path(filepath_pd, seq)    # pseudo depth map
        pseudo_gt = get_file_path(filepath_pg, seq) # pseudo GT map

    print(mode)
    print(f"image: {len(images)} | lidar : {len(lidars)} | gt depth: {len(gt)} | pseudo depth: {len(pseudo)} | pseudo GT: {len(pseudo_gt)}")

    if mode == 'val':
        print("Data loading complete.\n")

    return images, lidars, gt, pseudo, pseudo_gt

if __name__ == '__main__':
    datapath = ''
    images, lidars, gt, pseudo, pseudo_gt = dataloader(datapath)
    print("Data loading complete.\n")