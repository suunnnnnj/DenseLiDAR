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


def dataloader(filepath, mode='train'):
    images = []
    gt_lidars = []
    lidars = []
    pseudo_depths = []
    pseudo_gts = []

    if mode not in ['train', 'val']:
        raise ValueError("mode should be either 'train' or 'val'")

    temp = filepath
    filepath_ri = os.path.join(temp, f'kitti_raw/{mode}')              # raw image
    filepath_gl = os.path.join(temp, f'data_depth_annotated/{mode}')   # gt lidar
    filepath_rl = os.path.join(temp, f'data_depth_velodyne/{mode}')    # raw lidar
    filepath_pd = os.path.join(temp, f'pseudo_depth_map/{mode}')       # pseudo depth map
    filepath_pg = os.path.join(temp, f'pseudo_gt_map/{mode}')          # pseudo gt map

    seqs = [seq for seq in os.listdir(filepath_ri) if seq.find('sync') > -1]
    left_fold = 'proj_depth/image_02'
    right_fold = 'proj_depth/image_03'
    
    for seq in seqs:
        # raw image
        left_image = os.path.join(filepath_ri, seq, left_fold)
        left_image = [os.path.join(left_image, img) for img in os.listdir(left_image) if is_image_file(img)]
        left_image.sort()
        images = np.append(images, left_image)

        right_images = os.path.join(filepath_ri, seq, right_fold)
        right_image = [os.path.join(right_images, img) for img in os.listdir(right_images) if is_image_file(img)]
        right_image.sort()
        images = np.append(images, right_image)

        # gt lidar
        left_gt_lidar = os.path.join(filepath_gl, seq, left_fold)
        left_gt_lidar = [os.path.join(left_gt_lidar, img) for img in os.listdir(left_gt_lidar) if is_image_file(img)]
        left_gt_lidar.sort()
        gt_lidars = np.append(gt_lidars, left_gt_lidar)

        right_gt_lidar = os.path.join(filepath_gl, seq, right_fold)
        right_gt_lidar = [os.path.join(right_gt_lidar, img) for img in os.listdir(right_gt_lidar) if is_image_file(img)]
        right_gt_lidar.sort()
        gt_lidars = np.append(gt_lidars, right_gt_lidar)

        # raw lidar
        left_lidar = os.path.join(filepath_rl, seq, left_fold)
        left_lidar = [os.path.join(left_lidar, img) for img in os.listdir(left_lidar) if is_image_file(img)]
        left_lidar.sort()
        lidars = np.append(lidars, left_lidar)

        right_lidar = os.path.join(filepath_rl, seq, right_fold)
        right_lidar = [os.path.join(right_lidar, img) for img in os.listdir(right_lidar) if is_image_file(img)]
        right_lidar.sort()
        lidars = np.append(lidars, right_lidar)

        # pseudo depth map
        left_pseudo_depth = os.path.join(filepath_pd, seq, left_fold)
        left_pseudo_depth = [os.path.join(left_pseudo_depth, img) for img in os.listdir(left_pseudo_depth) if is_image_file(img)]
        left_pseudo_depth.sort()
        pseudo_depths = np.append(pseudo_depths, left_pseudo_depth)

        right_pseudo_depth = os.path.join(filepath_pd, seq, right_fold)
        right_pseudo_depth = [os.path.join(right_pseudo_depth, img) for img in os.listdir(right_pseudo_depth) if is_image_file(img)]
        right_pseudo_depth.sort()
        pseudo_depths = np.append(pseudo_depths, right_pseudo_depth)

        # pseudo gt map
        left_pseudo_gt = os.path.join(filepath_pg, seq, left_fold)
        left_pseudo_gt = [os.path.join(left_pseudo_gt, img) for img in os.listdir(left_pseudo_gt) if is_image_file(img)]
        left_pseudo_gt.sort()
        pseudo_gts = np.append(pseudo_gts, left_pseudo_gt)

        right_pseudo_gt = os.path.join(filepath_pg, seq, right_fold)
        right_pseudo_gt = [os.path.join(right_pseudo_gt, img) for img in os.listdir(right_pseudo_gt) if is_image_file(img)]
        right_pseudo_gt.sort()
        pseudo_gts = np.append(pseudo_gts, right_pseudo_gt)

    raw_image = images
    gt_lidar = gt_lidars
    raw_lidar = lidars
    pseudo_depth_map = pseudo_depths
    pseudo_gt_map = pseudo_gts

    print(f"[Number of {mode} files] image: {len(raw_image)} | gt lidar: {len(gt_lidar)} | raw lidar: {len(raw_lidar)} | pseudo depth map: {len(pseudo_depth_map)} | pseudo gt map: {len(pseudo_gt_map)}\n")

    return raw_image, gt_lidar, raw_lidar, pseudo_depth_map, pseudo_gt_map


if __name__ == '__main__':
    datapath = ''
    raw_image, gt_lidar, raw_lidar, pseudo_depth_map, pseudo_gt_map = dataloader(datapath)
    print("Data loading complete.\n")