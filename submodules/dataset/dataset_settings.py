import glob
import os
import shutil

from pathlib import Path
from depth_completion import ip_basic
from utils.data_matching import matching_sync, matching_file_dir
from utils.image_preprocessing import image_preprocessing
from utils.data_remove import *

# YOUR DATA PATH
root_dir = os.path.join(Path(__file__).resolve().parents[2], 'datasets')
print(root_dir)

# 1. Remove kitti_raw's zip files
remove_zip(root_dir)

# 2. Image preprocessing: remove unused image files / split train and val / matching sync
image_train_list, image_val_list = image_preprocessing(root_dir)

# 3. Remove lidar sync not in image folder
matching_sync(root_dir, image_train_list, image_val_list)

# 4. making pseudo dense depth and pseudo gt map for using ip_basic
pseudo_depth_path = os.path.join(root_dir, 'pseudo_depth_map')
pseudo_gt_path = os.path.join(root_dir, 'pseudo_gt_map')
print('start making pseudo_depth_map')
ip_basic(os.path.join(root_dir, 'data_depth_velodyne'), pseudo_depth_path)
print('\nstart making pseudo_gt_map')
ip_basic(os.path.join(root_dir, 'data_depth_annotated'), pseudo_gt_path)

try:
    shutil.rmtree(os.path.join(root_dir, 'pseudo_depth_map', 'image_02'))
    shutil.rmtree(os.path.join(root_dir, 'pseudo_gt_map', 'image_02'))
except Exception as e:
    print(e)

matching_file_dir(root_dir)

remove_empty(root_dir)

remove_unmatching_file(root_dir)