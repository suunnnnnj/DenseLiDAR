import os
import glob

from depth_completion import ip_basic

# YOUR DATA PATH
dataset_path = '/home/mobiltech/Desktop/SSDC/datasets'

# 1. remove kitti_raw's zip files
print('remove kitti_raw\'s zip files')
zip_files = glob.glob(os.path.join(dataset_path, 'kitti_raw', "*.zip"))
for zip_file in zip_files:
    try:
        os.remove(zip_file)
    except Exception as e:
        print(f"Error deleting {zip_file}: {e}")
print('zip file removal completed')

# 2. making pseudo dense depth and pseudo gt map for using ip_basic
pseudo_depth_path = os.path.join(dataset_path, 'pseudo_depth_map')
pseudo_gt_path = os.path.join(dataset_path, 'pseudo_gt_map')
print('start making pseudo_depth_path')
ip_basic(dataset_path, pseudo_depth_path)
print('\nstart making pseudo_depth_path')
ip_basic(dataset_path, pseudo_gt_path)