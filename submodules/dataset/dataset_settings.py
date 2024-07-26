import os
import glob

from depth_completion import ip_basic
from data_preprocessing import image_preprocessing

# YOUR DATA PATH
root_dir = '/home/mobiltech/Desktop/SSDC/datasets'

# 1. remove kitti_raw's zip files
print('remove kitti_raw\'s zip files')
zip_files = glob.glob(os.path.join(root_dir, 'kitti_raw', "*.zip"))
for zip_file in zip_files:
    try:
        os.remove(zip_file)
    except Exception as e:
        print(f"Error deleting {zip_file}: {e}")
print('zip file removal completed')

# 2.
image_preprocessing(root_dir)

# 4. making pseudo dense depth and pseudo gt map for using ip_basic
pseudo_depth_path = os.path.join(root_dir, 'pseudo_depth_map')
pseudo_gt_path = os.path.join(root_dir, 'pseudo_gt_map')
print('start making pseudo_depth_map')
#ip_basic(root_dir, pseudo_depth_path)
print('\nstart making pseudo_gt_map')
#ip_basic(root_dir, pseudo_gt_path)