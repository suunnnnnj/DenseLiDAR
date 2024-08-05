import os
import shutil

from submodules.dataset.depth_completion import ip_basic
from submodules.dataset.utils.data_matching import matching_sync, matching_file_dir
from submodules.dataset.utils.image_preprocessing import image_preprocessing

# YOUR DATA PATH
root_dir = '/home/mobiltech/Desktop/DenseLiDAR/datasets'

# 1. Remove kitti_raw's zip files
'''print('remove kitti_raw\'s zip files')
zip_files = glob.glob(os.path.join(root_dir, 'kitti_raw', "*.zip"))
for zip_file in zip_files:
    try:
        os.remove(zip_file)
    except Exception as e:
        print(f"Error deleting {zip_file}: {e}")
print('zip file removal completed')'''

# 2. Image preprocessing: remove unused image files / split train and val / matching sync
#image_train_list, image_val_list = image_preprocessing(root_dir)

# 3. Remove lidar sync not in image folder
#matching_sync(root_dir, image_train_list, image_val_list)

# 4. making pseudo dense depth and pseudo gt map for using ip_basic
'''pseudo_depth_path = os.path.join(root_dir, 'pseudo_depth_map')
pseudo_gt_path = os.path.join(root_dir, 'pseudo_gt_map')
print('start making pseudo_depth_map')
ip_basic(os.path.join(root_dir, 'data_depth_velodyne'), pseudo_depth_path)
print('\nstart making pseudo_gt_map')
ip_basic(os.path.join(root_dir, 'data_depth_annotated'), pseudo_gt_path)'''

try:
    shutil.rmtree(os.path.join(root_dir, 'pseudo_depth_map', 'image_02'))
    shutil.rmtree(os.path.join(root_dir, 'pseudo_gt_map', 'image_02'))
except Exception as e:
    print(e)

matching_file_dir(root_dir)