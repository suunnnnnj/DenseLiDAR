import os
import shutil
from tqdm import tqdm

root_dir = '/home/research1/Desktop/gachon/SSDC/datasets'


train_dir = os.path.join(root_dir, 'kitti_raw/train')
val_dir = os.path.join(root_dir, 'kitti_raw/val')

train_list = os.listdir(train_dir)
val_list = os.listdir(val_dir)

folder_list = ['data_depth_annotated', 'data_depth_velodyne', 'pseudo_depth_map', 'pseudo_gt_map']

for data_folder in tqdm(folder_list):
    current_dir = os.path.join(root_dir, data_folder, 'train')
    train_list = os.listdir(current_dir) # other folder's train list
    for dir in train_list:
        if not os.path.exists(os.path.join(train_dir, dir)):
            shutil.rmtree(os.path.join(current_dir, dir))

for data_folder in tqdm(folder_list):
    current_dir = os.path.join(root_dir, data_folder, 'val')
    val_list = os.listdir(current_dir) # other folder's val list
    for dir in val_list:
        if not os.path.exists(os.path.join(val_dir, dir)):
            shutil.rmtree(os.path.join(current_dir, dir))