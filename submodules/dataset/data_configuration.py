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

def matching_file_path(src_base_path, dest_base_path):
    # Iterate through all subfolders in the source base path
    for folder_name in os.listdir(src_base_path):
        src_path1 = os.path.join(src_base_path, folder_name, 'image_02/data')
        dest_path2 = os.path.join(dest_base_path, folder_name, 'proj_depth/image_02')

        # Ensure the source directory exists
        if not os.path.exists(src_path1):
            print(f"Source path does not exist: {src_path1}")
            continue

        # Ensure the destination directory exists
        os.makedirs(dest_path2, exist_ok=True)

        # Move files from the current source folder to the corresponding destination folder
        for file_name in os.listdir(src_path1):
            full_file_name = os.path.join(src_path1, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, dest_path2)
                print(f'Moved {full_file_name} to {dest_path2}')

        # Remove the source directory after moving the files
        if os.path.exists(src_path1) and not os.listdir(src_path1):
            os.rmdir(src_path1)
            print(f'Deleted empty source folder: {src_path1}')
        else:
            print(f"Source folder not empty, can't delete: {src_path1}")