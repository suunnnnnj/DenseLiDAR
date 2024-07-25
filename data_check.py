import os
import argparse

def count_files_in_folder(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # List all files in the folder
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    print(f"Number of files in '{folder_path}': {file_count}")

    return file_count


root_dir = '/home/research1/Desktop/gachon/SSDC/datasets/'
lidar_dir = 'data_depth_velodyne/train'
image_dir = 'kitti_raw/train'
left_right = ['proj_depth/image_02', 'proj_depth/image_03']


lidar_list = os.listdir(os.path.join(root_dir, lidar_dir))
image_list = os.listdir(os.path.join(root_dir, image_dir))

lidar_list.sort()
image_list.sort()

for sync in lidar_list:
    sync_pathl = os.path.join(root_dir, lidar_dir, sync)
    sync_pathi = os.path.join(root_dir, image_dir, sync)
    for pos in left_right:
        lidar_path = os.path.join(sync_pathl, pos)
        image_path = os.path.join(sync_pathi, pos)
        lidar_count = count_files_in_folder(lidar_path)  
        image_count = count_files_in_folder(image_path)
        if lidar_count != image_count:
            print("doesn't match these files")
