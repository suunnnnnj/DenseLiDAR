import os
import shutil
from tqdm import tqdm

'''train_dir = os.path.join(root_dir, 'kitti_raw/train')
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
            shutil.rmtree(os.path.join(current_dir, dir))'''

def matching_sync(root_dir, image_train_list, image_val_list):
    fold_list = ['data_depth_annotated', 'data_depth_velodyne']
    for fold in fold_list:
        for dir in ['train', 'val']:
            print(f'remove in {fold}/{dir} not in kitti_raw')
            sync_dir = os.path.join(root_dir, fold, dir)
            sync_list = os.listdir(sync_dir)

            if dir == 'train':
                reference_list = image_train_list
            else:
                reference_list = image_val_list

            for folder in sync_list:
                if folder not in reference_list:
                    folder_path = os.path.join(sync_dir, folder)
                    shutil.rmtree(folder_path)

def matching_file_dir(root_dir):
    # Iterate through all subfolders in the source base path
    pos_list = ['image_02', 'image_03'] # left / right
    mode_list = ['train', 'val']

    for folder_name in os.listdir(root_dir):
        # Ensure the destination directory exists
        # Move files from the current source folder to the corresponding destination folder
        for mode in mode_list:
            for sync in os.listdir(os.path.join(root_dir, folder_name, mode)):
                for pos in pos_list:
                    path = os.path.join(root_dir, folder_name, mode, sync, 'proj_depth')
                    if folder_name != 'kitti_raw':
                        src_path = os.path.join(path, os.listdir(path)[0], pos)
                        dest_path = os.path.join(path, pos)
                        os.makedirs(dest_path, exist_ok=True)
                        print(src_path)
                        print(dest_path)
                    elif folder_name== 'kitti_raw':
                        '''src_path = os.path.join(path, os.listdir(path)[0], pos)
                        dest_path = os.path.join(path, pos)
                        os.makedirs(dest_path, exist_ok=True)
                        print(src_path)
                        print(dest_path)'''

                        continue
                    for file_name in src_path:
                        full_file_name = os.listdir(os.path.join(src_path, file_name))
                        print(full_file_name)
                        if os.path.isfile(full_file_name):
                            shutil.move(full_file_name, dest_path)

            # Remove the source directory after moving the files
            '''if os.path.exists(src_path) and not os.listdir(src_path):
                os.rmdir(src_path)
                print(f'Deleted empty source folder: {src_path}')
            else:
                print(f"Source folder not empty, can't delete: {src_path}")'''
