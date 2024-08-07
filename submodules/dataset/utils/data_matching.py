import os
import shutil

from submodules.dataset.utils.get_func import get_inner_folder


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
    pos_list = ['image_02', 'image_03']  # left / right
    mode_list = ['train', 'val']

    for folder_name in os.listdir(root_dir):  # folder_name ex: data_depth_annotated
        for mode in mode_list:  # train / val
            for sync in os.listdir(os.path.join(root_dir, folder_name, mode)):
                path = os.path.join(root_dir, folder_name, mode, sync, 'proj_depth')
                fold_name = get_inner_folder(folder_name)

                for pos in pos_list:  # image_02 / image_03

                    if folder_name != 'kitti_raw':

                        src_path = os.path.join(path, fold_name, pos)
                        dest_path = os.path.join(path)
                        try:
                            shutil.move(src_path, dest_path)
                        except Exception as e:
                            print(e)
                    elif folder_name == 'kitti_raw':
                        src_path = os.path.join(os.path.dirname(path), pos, 'data')
                        dest_path = os.path.join(path, pos)
                        try:
                            shutil.move(src_path, dest_path)
                        except Exception as e:
                            print(e)