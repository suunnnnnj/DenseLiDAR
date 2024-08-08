import os
import glob
import shutil

from submodules.utils.get_func import get_inner_folder


def remove_zip(root_dir):
    print('remove kitti_raw\'s zip files')
    zip_files = glob.glob(os.path.join(root_dir, 'kitti_raw', "*.zip"))
    for zip_file in zip_files:
        try:
            os.remove(zip_file)
        except Exception as e:
            print(f"Error deleting {zip_file}: {e}")
    print('zip file removal completed')

def remove_empty(root_dir):
    pos_list = ['image_02', 'image_03']  # left / right
    mode_list = ['train', 'val']
    for folder_name in os.listdir(root_dir):  # folder_name ex: data_depth_annotated
        for mode in mode_list:  # train / val
            for sync in os.listdir(os.path.join(root_dir, folder_name, mode)):
                folder_name = 'kitti_raw'
                if folder_name != 'kitti_raw':
                    path = os.path.join(root_dir, folder_name, mode, sync, 'proj_depth', get_inner_folder(folder_name))
                    try:
                        os.rmdir(path)
                    except Exception as e:
                        print(e)
                elif folder_name == 'kitti_raw':
                    path = os.path.join(root_dir, folder_name, mode, sync)

                    try:
                        shutil.rmtree(os.path.join(path, 'image_02'))
                        shutil.rmtree(os.path.join(path, 'image_03'))
                    except Exception as e:
                        print(e)

def remove_unmatching_file(root_dir):
    path_1 = os.path.join(root_dir, 'kitti_raw/train/2011_09_26_drive_0009_sync/proj_depth/image_02')
    path_2 = os.path.join(root_dir, 'kitti_raw/train/2011_09_26_drive_0009_sync/proj_depth/image_03')
    for n in range(177, 181):
        os.remove(os.path.join(path_1, f'0000000{n}.png'))
        os.remove(os.path.join(path_2, f'0000000{n}.png'))