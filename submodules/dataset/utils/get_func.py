import os
from tqdm import tqdm

def get_sync_path(root_dir, kitti_raw_list):
    for date in tqdm(kitti_raw_list):
        date_path = os.path.join(root_dir, date)
        if os.path.exists(date_path) and os.path.isdir(date_path):
            for sync in os.listdir(date_path):
                sync_path = os.path.join(date_path, sync)
    return sync_path

def get_last_5(dir_path):
    last_5 = []
    file_names = os.listdir(dir_path)
    file_names.sort()
    last_file_name = file_names[-1] if file_names else None
    last_file_int = int(last_file_name[:-4])

    for i in range(5):
        file_name = last_file_int - i
        file = str(file_name).zfill(10) + '.png'
        last_5.append(file)

    return last_5

def get_inner_folder(folder):
    if folder == 'data_depth_annotated' or folder == 'pseudo_gt_map':
        return 'groundtruth'
    elif folder == 'data_depth_velodyne' or folder == 'pseudo_depth_map':
        return 'velodyne_raw'