import os
import shutil

from tqdm import tqdm

img_dir = ['image_02/data/', 'image_03/data/']
first_5 = ['0000000000.png', '0000000001.png', '0000000002.png', '0000000003.png', '0000000004.png']

def remove_unused_files(root_dir, kitti_raw_list):
    for date in tqdm(kitti_raw_list):
        date_path = os.path.join(root_dir, date)
        if os.path.exists(date_path) and os.path.isdir(date_path):
            for sync in os.listdir(date_path):
                sync_path = os.path.join(date_path, sync)
                if os.path.exists(sync_path) and os.path.isdir(sync_path):
                    try:
                        shutil.rmtree(os.path.join(sync_path, 'image_00'))
                        shutil.rmtree(os.path.join(sync_path, 'image_01'))
                        shutil.rmtree(os.path.join(sync_path, 'oxts'))
                        shutil.rmtree(os.path.join(sync_path, 'velodyne_points'))
                    except Exception as e:
                        print(e)
                        continue

def split_train_val(root_dir, kitti_raw_list, train_list, image_train_dir, val_list, image_val_dir):
    for kitti_raw in tqdm(kitti_raw_list):
        # train_list에 해당하는 폴더를 train 디렉토리로 이동
        for folder in train_list:
            source_folder_path = os.path.join(root_dir, kitti_raw, folder)
            if os.path.isdir(source_folder_path):
                shutil.move(source_folder_path, image_train_dir)

        # val_list에 해당하는 폴더를 val 디렉토리로 이동
        for folder in val_list:
            source_folder_path = os.path.join(root_dir, kitti_raw, folder)
            if os.path.isdir(source_folder_path):
                shutil.move(source_folder_path, image_val_dir)
        shutil.rmtree(os.path.join(root_dir, kitti_raw))

def get_last_file_name(directory):
    file_names = os.listdir(directory)
    file_names.sort()

    return file_names[-1] if file_names else None

def get_last_5(dir_path):
    last_5 = []
    last_file_name = get_last_file_name(dir_path)
    last_file_int = int(last_file_name[:-4])

    for i in range(5):
        file_name = last_file_int - i
        file = str(file_name).zfill(10) + '.png'
        last_5.append(file)

    return last_5

def remove_first_last_5(root_dir, image_train_dir, image_val_dir):
    train_dir = os.path.join(root_dir, 'kitti_raw/train')
    val_dir = os.path.join(root_dir, 'kitti_raw/val')

    train_list = os.listdir(train_dir)
    val_list = os.listdir(val_dir)
    mode_list = [train_list, val_list]

    print("\n\nRemove first and last 5 files:\n")
    for mode in mode_list:
        for dir in mode:
            if mode == train_list:
                current_dir = os.path.join(root_dir, os.path.join(image_train_dir, dir))
            elif mode == val_list:
                current_dir = os.path.join(root_dir, os.path.join(image_val_dir, dir))
            for img in img_dir:
                try:
                    dir_path = os.path.join(current_dir, img)
                    files = os.listdir(dir_path)

                    # 파일이 있는 경우에만 처리
                    if files:
                        # 첫 5개 파일 삭제
                        for file in first_5:  # 첫 5개 파일
                            file_path = os.path.join(dir_path, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            except Exception as e:
                                print(e)
                        last_5 = get_last_5(dir_path)
                        for file in last_5:  # 첫 5개 파일
                            file_path = os.path.join(dir_path, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            except Exception as e:
                                print(e)

                    else:
                        print("Directory is Empty.")
                except Exception as e:
                    print(e)
                    pass

'''def function():
    # remove original empty directory
    for dirname in kitti_raw_list:
        try:
            shutil.rmtree(os.path.join(root_dir, dirname))
        except Exception as e:
            print(e)
            pass

    img_dir_src = ['image_02/data/', 'image_03/data/']
    img_dir_dst = ['proj_depth/raw_image/image_02/', 'proj_depth/raw_image/image_03/']

    for dir in train_list:
        current_dir = os.path.join(root_dir, os.path.join(image_train_dir, dir))
        # print(current_dir)
        for img_s, img_d in img_dir_src, img_dir_dst:
            try:
                src_path = os.path.join(current_dir, img_s)
                dst_path = os.path.join(current_dir, img_d)
                print(dst_path)
                shutil.copy(src_path, dst_path)

            except Exception as e:
                print(e)
                pass



    src_base_path = "/home/research1/Desktop/gachon/SSDC/datasets/kitti_raw/val"
    dest_base_path = "/home/research1/Desktop/gachon/SSDC/datasets/kitti_raw/val"

    #move_files_from_multiple_folders(src_base_path, dest_base_path)


    # Root directories
    root_dirs = [
        "datasets/kitti_raw/val/",
        "datasets/data_depth_annotated/val/",
        "datasets/data_depth_velodyne/val/",
        "datasets/pseudo_depth_map/val/",
        "datasets/pseudo_gt_map/val/"
    ]

    # Folders to move
    folders_to_move = ["image_02", "image_03"]

    for root_dir in root_dirs:
        # List all subdirectories
        sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for sub_dir in sub_dirs:
            # Construct paths to the projection directories
            proj_depth_dir = os.path.join(root_dir, sub_dir, "proj_depth")

            if os.path.exists(proj_depth_dir):
                sub_proj_dirs = [d for d in os.listdir(proj_depth_dir) if
                                 os.path.isdir(os.path.join(proj_depth_dir, d))]

                for sub_proj_dir in sub_proj_dirs:
                    base_dir = os.path.join(proj_depth_dir, sub_proj_dir)
                    target_dir = proj_depth_dir

                    for folder in folders_to_move:
                        src_folder = os.path.join(base_dir, folder)
                        dst_folder = os.path.join(target_dir, folder)

                        if os.path.exists(src_folder):
                            shutil.move(src_folder, dst_folder)
                            print(f"Moved {src_folder} to {dst_folder}")
                        else:
                            print(f"{src_folder} does not exist.")
            else:
                print(f"{proj_depth_dir} does not exist.")'''


def image_preprocessing(root_dir):
    train_dir = os.path.join(root_dir, 'data_depth_annotated/train')
    val_dir = os.path.join(root_dir, 'data_depth_annotated/val')

    train_list = os.listdir(train_dir)
    val_list = os.listdir(val_dir)

    image_train_dir = os.path.join(root_dir, 'kitti_raw/train')
    image_val_dir = os.path.join(root_dir, 'kitti_raw/val')

    kitti_raw_list = ['kitti_raw/2011_09_26',
                      'kitti_raw/2011_09_28',
                      'kitti_raw/2011_09_29',
                      'kitti_raw/2011_09_30',
                      'kitti_raw/2011_10_03']

    print("remove unused files")
    remove_unused_files(root_dir, kitti_raw_list)

    print("Split train add val:\n")
    split_train_val(root_dir, kitti_raw_list, train_list, image_train_dir, val_list, image_val_dir)

    remove_first_last_5(root_dir, image_train_dir, image_val_dir)


    # trainX 26: 35 / 28: 21 ~ / 29:

