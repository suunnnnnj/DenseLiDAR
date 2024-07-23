import os
import shutil
from tqdm import tqdm

root_dir = '/home/research1/Desktop/gachon/SSDC/datasets'


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

print("Split train add val:\n")
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

            # trainX 26: 35 / 28: 21 ~ / 29:

# remove original empty directory
for dirname in kitti_raw_list:
    try:
        shutil.rmtree(os.path.join(root_dir, dirname))
    except Exception as e:
        print(e)
        pass

img_dir = ['image_02/data/', 'image_03/data/']
first_5 = ['0000000000.png', '0000000001.png', '0000000002.png', '0000000003.png', '0000000004.png']

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

train_dir = os.path.join(root_dir, 'kitti_raw/train')
val_dir = os.path.join(root_dir, 'kitti_raw/val')

train_list = os.listdir(train_dir)
val_list = os.listdir(val_dir)

print("\n\nRemove first and last 5 files:\n")
for dir in tqdm(train_list):
    current_dir = os.path.join(root_dir, os.path.join(image_train_dir, dir))
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