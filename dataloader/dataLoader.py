import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class KITTIDepthDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): 모든 데이터셋 폴더가 있는 디렉터리.
            mode (string): 'train', 'val', 'test' 중 데이터셋 분할을 지정.
            transform (callable, optional): 샘플에 적용할 선택적 변환 함수.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.resize_shape = (512, 256)

        if mode in ['train', 'val']:
            self.raw_paths = self._get_file_paths(os.path.join(root_dir, 'kitti_raw', mode))
            self.annotated_paths, self.velodyne_paths, self.pseudo_depth_map, self.pseudo_gt_map = self._get_corresponding_paths(root_dir, self.raw_paths, mode)

            # 경로 확인
            print(f"Total raw paths: {len(self.raw_paths)}")
            print(f"Total annotated paths: {len(self.annotated_paths)}")
            print(f"Total velodyne paths: {len(self.velodyne_paths)}")
            print(f"Total pseudo depth map paths: {len(self.pseudo_depth_map)}")
            print(f"Total pseudo gt map paths: {len(self.pseudo_gt_map)}")

        elif mode == 'test':
            self.test_image_paths = self._get_file_paths(
                os.path.join(root_dir, 'data_depth_selection', 'depth_selection', 'test_depth_completion_anonymous', 'image'))
            self.test_velodyne_paths = self._get_file_paths(
                os.path.join(root_dir, 'data_depth_selection','depth_selection', 'test_depth_completion_anonymous', 'velodyne_raw'))
            self.test_depth_path = self._get_file_paths(
                os.path.join(root_dir, 'data_depth_selection','depth_selection', 'test_depth_completion_anonymous', 'lidar_raw'))
            print(f"Loaded {len(self.test_image_paths)} test image files, {len(self.test_velodyne_paths)} test velodyne files, {len(self.test_depth_path)} test depth files.")
        else:
            raise ValueError("Mode should be 'train', 'val', or 'test'")

    def _get_file_paths(self, dir_path):
        file_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    file_paths.append(os.path.join(root, file))
        return sorted(file_paths)

    def _get_corresponding_paths(self, root_dir, raw_paths, mode):
        annotated_paths = []
        velodyne_paths = []
        pseudo_depth_map = []
        pseudo_gt_map = []

        for raw_path in raw_paths:
            filename = os.path.basename(raw_path)
            date_drive = self._get_date_drive_from_path(raw_path)
            annotated_path = self._find_file_in_subdirs(os.path.join(root_dir, 'data_depth_annotated', mode), filename)
            velodyne_path = self._find_file_in_subdirs(os.path.join(root_dir, 'data_depth_velodyne', mode), filename)
            pseudo_depth_map_path = self._find_file_in_subdirs(os.path.join(root_dir, 'pseudo_depth_map', mode), filename)
            pseudo_gt_map_path = self._find_file_in_subdirs(os.path.join(root_dir, 'pseudo_gt_map', mode), filename)

            # 디버깅 출력
            print(f"Checking paths for {filename}:")
            print(f"Annotated path: {annotated_path}, Exists: {annotated_path is not None}")
            print(f"Velodyne path: {velodyne_path}, Exists: {velodyne_path is not None}")
            print(f"Pseudo depth map path: {pseudo_depth_map_path}, Exists: {pseudo_depth_map_path is not None}")
            print(f"Pseudo GT map path: {pseudo_gt_map_path}, Exists: {pseudo_gt_map_path is not None}")

            if annotated_path and velodyne_path and pseudo_depth_map_path and pseudo_gt_map_path:
                annotated_paths.append(annotated_path)
                velodyne_paths.append(velodyne_path)
                pseudo_depth_map.append(pseudo_depth_map_path)
                pseudo_gt_map.append(pseudo_gt_map_path)

        return annotated_paths, velodyne_paths, pseudo_depth_map, pseudo_gt_map

    def _get_date_drive_from_path(self, path):
        # Assuming the date_drive is the parent folder of the _sync folder
        return os.path.basename(os.path.dirname(path))

    def _find_file_in_subdirs(self, root_dir, filename):
        for root, _, files in os.walk(root_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def __len__(self):
        if self.mode in ['train', 'val']:
            return len(self.raw_paths)
        elif self.mode == 'test':
            return min(len(self.test_image_paths), len(self.test_velodyne_paths), len(self.test_depth_path))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode in ['train', 'val']:
            if idx >= len(self.annotated_paths):
                raise IndexError(f"Index {idx} out of range for dataset with length {len(self.annotated_paths)}")

            raw_img_path = self.raw_paths[idx]
            annotated_img_path = self.annotated_paths[idx]
            velodyne_img_path = self.velodyne_paths[idx]
            pseudo_depth_map_path = self.pseudo_depth_map[idx]
            pseudo_gt_map_path = self.pseudo_gt_map[idx]

            print(f"Loading raw image from: {raw_img_path}")
            print(f"Loading annotated image from: {annotated_img_path}")
            print(f"Loading velodyne image from: {velodyne_img_path}")
            print(f"Loading pseudo depth map from: {pseudo_depth_map_path}")
            print(f"Loading pseudo ground truth map from: {pseudo_gt_map_path}")

            raw_image = cv2.imread(raw_img_path)
            annotated_image = cv2.imread(annotated_img_path, cv2.IMREAD_ANYDEPTH)
            velodyne_image = cv2.imread(velodyne_img_path, cv2.IMREAD_ANYDEPTH)
            pseudo_depth_map = cv2.imread(pseudo_depth_map_path, cv2.IMREAD_ANYDEPTH)
            pseudo_gt_map = cv2.imread(pseudo_gt_map_path, cv2.IMREAD_ANYDEPTH)

            # 이미지 크기 조정
            raw_image = cv2.resize(raw_image, self.resize_shape, interpolation=cv2.INTER_CUBIC)
            annotated_image = cv2.resize(annotated_image, self.resize_shape, interpolation=cv2.INTER_CUBIC)
            velodyne_image = cv2.resize(velodyne_image, self.resize_shape, interpolation=cv2.INTER_CUBIC)
            pseudo_depth_map = cv2.resize(pseudo_depth_map, self.resize_shape, interpolation=cv2.INTER_CUBIC)
            pseudo_gt_map = cv2.resize(pseudo_gt_map, self.resize_shape, interpolation=cv2.INTER_CUBIC)

            # 이미지 정규화
            raw_image = raw_image / 256.0
            annotated_image = annotated_image / 256.0
            velodyne_image = velodyne_image / 256.0
            pseudo_depth_map = pseudo_depth_map / 256.0
            pseudo_gt_map = pseudo_gt_map / 256.0

            annotated_image = normalize_non_zero_pixels(annotated_image)
            velodyne_image = normalize_non_zero_pixels(velodyne_image)
            pseudo_depth_map = normalize_non_zero_pixels(pseudo_depth_map)
            pseudo_gt_map = normalize_non_zero_pixels(pseudo_gt_map)
    	
            sample = {
                'raw_image': raw_image,
                'annotated_image': annotated_image,
                'velodyne_image': velodyne_image,
                'pseudo_depth_map': pseudo_depth_map,
                'pseudo_gt_map': pseudo_gt_map
            }
            if self.transform:
                sample = self.transform(sample)

            return sample

        elif self.mode == 'test':
            if idx >= len(self.test_image_paths):
                raise IndexError(f"Index {idx} out of range for test dataset with length {len(self.test_image_paths)}")

            test_velodyne_path = self.test_velodyne_paths[idx]
            test_depth_path = self.test_depth_path[idx]
            test_image_path = self.test_image_paths[idx]

            test_velodyne_image = cv2.imread(test_velodyne_path, cv2.IMREAD_ANYDEPTH)
            test_depth_image = cv2.imread(test_depth_path, cv2.IMREAD_ANYDEPTH)
            test_image = cv2.imread(test_image_path)

            # 이미지 크기 조정
            test_velodyne_image = cv2.resize(test_velodyne_image, self.resize_shape, interpolation=cv2.INTER_CUBIC)
            test_depth_image = cv2.resize(test_depth_image, self.resize_shape, interpolation=cv2.INTER_CUBIC)
            test_image = cv2.resize(test_image, self.resize_shape, interpolation=cv2.INTER_CUBIC)

            # 이미지 정규화
            test_velodyne_image = test_velodyne_image / 256.0
            test_depth_image = test_depth_image / 256.0
            test_image = test_image / 256.0

            test_velodyne_image = normalize_non_zero_pixels(test_velodyne_image)
            test_depth_image = normalize_non_zero_pixels(test_depth_image)
            
            sample = {
                'test_velodyne_image': test_velodyne_image,
                'test_depth_image': test_depth_image,
                'test_image': test_image
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

class ToTensor(object):
    """샘플의 ndarray를 텐서로 변환."""

    def __call__(self, sample):
        if 'annotated_image' in sample:
            raw_image = sample['raw_image']
            annotated_image = sample['annotated_image']
            velodyne_image = sample['velodyne_image']
            pseudo_depth_map = sample['pseudo_depth_map']
            pseudo_gt_map = sample['pseudo_gt_map']

            return {
                'raw_image': torch.tensor(raw_image, dtype=torch.float32).permute(2, 0, 1),
                'annotated_image': torch.tensor(annotated_image, dtype=torch.float32).unsqueeze(0),
                'velodyne_image': torch.tensor(velodyne_image, dtype=torch.float32).unsqueeze(0),
                'pseudo_depth_map': torch.tensor(pseudo_depth_map, dtype=torch.float32).unsqueeze(0),
                'pseudo_gt_map': torch.tensor(pseudo_gt_map, dtype=torch.float32).unsqueeze(0)
            }
        else:
            test_velodyne_image = sample['test_velodyne_image']
            test_depth_image = sample['test_depth_image']
            test_image = sample['test_image']

            return {
                'test_velodyne_image': torch.tensor(test_velodyne_image, dtype=torch.float32).unsqueeze(0),
                'test_depth_image': torch.tensor(test_depth_image, dtype=torch.float32).unsqueeze(0),
                'test_image': torch.tensor(test_image, dtype=torch.float32).permute(2, 0, 1)
            }

def normalize_non_zero_pixels(pixels):
    non_zero_mask = (pixels != 0)
    non_zero_pixels = pixels[non_zero_mask]

    if non_zero_pixels.size == 0:  # 모든 픽셀이 0인 경우
        return pixels.astype(np.float32)
    
    normalized_pixels = (non_zero_pixels - np.min(non_zero_pixels)) / (np.max(non_zero_pixels) - np.min(non_zero_pixels))
    
    result = np.zeros_like(pixels, dtype=np.float32)
    result[non_zero_mask] = normalized_pixels

    return result
