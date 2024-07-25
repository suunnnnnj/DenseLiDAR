#data loader
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
            self.annotated_paths = self._get_file_paths(os.path.join(root_dir, 'data_depth_annotated', mode))
            self.velodyne_paths = self._get_file_paths(os.path.join(root_dir, 'data_depth_velodyne', mode))
            self.pseudo_depth_map = self._get_file_paths(os.path.join(root_dir, 'pseudo_depth_map', mode))
            self.pseudo_gt_map = self._get_file_paths(os.path.join(root_dir, 'pseudo_gt_map', mode))

            # 공통된 파일 경로 유지하도록 경로 필터링
            common_files = self._filter_common_files(mode)

            print(f"Common files: {len(common_files)}")

            self.raw_paths = [path for path in self.raw_paths if self._get_relative_path(path, mode) in common_files]
            self.annotated_paths = [path for path in self.annotated_paths if self._get_relative_path(path, mode) in common_files]
            self.velodyne_paths = [path for path in self.velodyne_paths if self._get_relative_path(path, mode) in common_files]
            self.pseudo_depth_map = [path for path in self.pseudo_depth_map if self._get_relative_path(path, mode) in common_files]
            self.pseudo_gt_map = [path for path in self.pseudo_gt_map if self._get_relative_path(path, mode) in common_files]

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

    def _get_relative_path(self, path, base_folder):
        parts = path.split('/')
        try:
            base_idx = parts.index(base_folder)
            return '/'.join(parts[base_idx:])
        except ValueError:
            return None

    def _filter_common_files(self, mode):
        raw_relative_paths = set(self._get_relative_path(path, mode) for path in self.raw_paths)
        annotated_relative_paths = set(self._get_relative_path(path, mode) for path in self.annotated_paths)
        velodyne_relative_paths = set(self._get_relative_path(path, mode) for path in self.velodyne_paths)
        pseudo_depth_map_relative_paths = set(self._get_relative_path(path, mode) for path in self.pseudo_depth_map)
        pseudo_gt_map_relative_paths = set(self._get_relative_path(path, mode) for path in self.pseudo_gt_map)

        # None 값을 제거
        raw_relative_paths.discard(None)
        annotated_relative_paths.discard(None)
        velodyne_relative_paths.discard(None)
        pseudo_depth_map_relative_paths.discard(None)
        pseudo_gt_map_relative_paths.discard(None)

        return raw_relative_paths & annotated_relative_paths & velodyne_relative_paths & pseudo_depth_map_relative_paths & pseudo_gt_map_relative_paths

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