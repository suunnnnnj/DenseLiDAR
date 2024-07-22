import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms

class KITTIDepthDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the dataset folders.
            mode (string): 'train', 'val', or 'test' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.resize_transform = transforms.Resize((256, 512), interpolation=InterpolationMode.BICUBIC, antialias=True)

        if mode in ['train', 'val']:
            self.annotated_paths = self._get_file_paths(os.path.join(root_dir, 'data_depth_annotated', mode))
            self.velodyne_paths = self._get_file_paths(os.path.join(root_dir, 'data_depth_velodyne', mode))
            self.pseudo_depth_map = self._get_file_paths(os.path.join(root_dir, 'pseudo_depth_map', mode))
            self.pseudo_gt_map = self._get_file_paths(os.path.join(root_dir, 'pseudo_gt_map', mode))
            self.raw_paths = self._get_raw_file_paths(os.path.join(root_dir, 'kitti_raw'), self.annotated_paths)
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

    def _get_raw_file_paths(self, raw_dir, reference_paths):
        raw_file_paths = []
        reference_files = set(os.path.basename(path) for path in reference_paths)
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file in reference_files:
                    raw_file_paths.append(os.path.join(root, file))
        return sorted(raw_file_paths)

    def __len__(self):
        if self.mode in ['train', 'val']:
            return min(len(self.annotated_paths), len(self.velodyne_paths), len(self.pseudo_depth_map), len(self.pseudo_gt_map), len(self.raw_paths))
        elif self.mode == 'test':
            return min(len(self.test_image_paths), len(self.test_velodyne_paths), len(self.test_depth_path), len(self.test_radar_paths))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode in ['train', 'val']:
            annotated_img_path = self.annotated_paths[idx]
            velodyne_img_path = self.velodyne_paths[idx]
            pseudo_depth_map = self.pseudo_depth_map[idx]
            pseudo_gt_map = self.pseudo_gt_map[idx]
            raw_img_path = self.raw_paths[idx]

            annotated_image = Image.open(annotated_img_path).convert('L')
            velodyne_image = Image.open(velodyne_img_path).convert('L')
            pseudo_depth_map = Image.open(pseudo_depth_map).convert('L')
            pseudo_gt_map = Image.open(pseudo_gt_map).convert('L')
            raw_image = Image.open(raw_img_path).convert('RGB')

            # Resize images
            annotated_image = self.resize_transform(annotated_image)
            velodyne_image = self.resize_transform(velodyne_image)
            pseudo_depth_map = self.resize_transform(pseudo_depth_map)
            pseudo_gt_map = self.resize_transform(pseudo_gt_map)
            raw_image = self.resize_transform(raw_image)
    	
            sample = {
                'annotated_image': annotated_image,
                'velodyne_image': velodyne_image,
                'pseudo_depth_map': pseudo_depth_map,
                'pseudo_gt_map': pseudo_gt_map,
                'raw_image': raw_image
            }
            if self.transform:
                sample = self.transform(sample)

            return sample

        elif self.mode == 'test':
            test_image_path = self.test_image_paths[idx]
            test_velodyne_path = self.test_velodyne_paths[idx]
            test_depth_path = self.test_depth_path[idx]

            test_image = Image.open(test_image_path).convert('RGB')
            test_velodyne_image = Image.open(test_velodyne_path)
            test_depth_image = Image.open(test_depth_path)

            # Resize images
            test_image = self.resize_transform(test_image)
            test_velodyne_image = self.resize_transform(test_velodyne_image)
            test_depth_image = self.resize_transform(test_depth_image)

            sample = {
                'test_image': test_image,
                'test_velodyne_image': test_velodyne_image,
                'test_depth_image': test_depth_image,
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if 'annotated_image' in sample:
            annotated_image = sample['annotated_image']
            velodyne_image = sample['velodyne_image']
            pseudo_depth_map = sample['pseudo_depth_map']
            pseudo_gt_map = sample['pseudo_gt_map']
            raw_image = sample['raw_image']
            return {
                'annotated_image': transforms.ToTensor()(annotated_image),
                'velodyne_image': transforms.ToTensor()(velodyne_image),
                'pseudo_depth_map': transforms.ToTensor()(pseudo_depth_map),
                'pseudo_gt_map': transforms.ToTensor()(pseudo_gt_map),
                'raw_image': transforms.ToTensor()(raw_image)
            }
        else:
            test_image = sample['test_image']
            test_velodyne_image = sample['test_velodyne_image']
            test_depth_image = sample['test_depth_image']
            return {
                'test_image': transforms.ToTensor()(test_image),
                'test_velodyne_image': transforms.ToTensor()(test_velodyne_image),
                'test_depth_image': transforms.ToTensor()(test_depth_image),
            }