"""
 Example usage:
 dataset = KITTIDepthDataset(root_dir='path/to/dataset', mode='train', transform=ToTensor())
 dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

 for batch in dataloader:
     print(batch)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import transform
import torchvision.transforms as transforms

class KITTIDepthDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, target_size=(256, 512)):
        """
        Args:
            root_dir (string): Directory with all the dataset folders.
            mode (string): 'train', 'val', or 'test' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Desired output size of the images (height, width).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size

        if mode in ['train', 'val']:
            self.annotated_paths = self._get_file_paths(os.path.join(root_dir, 'data_depth_annotated', mode))
            self.velodyne_paths = self._get_file_paths(os.path.join(root_dir, 'data_depth_velodyne', mode))
            self.raw_paths = self._get_raw_file_paths(os.path.join(root_dir, 'kitti_raw'), self.annotated_paths)
            self.data = self._load_data(self.annotated_paths, self.velodyne_paths, self.raw_paths)
        elif mode == 'test':
            self.test_image_paths = self._get_file_paths(
                os.path.join(root_dir, 'data_depth_selection', 'depth_selection', 'test_depth_completion_anonymous', 'image'))
            self.test_velodyne_paths = self._get_file_paths(
                os.path.join(root_dir, 'data_depth_selection','depth_selection', 'test_depth_completion_anonymous', 'velodyne_raw'))
            self.data = self._load_test_data(self.test_image_paths, self.test_velodyne_paths)
            print(f"Loaded {len(self.test_image_paths)} test image files and {len(self.test_velodyne_paths)} test velodyne files.")
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

    def _load_data(self, annotated_paths, velodyne_paths, raw_paths):
        data = []
        for i in range(len(annotated_paths)):
            annotated_image = Image.open(annotated_paths[i]).convert('RGB')
            velodyne_image = Image.open(velodyne_paths[i]).convert('L')
            raw_image = Image.open(raw_paths[i]).convert('RGB')

            # Resize images using skimage
            annotated_image = transform.resize(np.array(annotated_image), self.target_size, anti_aliasing=True)
            velodyne_image = transform.resize(np.array(velodyne_image), self.target_size, anti_aliasing=True)
            raw_image = transform.resize(np.array(raw_image), self.target_size, anti_aliasing=True)

            # Convert back to PIL Image
            annotated_image = Image.fromarray((annotated_image * 255).astype(np.uint8))
            velodyne_image = Image.fromarray((velodyne_image * 255).astype(np.uint8))
            raw_image = Image.fromarray((raw_image * 255).astype(np.uint8))

            sample = {
                'annotated_image': annotated_image,
                'velodyne_image': velodyne_image,
                'raw_image': raw_image
            }
            data.append(sample)
        return data

    def _load_test_data(self, test_image_paths, test_velodyne_paths):
        data = []
        for i in range(len(test_image_paths)):
            test_image = Image.open(test_image_paths[i]).convert('RGB')
            test_velodyne_image = Image.open(test_velodyne_paths[i]).convert('L')

            # Resize images using skimage
            test_image = transform.resize(np.array(test_image), self.target_size, anti_aliasing=True)
            test_velodyne_image = transform.resize(np.array(test_velodyne_image), self.target_size, anti_aliasing=True)

            # Convert back to PIL Image
            test_image = Image.fromarray((test_image * 255).astype(np.uint8))
            test_velodyne_image = Image.fromarray((test_velodyne_image * 255).astype(np.uint8))

            sample = {
                'test_image': test_image,
                'test_velodyne_image': test_velodyne_image
            }
            data.append(sample)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

# Transformations (if any)
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if 'annotated_image' in sample:
            annotated_image, velodyne_image, raw_image = sample['annotated_image'], sample['velodyne_image'], sample['raw_image']
            return {
                'annotated_image': transforms.ToTensor()(annotated_image),
                'velodyne_image': transforms.ToTensor()(velodyne_image),
                'raw_image': transforms.ToTensor()(raw_image)
            }
        else:
            test_image, test_velodyne_image = sample['test_image'], sample['test_velodyne_image']
            return {
                'test_image': transforms.ToTensor()(test_image),
                'test_velodyne_image': transforms.ToTensor()(test_velodyne_image)
            }


