import os
import torch.utils.data as data
import random
import skimage
import skimage.io
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_transform():
    t_list = [
        transforms.ToTensor(),
    ]
    return transforms.Compose(t_list)


def default_loader(path):
    img = skimage.io.imread(path)
    return img


def point_loader(path):
    img = skimage.io.imread(path)
    depth = img *1.0 / 256.0
    depth = np.reshape(depth, [img.shape[0], img.shape[1], 1]).astype(np.float32)
    return depth


class myImageFloder(data.Dataset):
    def __init__(self, raw_image, raw_lidar, gt_lidar, pseudo_depth_map, pseudo_gt_map, training,
                 loader=default_loader, ploader = point_loader):
        self.raw_image = raw_image
        self.gt_lidar = gt_lidar
        self.raw_lidar = raw_lidar
        self.pseudo_depth_map = pseudo_depth_map
        self.pseudo_gt_map = pseudo_gt_map

        self.loader = loader
        self.gtploader = ploader
        self.ploader = ploader 
        self.training = training

    def __getitem__(self, index):
        raw_image = self.raw_image[index]
        gt_lidar = self.gt_lidar[index]
        raw_lidar = self.raw_lidar[index]
        pseudo_depth_map = self.pseudo_depth_map[index]
        pseudo_gt_map = self.pseudo_gt_map[index]

        raw_image = self.loader(raw_image)
        h, w, c = raw_image.shape

        # shape of raw_image: (375, 1242, 3)

        # data load & normalization
        gt_lidar = self.gtploader(gt_lidar)
        raw_lidar = self.ploader(raw_lidar)
        pseudo_depth_map = self.ploader(pseudo_depth_map)
        pseudo_gt_map = self.ploader(pseudo_gt_map)
        
        # shape of loaded data: gt_lidar (375, 1242, 1) | raw_lidar (375, 1242, 1) | pseudo_depth_map (375, 1242, 1) | pseduo_gt_map (375, 1242, 1)

        # random crop
        th, tw = 256,512
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        raw_image = raw_image[y1:y1 + th, x1:x1 + tw, :]
        gt_lidar = gt_lidar[y1:y1 + th, x1:x1 + tw,:]
        raw_lidar = raw_lidar[y1:y1 + th, x1:x1 + tw, :]
        pseudo_depth_map = pseudo_depth_map[y1:y1 + th, x1:x1 + tw, :]
        pseudo_gt_map = pseudo_gt_map[y1:y1 + th, x1:x1 + tw, :]
        processed = get_transform()

        # shape of cropped data: raw_image (256, 512, 3) | gt_lidar (256, 512, 1) | raw_lidar (256, 512, 1) | pseudo_depth_map (256, 512, 1) | pseduo_gt_map (256, 512, 1)
        
        # convert to tensor
        raw_image = processed(raw_image)
        gt_lidar = processed(gt_lidar)
        raw_lidar = processed(raw_lidar)
        pseudo_depth_map = processed(pseudo_depth_map)
        pseudo_gt_map = processed(pseudo_gt_map)

        # shape of transformed data: raw_image torch.Size([3, 256, 512]) | gt_lidar torch.Size([1, 256, 512]) | raw_lidar torch.Size([1, 256, 512]) | pseudo_depth_map torch.Size([1, 256, 512]) | pseduo_gt_map torch.Size([1, 256, 512])

        return raw_image, gt_lidar, raw_lidar, pseudo_depth_map, pseudo_gt_map

    def __len__(self):
        return len(self.raw_image)


if __name__ == '__main__':
    print("")