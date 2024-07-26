import os
import torch.utils.data as data
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import skimage
import skimage.io
import numpy as np
import torchvision.transforms as transforms


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Preprocess
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
INSTICS = {"2011_09_26": [721.5377, 596.5593, 149.854],
           "2011_09_28": [707.0493, 604.0814, 162.5066],
           "2011_09_29": [718.3351, 600.3891, 159.5122],
           "2011_09_30": [707.0912, 601.8873, 165.1104],
           "2011_10_03": [718.856, 607.1928, 161.2157]
}
# INSTICS = {"NYU": [582.6245, 313.0448, 238.4439]}
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
    ]

    return transforms.Compose(t_list)

def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    img = skimage.io.imread(path)
    return img

def point_loader(path):
    img = skimage.io.imread(path)
    depth = img *1.0 / 256.0
    depth = np.reshape(depth, [img.shape[0], img.shape[1], 1]).astype(np.float32)
    return depth

class myImageFloder(data.Dataset):
    def __init__(self, raw_image,law_lidar,gt_lidar,pseudo_depth,gt_depth,training,
                 loader=default_loader, ploader = point_loader):
        self.raw_image = raw_image
        self.law_lidar = law_lidar
        self.gt_lidar = gt_lidar
        self.pseudo_depth = pseudo_depth
        self.gt_depth = gt_depth
        self.loader = loader
        self.gtploader = ploader
        self.ploader = ploader 
        self.training = training

    def __getitem__(self, index):
        raw_image = self.raw_image[index]
        gt_lidar = self.gt_lidar[index]
        law_lidar = self.law_lidar[index]
        pseudo_depth = self.pseudo_depth[index]
        gt_depth = self.gt_depth[index]

        raw_image = self.loader(raw_image)  #raw image shape: (375, 1242, 3)
        h,w,c= raw_image.shape

        # dataload & normalization
        gt_point = self.gtploader(gt_lidar)
        law_lidar = self.ploader(law_lidar)
        pseudo_depth = self.ploader(pseudo_depth)
        gt_depth = self.ploader(gt_depth)
        
        # Data loaded: gt_point (375, 1242, 1), law_lidar (375, 1242, 1), pseudo_depth (375, 1242, 1), gt_depth (375, 1242, 1)

        #random crop
        th, tw = 256,512
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        raw_image = raw_image[y1:y1 + th, x1:x1 + tw, :]
        gt_point = gt_point[y1:y1 + th, x1:x1 + tw,:]
        law_lidar = law_lidar[y1:y1 + th, x1:x1 + tw, :]
        pseudo_depth = pseudo_depth[y1:y1 + th, x1:x1 + tw, :]
        gt_depth = gt_depth[y1:y1 + th, x1:x1 + tw, :]
        processed = get_transform(augment=False)

        # Cropped data shapes: raw_image (256, 512, 3), gt_point (256, 512, 1), law_lidar (256, 512, 1), pseudo_depth (256, 512, 1), gt_depth (256, 512, 1)
        
        # ToTensor
        raw_image = processed(raw_image)
        law_lidar = processed(law_lidar)
        pseudo_depth = processed(pseudo_depth)
        gt_depth = processed(gt_depth)
        gt_point = processed(gt_point)

        # Data transformed: raw_image torch.Size([3, 256, 512]), law_lidar torch.Size([1, 256, 512]), pseudo_depth torch.Size([1, 256, 512]), gt_depth torch.Size([1, 256, 512]), gt_point torch.Size([1, 256, 512])

        return raw_image,gt_point,law_lidar,pseudo_depth,gt_depth

    def __len__(self):
        return len(self.raw_image)

if __name__ == '__main__':
    print("")