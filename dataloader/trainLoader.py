import os
import torch.utils.data as data
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import skimage
import skimage.io
import skimage.transform
import numpy as np
import torchvision.transforms as transforms


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

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
    def __init__(self, left,sparse,gtp,p_depth,gt_depth,training,   #raw image, raw lidar, gt lidar, pseudo depth, gt depth
                 loader=default_loader, ploader = point_loader):
        self.left = left
        self.sparse = sparse
        self.gtp = gtp
        self.p_depth = p_depth
        self.gt_depth = gt_depth
        self.loader = loader
        self.gtploader = ploader #inloader => gtploader gt lidar
        self.ploader = ploader #sloader => ploader raw lidar
        self.training = training
    def __getitem__(self, index):
        left = self.left[index]
        gtp = self.gtp[index]
        sparse = self.sparse[index]
        p_depth = self.p_depth[index]
        gt_depth = self.gt_depth[index]

        left_img = self.loader(left)
        #Left image loaded with shape: (375, 1242, 3) from path: sample/kitti_raw/train/2011_09_26_drive_0001_sync/proj_depth/image_02/0000000002.png

        index_str = self.left[index].split('/')[-4][0:10]
        params_t = INSTICS[index_str]
        params = np.ones((256,512,3),dtype=np.float32)
        params[:, :, 0] = params[:,:,0] * params_t[0]
        params[:, :, 1] = params[:, :, 1] * params_t[1]
        params[:, :, 2] = params[:, :, 2] * params_t[2]
        #Params initialized with shape: (256, 512, 3)

        h,w,c= left_img.shape
        # dataload 및 정규화
        gt_point = self.gtploader(gtp)
        sparse = self.ploader(sparse)
        p_depth = self.ploader(p_depth)
        gt_depth = self.ploader(gt_depth)
        # Data loaded: gt_point (375, 1242, 1), sparse (375, 1242, 1), p_depth (375, 1242, 1), gt_depth (375, 1242, 1)

        #random crop
        th, tw = 256,512
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        params = np.reshape(params, [256, 512, 3]).astype(np.float32)

        left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
        gt_point = gt_point[y1:y1 + th, x1:x1 + tw,:]
        sparse = sparse[y1:y1 + th, x1:x1 + tw, :]
        p_depth = p_depth[y1:y1 + th, x1:x1 + tw, :]
        gt_depth = gt_depth[y1:y1 + th, x1:x1 + tw, :]
        processed = get_transform(augment=False)
        # Cropped data shapes: left_img (256, 512, 3), gt_point (256, 512, 1), sparse (256, 512, 1), p_depth (256, 512, 1), gt_depth (256, 512, 1)
        # ToTensor
        left_img = processed(left_img)
        sparse = processed(sparse)
        p_depth = processed(p_depth)
        gt_depth = processed(gt_depth)
        gt_point = processed(gt_point)
        # Data transformed: left_img torch.Size([3, 256, 512]), sparse torch.Size([1, 256, 512]), p_depth torch.Size([1, 256, 512]), gt_depth torch.Size([1, 256, 512]), gt_point torch.Size([1, 256, 512])

        return left_img,gt_point,sparse,p_depth,gt_depth,params

    def __len__(self):
        return len(self.left)


if __name__ == '__main__':
    print("")