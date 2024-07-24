import os
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.dirname(ROOT_DIR)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    images = []
    lidars = []
    depths = []
    pseudo = []
    dense = []

    temp = filepath
    filepath_ri = temp + 'kitti_raw/train'
    filepath_gtl = temp + 'data_depth_annotated/train'
    filepath_r1 = temp + 'data_depth_velodyne/train'
    filepath_pd = temp + 'pseudo_depth_map/train'
    filepath_dd = temp + 'pseudo_gt_map/train'

    seqs = [seq for seq in os.listdir(filepath_ri) if seq.find('sync') > -1]
    left_fold = 'proj_depth/image_02'
    right_fold = 'proj_depth/image_03'
    
    for seq in seqs:
        temp = os.path.join(temp, seq)
        #raw image
        imgsl = os.path.join(filepath_ri, seq, left_fold)
        imagel = [os.path.join(imgsl, img) for img in os.listdir(imgsl)]
        imagel.sort()
        images = np.append(images, imagel)
        imgsr = os.path.join(filepath_ri, seq, right_fold)
        imager = [os.path.join(imgsr, img) for img in os.listdir(imgsr)]
        imager.sort()
        images = np.append(images, imager)
        #raw lidar
        lids2l = os.path.join(filepath_r1, seq, left_fold)
        lidar2l = [os.path.join(lids2l, img) for img in os.listdir(lids2l)]
        lidar2l.sort()
        lidars = np.append(lidars, lidar2l)
        lids2r = os.path.join(filepath_ri, seq, right_fold)
        lidar2r = [os.path.join(lids2r, img) for img in os.listdir(lids2r)]
        lidar2r.sort()
        lidars = np.append(lidars, lidar2r)
        #ground truth lidar
        gt_lidar = os.path.join(filepath_gtl, seq, left_fold)
        gt_lidar = [os.path.join(gt_lidar, img) for img in os.listdir(gt_lidar)]
        gt_lidar.sort()
        depths = np.append(depths, gt_lidar)
        depsr = os.path.join(filepath_gtl, seq, right_fold)
        depthr = [os.path.join(depsr, img) for img in os.listdir(depsr)]
        depthr.sort()
        depths = np.append(depths, depthr)
        #pseudo depth map
        pseudo_dml = os.path.join(filepath_pd, seq, left_fold)
        pseudo_dml = [os.path.join(pseudo_dml, img) for img in os.listdir(pseudo_dml)]
        pseudo_dml.sort()
        pseudo = np.append(pseudo, pseudo_dml)
        pseudo_dmr = os.path.join(filepath_pd, seq, right_fold)
        pseudo_dmr = [os.path.join(pseudo_dmr, img) for img in os.listdir(pseudo_dmr)]
        pseudo_dmr.sort()
        pseudo = np.append(pseudo, pseudo_dmr)
        #GT depth map
        dense_dml = os.path.join(filepath_dd, seq, left_fold)
        dense_dml = [os.path.join(dense_dml, img) for img in os.listdir(dense_dml)]
        dense_dml.sort()
        dense = np.append(dense, dense_dml)
        dense_dmr = os.path.join(filepath_dd, seq, right_fold)
        dense_dmr = [os.path.join(dense_dmr, img) for img in os.listdir(dense_dmr)]
        dense_dmr.sort()
        dense = np.append(dense, dense_dmr)
    left_train = images
    lidar2_train = lidars
    depth_train = depths
    pseudo_train = pseudo
    dense_train = dense

    return left_train,lidar2_train,depth_train,pseudo_train, dense_train


if __name__ == '__main__':
    datapath = ''