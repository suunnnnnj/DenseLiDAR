from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
from model import DenseLiDAR
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCompletion')
parser.add_argument('--loadmodel', default='',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = DenseLiDAR(1)
# model = nn.DataParallel(model, device_ids=[0])
model.cuda()

modelpath = os.path.join(ROOT_DIR, args.loadmodel)

if modelpath is not None:
    checkpoint = torch.load(modelpath, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())  # Check the structure of the checkpoint

    # Check if 'model_state_dict' directly exists in checkpoint or inside another key
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove unnecessary keys
    state_dict = {k.replace('DCU.', ''): v for k, v in state_dict.items() if k.startswith('DCU.')}

    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore unmatched keys

    print(f"Loaded pretrained model weights from {modelpath}.")


def test(imgL,sparse,pseudo_depth):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           sparse = torch.FloatTensor(sparse).cuda()
           pseudo_depth = torch.FloatTensor(pseudo_depth).cuda()

        imgL= Variable(imgL)
        sparse = Variable(sparse)
        pseudo_depth = Variable(pseudo_depth)

        device = "cuda"
        with torch.no_grad():
            dense_depth = model(imgL, sparse, pseudo_depth, device)

        dense_depth = torch.squeeze(dense_depth)

        return dense_depth.data.cpu().numpy()

def scale_crop():
    t_list = [
        transforms.ToTensor()
    ]

    return transforms.Compose(t_list)

def get_transform():
    return scale_crop()

def main():
   processed = get_transform()

   pseudo_depth_fold = 'demo_pseudo_depth.png'
   left_fold = 'demo_image.png'
   lidar2_raw ='demo_velodyne.png'
   
   imgL_o = skimage.io.imread(left_fold)
   imgL_o = np.reshape(imgL_o, [imgL_o.shape[0], imgL_o.shape[1],3])
   imgL = processed(imgL_o).numpy()
   imgL = np.reshape(imgL, [1, 3, imgL_o.shape[0], imgL_o.shape[1]])

   p_depth = skimage.io.imread(pseudo_depth_fold).astype(np.float32)
   p_depth = p_depth * 1.0 / 256.0
   p_depth = np.reshape(p_depth, [imgL_o.shape[0], imgL_o.shape[1], 1])
   p_depth = processed(p_depth).numpy()
   p_depth = np.reshape(p_depth, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])

   sparse = skimage.io.imread(lidar2_raw).astype(np.float32)
   sparse = sparse *1.0 / 256.0
   sparse = np.reshape(sparse, [imgL_o.shape[0], imgL_o.shape[1], 1])
   sparse = processed(sparse).numpy()
   sparse = np.reshape(sparse, [1, 1, imgL_o.shape[0], imgL_o.shape[1]])

   output1 = "dense_depth_output.png"

   pred = test(imgL, sparse, p_depth)
   pred = np.where(pred <= 0.0, 0.9, pred)

   pred_show = pred * 256.0
   pred_show = pred_show.astype('uint16')
   res_buffer = pred_show.tobytes()
   img = Image.new("I",pred_show.T.shape)
   img.frombytes(res_buffer,'raw',"I;16")
   img.save(output1)

if __name__ == '__main__':
   main()




