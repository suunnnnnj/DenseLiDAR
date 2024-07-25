from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import skimage
import skimage.io
import skimage.transform
import numpy as np
from model import DenseLiDAR
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='deepCompletion')
parser.add_argument('--loadmodel', default='', help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = DenseLiDAR(1)
# model = nn.DataParallel(model, device_ids=[0])
model.cuda()

model_path = os.path.join(ROOT_DIR, args.loadmodel)

if model_path is not None:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
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

    print(f"Loaded pretrained model weights from {model_path}.")


def test(ori_image,sparse,pseudo_depth):
        model.eval()

        if args.cuda:
           ori_image = torch.FloatTensor(ori_image).cuda()
           sparse = torch.FloatTensor(sparse).cuda()
           pseudo_depth = torch.FloatTensor(pseudo_depth).cuda()

        ori_image= Variable(ori_image)
        sparse = Variable(sparse)
        pseudo_depth = Variable(pseudo_depth)

        device = "cuda"
        with torch.no_grad():
            dense_depth = model(ori_image, sparse, pseudo_depth, device)

        dense_depth = torch.squeeze(dense_depth)

        return dense_depth.data.cpu().numpy()


def get_transform():
    t_list = [
        transforms.ToTensor()
    ]

    return transforms.Compose(t_list)


def main():
   processed = get_transform()

   image_path = 'demo_image.png'
   sparse_path ='demo_velodyne.png'
   pseudo_depth_path = 'demo_pseudo_depth.png'
   
   ori_image = skimage.io.imread(image_path)
   ori_image = np.reshape(ori_image, [ori_image.shape[0], ori_image.shape[1], 3])
   image = processed(ori_image).numpy()
   image = np.reshape(image, [1, 3, ori_image.shape[0], ori_image.shape[1]])

   sparse = skimage.io.imread(sparse_path).astype(np.float32)
   sparse = sparse * 1.0 / 256.0
   sparse = np.reshape(sparse, [ori_image.shape[0], ori_image.shape[1], 1])
   sparse = processed(sparse).numpy()
   sparse = np.reshape(sparse, [1, 1, ori_image.shape[0], ori_image.shape[1]])

   pseudo_depth = skimage.io.imread(pseudo_depth_path).astype(np.float32)
   pseudo_depth = pseudo_depth * 1.0 / 256.0
   pseudo_depth = np.reshape(pseudo_depth, [ori_image.shape[0], ori_image.shape[1], 1])
   pseudo_depth = processed(pseudo_depth).numpy()
   pseudo_depth = np.reshape(pseudo_depth, [1, 1, ori_image.shape[0], ori_image.shape[1]])

   output_path = "dense_depth_output.png"

   final_dense_depth = test(image, sparse, pseudo_depth)
   final_dense_depth = np.where(final_dense_depth <= 0.0, 0.9, final_dense_depth)

   final_dense_depth = final_dense_depth * 256.0
   final_dense_depth = final_dense_depth.astype('uint16')

   res_buffer = final_dense_depth.tobytes()

   output_image = Image.new("I",final_dense_depth.T.shape)
   output_image.frombytes(res_buffer,'raw',"I;16")
   output_image.save(output_path)

if __name__ == '__main__':
   main()